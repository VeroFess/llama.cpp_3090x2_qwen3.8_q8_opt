#include "qwen38-paged-attention.cuh"

#include "cpy-utils.cuh"
#include "fattn.cuh"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <mma.h>

using namespace nvcuda;

static __device__ __forceinline__ float qwen38_reduce_sum(float value, float * smem, int tid, int head_dim) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_warps = (head_dim + 31) >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        smem[warp] = value;
    }
    __syncthreads();

    float result = tid < n_warps ? smem[tid] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            result += __shfl_down_sync(0xffffffffu, result, offset);
        }
        if (lane == 0) {
            smem[0] = result;
        }
    }
    __syncthreads();
    return smem[0];
}

static __global__ void qwen38_paged_write_q8_0(
        const float * __restrict__ k_new,
        const float * __restrict__ v_new,
        char * __restrict__ kv_cache,
        const int32_t * __restrict__ write_slots,
        const int32_t * __restrict__ batch_offsets,
        const int32_t * __restrict__ batch_lens,
        size_t stride_token,
        size_t stride_head,
        size_t stride_block,
        int head_dim,
        int n_heads_kv,
        int block_size) {
    const int head = blockIdx.x;
    const int sequence = blockIdx.y;
    const int quant_block = threadIdx.x;
    const int n_quant_blocks = head_dim / QK8_0;
    if (quant_block >= n_quant_blocks) {
        return;
    }

    const int sequence_begin = batch_offsets[sequence];
    const int n_tokens = batch_lens[sequence];
    for (int i = 0; i < n_tokens; ++i) {
        const int token = sequence_begin + i;
        const int slot = write_slots[token];
        const int physical_block = slot / block_size;
        const int token_in_block = slot % block_size;
        const size_t input = (static_cast<size_t>(token) * n_heads_kv + head) * head_dim + quant_block * QK8_0;
        const size_t k_offset = static_cast<size_t>(physical_block) * stride_block + static_cast<size_t>(head) * stride_head + static_cast<size_t>(token_in_block) * stride_token;
        const size_t v_offset = static_cast<size_t>(physical_block) * stride_block + static_cast<size_t>(n_heads_kv + head) * stride_head + static_cast<size_t>(token_in_block) * stride_token;
        auto * k_row = reinterpret_cast<block_q8_0 *>(kv_cache + k_offset);
        auto * v_row = reinterpret_cast<block_q8_0 *>(kv_cache + v_offset);
        quantize_f32_q8_0_block(k_new + input, k_row + quant_block);
        quantize_f32_q8_0_block(v_new + input, v_row + quant_block);
    }
}

static __global__ void qwen38_paged_fill_mask_f16(
        const int32_t * __restrict__ context_lens,
        int context_capacity,
        int n_tokens,
        half * __restrict__ mask) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t elements = (int64_t) context_capacity * n_tokens;
    if (index >= elements) {
        return;
    }
    const int key = index % context_capacity;
    const int query = index / context_capacity;
    const int query_position = context_lens[0] - n_tokens + query;
    mask[index] = key <= query_position ? __float2half(0.0f) : __float2half(-INFINITY);
}

static void qwen38_init_contiguous_tensor(
        ggml_tensor & tensor,
        ggml_type type,
        void * data,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    memset(&tensor, 0, sizeof(tensor));
    tensor.type = type;
    tensor.data = data;
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = ne3;
    tensor.nb[0] = ggml_type_size(type);
    tensor.nb[1] = ggml_row_size(type, ne0);
    tensor.nb[2] = tensor.nb[1] * ne1;
    tensor.nb[3] = tensor.nb[2] * ne2;
}

static void qwen38_paged_flash_prefill(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_tensor * scratch,
        const ggml_tensor * q,
        const ggml_tensor * kv_cache,
        const ggml_tensor * context_lens,
        int n_heads_kv,
        int context_tokens,
        int contiguous_start,
        float scale) {
    const int head_dim = q->ne[0];
    const int n_heads = q->ne[1];
    const int n_tokens = q->ne[2];
    const int context_capacity = GGML_PAD(context_tokens, 256);
    half * mask_data = static_cast<half *>(scratch->data);
    GGML_ASSERT((size_t) context_capacity * n_tokens * sizeof(half) <= ggml_nbytes(scratch));

    const int64_t mask_elements = (int64_t) context_capacity * n_tokens;
    qwen38_paged_fill_mask_f16<<<(mask_elements + 255) / 256, 256, 0, ctx.stream()>>>(
        static_cast<const int32_t *>(context_lens->data), context_capacity, n_tokens, mask_data);

    ggml_tensor q_view = *q;
    q_view.ne[0] = head_dim;
    q_view.ne[1] = n_tokens;
    q_view.ne[2] = n_heads;
    q_view.ne[3] = 1;
    q_view.nb[0] = q->nb[0];
    q_view.nb[1] = q->nb[2];
    q_view.nb[2] = q->nb[1];
    q_view.nb[3] = q->nb[3];

    ggml_tensor k_view;
    ggml_tensor v_view;
    ggml_tensor mask_view;
    ggml_tensor output_view;
    GGML_ASSERT(contiguous_start >= 0);
    char * cache_data = static_cast<char *>(kv_cache->data);
    qwen38_init_contiguous_tensor(k_view, GGML_TYPE_Q8_0, cache_data + (size_t) contiguous_start * kv_cache->nb[2], head_dim, context_capacity, n_heads_kv, 1);
    qwen38_init_contiguous_tensor(v_view, GGML_TYPE_Q8_0, cache_data + (size_t) n_heads_kv * kv_cache->nb[3] + (size_t) contiguous_start * kv_cache->nb[2], head_dim, context_capacity, n_heads_kv, 1);
    k_view.nb[2] = kv_cache->nb[3];
    k_view.nb[3] = k_view.nb[2] * n_heads_kv;
    v_view.nb[2] = kv_cache->nb[3];
    v_view.nb[3] = v_view.nb[2] * n_heads_kv;
    qwen38_init_contiguous_tensor(mask_view, GGML_TYPE_F16, mask_data, context_capacity, n_tokens, 1, 1);
    qwen38_init_contiguous_tensor(output_view, GGML_TYPE_F32, dst->data, head_dim, n_heads, n_tokens, 1);
    output_view.op = GGML_OP_FLASH_ATTN_EXT;
    output_view.src[0] = &q_view;
    output_view.src[1] = &k_view;
    output_view.src[2] = &v_view;
    output_view.src[3] = &mask_view;
    const float params[] = { scale, 0.0f, 0.0f };
    ggml_set_op_params(&output_view, params, sizeof(params));
    ggml_flash_attn_ext_set_prec(&output_view, GGML_PREC_F32);
    ggml_cuda_flash_attn_ext_mma_q8_0_256(ctx, &output_view);
}

static __global__ void qwen38_paged_decode_q8_0(
        const float * __restrict__ q,
        const char * __restrict__ kv_cache,
        const int32_t * __restrict__ block_table,
        const int32_t * __restrict__ context_lens,
        const int32_t * __restrict__ batch_offsets,
        const int32_t * __restrict__ batch_lens,
        size_t stride_token,
        size_t stride_head,
        size_t stride_block,
        int n_heads_kv,
        int block_size,
        int max_blocks,
        float scale,
        float * __restrict__ output) {
    extern __shared__ float smem[];

    const int head = blockIdx.x;
    const int sequence = blockIdx.y;
    const int tid = threadIdx.x;
    const int n_heads = gridDim.x;
    const int head_dim = blockDim.x;
    const int kv_head = head / (n_heads / n_heads_kv);
    const int sequence_begin = batch_offsets[sequence];
    const int n_new_tokens = batch_lens[sequence];

    for (int i = 0; i < n_new_tokens; ++i) {
        const int token = sequence_begin + i;
        const int query_position = context_lens[sequence] - n_new_tokens + i;
        const int n_blocks = query_position / block_size + 1;
        const float q_value = q[(static_cast<size_t>(token) * n_heads + head) * head_dim + tid] * scale;

        float maximum = -FLT_MAX;
        float sum = 0.0f;
        float accumulator = 0.0f;
        for (int logical_block = 0; logical_block < n_blocks; ++logical_block) {
            const int physical_block = block_table[sequence * max_blocks + logical_block];
            const int begin = logical_block * block_size;
            const int end = min(begin + block_size, query_position + 1);
            for (int position = begin; position < end; ++position) {
                const int token_in_block = position % block_size;
                const size_t k_offset = static_cast<size_t>(physical_block) * stride_block + static_cast<size_t>(kv_head) * stride_head + static_cast<size_t>(token_in_block) * stride_token;
                const size_t v_offset = static_cast<size_t>(physical_block) * stride_block + static_cast<size_t>(n_heads_kv + kv_head) * stride_head + static_cast<size_t>(token_in_block) * stride_token;
                const auto * k_row = reinterpret_cast<const block_q8_0 *>(kv_cache + k_offset);
                const auto * v_row = reinterpret_cast<const block_q8_0 *>(kv_cache + v_offset);
                const int quant_block = tid / QK8_0;
                const int quant_index = tid % QK8_0;
                const float k_value = static_cast<float>(k_row[quant_block].d) * k_row[quant_block].qs[quant_index];
                const float v_value = static_cast<float>(v_row[quant_block].d) * v_row[quant_block].qs[quant_index];
                const float score = qwen38_reduce_sum(q_value * k_value, smem, tid, head_dim);
                const float next_maximum = fmaxf(maximum, score);
                const float old_scale = __expf(maximum - next_maximum);
                const float new_scale = __expf(score - next_maximum);
                sum = sum * old_scale + new_scale;
                accumulator = accumulator * old_scale + new_scale * v_value;
                maximum = next_maximum;
            }
        }

        const size_t output_index = (static_cast<size_t>(token) * n_heads + head) * head_dim + tid;
        output[output_index] = accumulator / (sum + 1e-6f);
    }
}

static __global__ void qwen38_paged_prefill_q8_0(
        const float * __restrict__ q,
        const char * __restrict__ kv_cache,
        const int32_t * __restrict__ block_table,
        const int32_t * __restrict__ context_lens,
        const int32_t * __restrict__ batch_offsets,
        const int32_t * __restrict__ batch_lens,
        size_t stride_token,
        size_t stride_head,
        size_t stride_block,
        int n_heads_kv,
        int max_blocks,
        float scale,
        float * __restrict__ output) {
    constexpr int tile_q = 32;
    constexpr int tile_k = 16;
    constexpr int head_dim = 256;
    __shared__ half sq[tile_q * head_dim];
    __shared__ half sk[tile_k * head_dim];
    __shared__ half sv[tile_k * head_dim];
    __shared__ float scores[tile_q * tile_k];
    __shared__ float maximum[tile_q];
    __shared__ float denominator[tile_q];
    __shared__ float alpha[tile_q];

    const int head = blockIdx.x;
    const int sequence = blockIdx.y;
    const int query_tile = blockIdx.z;
    const int tid = threadIdx.x;
    const int n_heads = gridDim.x;
    const int kv_head = head / (n_heads / n_heads_kv);
    const int sequence_begin = batch_offsets[sequence];
    const int sequence_tokens = batch_lens[sequence];
    const int query_begin = query_tile * tile_q;
    if (query_begin >= sequence_tokens) {
        return;
    }

    for (int index = tid; index < tile_q * head_dim; index += blockDim.x) {
        const int row = index / head_dim;
        const int dim = index % head_dim;
        const int token = sequence_begin + query_begin + row;
        const float value = query_begin + row < sequence_tokens ?
            q[((int64_t) token * n_heads + head) * head_dim + dim] * scale : 0.0f;
        sq[index] = __float2half(value);
    }
    if (tid < tile_q) {
        maximum[tid] = -FLT_MAX;
        denominator[tid] = 0.0f;
    }
    __syncthreads();

    float accumulator[tile_q];
#pragma unroll
    for (int row = 0; row < tile_q; ++row) {
        accumulator[row] = 0.0f;
    }

    const int last_query_position = context_lens[sequence] - sequence_tokens + min(query_begin + tile_q, sequence_tokens) - 1;
    const int n_pages = last_query_position / tile_k + 1;
    for (int page = 0; page < n_pages; ++page) {
        const int physical_page = block_table[sequence * max_blocks + page];
        for (int index = tid; index < tile_k * head_dim; index += blockDim.x) {
            const int token_in_page = index / head_dim;
            const int dim = index % head_dim;
            const int quant_block = dim / QK8_0;
            const int quant_index = dim % QK8_0;
            const size_t k_offset = (size_t) physical_page * stride_block + (size_t) kv_head * stride_head + (size_t) token_in_page * stride_token;
            const size_t v_offset = (size_t) physical_page * stride_block + (size_t) (n_heads_kv + kv_head) * stride_head + (size_t) token_in_page * stride_token;
            const auto * k_row = reinterpret_cast<const block_q8_0 *>(kv_cache + k_offset);
            const auto * v_row = reinterpret_cast<const block_q8_0 *>(kv_cache + v_offset);
            sk[index] = __float2half((float) k_row[quant_block].d * k_row[quant_block].qs[quant_index]);
            sv[index] = __float2half((float) v_row[quant_block].d * v_row[quant_block].qs[quant_index]);
        }
        __syncthreads();

        if (tid < 64) {
            const int query_warp = tid / 32;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
            for (int dim = 0; dim < head_dim; dim += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::load_matrix_sync(a, sq + query_warp * 16 * head_dim + dim, head_dim);
                wmma::load_matrix_sync(b, sk + dim, head_dim);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + query_warp * 16 * tile_k, c, tile_k, wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < tile_q) {
            const int row = tid;
            const int query_row = query_begin + row;
            const int query_position = context_lens[sequence] - sequence_tokens + query_row;
            float page_maximum = -FLT_MAX;
            if (query_row < sequence_tokens) {
                for (int col = 0; col < tile_k; ++col) {
                    const int key_position = page * tile_k + col;
                    if (key_position <= query_position) {
                        page_maximum = fmaxf(page_maximum, scores[row * tile_k + col]);
                    }
                }
            }
            if (page_maximum == -FLT_MAX) {
                alpha[row] = 1.0f;
                for (int col = 0; col < tile_k; ++col) {
                    scores[row * tile_k + col] = 0.0f;
                }
            } else {
                const float next_maximum = fmaxf(maximum[row], page_maximum);
                const float old_scale = maximum[row] == -FLT_MAX ? 0.0f : __expf(maximum[row] - next_maximum);
                float page_sum = 0.0f;
                for (int col = 0; col < tile_k; ++col) {
                    const int key_position = page * tile_k + col;
                    const float weight = key_position <= query_position ? __expf(scores[row * tile_k + col] - next_maximum) : 0.0f;
                    scores[row * tile_k + col] = weight;
                    page_sum += weight;
                }
                alpha[row] = old_scale;
                denominator[row] = denominator[row] * old_scale + page_sum;
                maximum[row] = next_maximum;
            }
        }
        __syncthreads();

#pragma unroll
        for (int row = 0; row < tile_q; ++row) {
            float value = accumulator[row] * alpha[row];
#pragma unroll
            for (int col = 0; col < tile_k; ++col) {
                value += scores[row * tile_k + col] * __half2float(sv[col * head_dim + tid]);
            }
            accumulator[row] = value;
        }
        __syncthreads();
    }

    for (int row = 0; row < tile_q; ++row) {
        if (query_begin + row < sequence_tokens) {
            const int token = sequence_begin + query_begin + row;
            output[((int64_t) token * n_heads + head) * head_dim + tid] = accumulator[row] / (denominator[row] + 1e-6f);
        }
    }
}

static __global__ void qwen38_paged_partial_q8_0(
        const float * __restrict__ q,
        const char * __restrict__ kv_cache,
        const int32_t * __restrict__ block_table,
        const int32_t * __restrict__ context_lens,
        const int32_t * __restrict__ batch_offsets,
        const int32_t * __restrict__ batch_lens,
        size_t stride_token,
        size_t stride_head,
        size_t stride_block,
        int n_heads_kv,
        int block_size,
        int max_blocks,
        int partitions,
        int n_tokens,
        float scale,
        float * __restrict__ scratch) {
    extern __shared__ float smem[];

    const int head = blockIdx.x;
    const int sequence = blockIdx.y;
    const int token = blockIdx.z / partitions;
    const int partition = blockIdx.z % partitions;
    const int sequence_begin = batch_offsets[sequence];
    const int sequence_tokens = batch_lens[sequence];
    if (token < sequence_begin || token >= sequence_begin + sequence_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int n_heads = gridDim.x;
    const int head_dim = blockDim.x;
    const int kv_head = head / (n_heads / n_heads_kv);
    const int query_position = context_lens[sequence] - sequence_tokens + token - sequence_begin;
    const int tokens_per_partition = (query_position + 1 + partitions - 1) / partitions;
    const int begin = partition * tokens_per_partition;
    const int end = min(begin + tokens_per_partition, query_position + 1);
    const size_t scratch_base = (((static_cast<size_t>(partition) * n_tokens + token) * n_heads + head) * (head_dim + 2));

    float maximum = -FLT_MAX;
    float sum = 0.0f;
    float accumulator = 0.0f;
    const float q_value = q[(static_cast<size_t>(token) * n_heads + head) * head_dim + tid] * scale;
    for (int position = begin; position < end; ++position) {
        const int logical_block = position / block_size;
        const int token_in_block = position % block_size;
        const int physical_block = block_table[sequence * max_blocks + logical_block];
        const size_t k_offset = static_cast<size_t>(physical_block) * stride_block + static_cast<size_t>(kv_head) * stride_head + static_cast<size_t>(token_in_block) * stride_token;
        const size_t v_offset = static_cast<size_t>(physical_block) * stride_block + static_cast<size_t>(n_heads_kv + kv_head) * stride_head + static_cast<size_t>(token_in_block) * stride_token;
        const auto * k_row = reinterpret_cast<const block_q8_0 *>(kv_cache + k_offset);
        const auto * v_row = reinterpret_cast<const block_q8_0 *>(kv_cache + v_offset);
        const int quant_block = tid / QK8_0;
        const int quant_index = tid % QK8_0;
        const float k_value = static_cast<float>(k_row[quant_block].d) * k_row[quant_block].qs[quant_index];
        const float v_value = static_cast<float>(v_row[quant_block].d) * v_row[quant_block].qs[quant_index];
        const float score = qwen38_reduce_sum(q_value * k_value, smem, tid, head_dim);
        const float next_maximum = fmaxf(maximum, score);
        const float old_scale = __expf(maximum - next_maximum);
        const float new_scale = __expf(score - next_maximum);
        sum = sum * old_scale + new_scale;
        accumulator = accumulator * old_scale + new_scale * v_value;
        maximum = next_maximum;
    }

    if (tid == 0) {
        scratch[scratch_base] = maximum;
        scratch[scratch_base + 1] = sum;
    }
    scratch[scratch_base + 2 + tid] = accumulator;
}

static __global__ void qwen38_paged_reduce_q8_0(
        const float * __restrict__ scratch,
        const int32_t * __restrict__ batch_offsets,
        const int32_t * __restrict__ batch_lens,
        int partitions,
        int n_tokens,
        float * __restrict__ output) {
    const int head = blockIdx.x;
    const int sequence = blockIdx.y;
    const int token = blockIdx.z;
    const int sequence_begin = batch_offsets[sequence];
    const int sequence_tokens = batch_lens[sequence];
    if (token < sequence_begin || token >= sequence_begin + sequence_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int n_heads = gridDim.x;
    const int head_dim = blockDim.x;
    float maximum = -FLT_MAX;
    float sum = 0.0f;
    float accumulator = 0.0f;
    for (int partition = 0; partition < partitions; ++partition) {
        const size_t base = (((static_cast<size_t>(partition) * n_tokens + token) * n_heads + head) * (head_dim + 2));
        const float partial_sum = scratch[base + 1];
        if (partial_sum == 0.0f) {
            continue;
        }
        const float partial_maximum = scratch[base];
        const float next_maximum = fmaxf(maximum, partial_maximum);
        const float old_scale = __expf(maximum - next_maximum);
        const float new_scale = __expf(partial_maximum - next_maximum);
        accumulator = accumulator * old_scale + scratch[base + 2 + tid] * new_scale;
        sum = sum * old_scale + partial_sum * new_scale;
        maximum = next_maximum;
    }
    output[(static_cast<size_t>(token) * n_heads + head) * head_dim + tid] = accumulator / (sum + 1e-6f);
}

void ggml_cuda_op_qwen38_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k_new = dst->src[1];
    const ggml_tensor * v_new = dst->src[2];
    const ggml_tensor * kv_cache = dst->src[3];
    const ggml_tensor * block_table = dst->src[4];
    const ggml_tensor * write_slots = dst->src[5];
    const ggml_tensor * context_lens = dst->src[6];
    const ggml_tensor * batch_offsets = dst->src[7];
    const ggml_tensor * batch_lens = dst->src[8];
    ggml_tensor * scratch_tensor = dst->src[9];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const int block_size = ggml_get_op_params_i32(dst, 1);
    const int max_blocks = ggml_get_op_params_i32(dst, 2);
    const int partitions = ggml_get_op_params_i32(dst, 3);
    const int context_tokens = ggml_get_op_params_i32(dst, 4);
    const int contiguous_start = ggml_get_op_params_i32(dst, 5);
    const int head_dim = q->ne[0];
    const int n_heads = q->ne[1];
    const int n_heads_kv = k_new->ne[1];
    const int n_sequences = batch_lens->ne[0];

    GGML_ASSERT(q->type == GGML_TYPE_F32 && k_new->type == GGML_TYPE_F32 && v_new->type == GGML_TYPE_F32);
    GGML_ASSERT(kv_cache->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(head_dim > 0 && head_dim <= 1024 && head_dim % QK8_0 == 0);
    GGML_ASSERT(n_heads > 0 && n_heads_kv > 0 && n_heads % n_heads_kv == 0);
    GGML_ASSERT(partitions > 0);

    qwen38_paged_write_q8_0<<<dim3(n_heads_kv, n_sequences), dim3(head_dim / QK8_0), 0, ctx.stream()>>>(
        static_cast<const float *>(k_new->data), static_cast<const float *>(v_new->data), static_cast<char *>(kv_cache->data),
        static_cast<const int32_t *>(write_slots->data), static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data),
        kv_cache->nb[1], kv_cache->nb[3], kv_cache->nb[2], head_dim, n_heads_kv, block_size);

    const size_t shared = ((head_dim + 31) / 32) * sizeof(float);
    float * output = static_cast<float *>(dst->data);
    static const bool flash_prefill = []() {
        const char * value = getenv("GGML_CUDA_QWEN38_PAGED_FLASH_PREFILL");
        return value == nullptr || atoi(value) != 0;
    }();
    if (flash_prefill && contiguous_start >= 0 && n_sequences == 1 && head_dim == 256 && block_size == 16) {
        qwen38_paged_flash_prefill(ctx, dst, scratch_tensor, q, kv_cache, context_lens, n_heads_kv, context_tokens, contiguous_start, scale);
        return;
    }
    static const int prefill_min_tokens = []() {
        const char * value = getenv("GGML_CUDA_QWEN38_PAGED_PREFILL_MIN_TOKENS");
        return value == nullptr ? 64 : std::max(1, atoi(value));
    }();
    if (head_dim == 256 && block_size == 16 && q->ne[2] >= prefill_min_tokens) {
        const int query_tiles = (q->ne[2] + 31) / 32;
        qwen38_paged_prefill_q8_0<<<dim3(n_heads, n_sequences, query_tiles), 256, 0, ctx.stream()>>>(
            static_cast<const float *>(q->data), static_cast<const char *>(kv_cache->data), static_cast<const int32_t *>(block_table->data),
            static_cast<const int32_t *>(context_lens->data), static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data),
            kv_cache->nb[1], kv_cache->nb[3], kv_cache->nb[2], n_heads_kv, max_blocks, scale, output);
        return;
    }
    if (partitions == 1) {
        qwen38_paged_decode_q8_0<<<dim3(n_heads, n_sequences), dim3(head_dim), shared, ctx.stream()>>>(
            static_cast<const float *>(q->data), static_cast<const char *>(kv_cache->data), static_cast<const int32_t *>(block_table->data),
            static_cast<const int32_t *>(context_lens->data), static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data),
            kv_cache->nb[1], kv_cache->nb[3], kv_cache->nb[2], n_heads_kv, block_size, max_blocks, scale, output);
    } else {
        float * scratch = static_cast<float *>(scratch_tensor->data);
        qwen38_paged_partial_q8_0<<<dim3(n_heads, n_sequences, q->ne[2] * partitions), dim3(head_dim), shared, ctx.stream()>>>(
            static_cast<const float *>(q->data), static_cast<const char *>(kv_cache->data), static_cast<const int32_t *>(block_table->data),
            static_cast<const int32_t *>(context_lens->data), static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data),
            kv_cache->nb[1], kv_cache->nb[3], kv_cache->nb[2], n_heads_kv, block_size, max_blocks, partitions, q->ne[2], scale, scratch);
        qwen38_paged_reduce_q8_0<<<dim3(n_heads, n_sequences, q->ne[2]), dim3(head_dim), 0, ctx.stream()>>>(
            scratch, static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data), partitions, q->ne[2], output);
    }
}
