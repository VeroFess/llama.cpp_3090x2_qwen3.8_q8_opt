#include "qwen38-paged-attention.cuh"

#include "cpy-utils.cuh"

#include <cfloat>

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

    const float scale = ggml_get_op_params_f32(dst, 0);
    const int block_size = ggml_get_op_params_i32(dst, 1);
    const int max_blocks = ggml_get_op_params_i32(dst, 2);
    const int partitions = ggml_get_op_params_i32(dst, 3);
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
        kv_cache->nb[1], kv_cache->nb[2], kv_cache->nb[3], head_dim, n_heads_kv, block_size);

    const size_t shared = ((head_dim + 31) / 32) * sizeof(float);
    float * output = static_cast<float *>(dst->data);
    if (partitions == 1) {
        qwen38_paged_decode_q8_0<<<dim3(n_heads, n_sequences), dim3(head_dim), shared, ctx.stream()>>>(
            static_cast<const float *>(q->data), static_cast<const char *>(kv_cache->data), static_cast<const int32_t *>(block_table->data),
            static_cast<const int32_t *>(context_lens->data), static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data),
            kv_cache->nb[1], kv_cache->nb[2], kv_cache->nb[3], n_heads_kv, block_size, max_blocks, scale, output);
    } else {
        float * scratch = output + ggml_nelements(q);
        qwen38_paged_partial_q8_0<<<dim3(n_heads, n_sequences, q->ne[2] * partitions), dim3(head_dim), shared, ctx.stream()>>>(
            static_cast<const float *>(q->data), static_cast<const char *>(kv_cache->data), static_cast<const int32_t *>(block_table->data),
            static_cast<const int32_t *>(context_lens->data), static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data),
            kv_cache->nb[1], kv_cache->nb[2], kv_cache->nb[3], n_heads_kv, block_size, max_blocks, partitions, q->ne[2], scale, scratch);
        qwen38_paged_reduce_q8_0<<<dim3(n_heads, n_sequences, q->ne[2]), dim3(head_dim), 0, ctx.stream()>>>(
            scratch, static_cast<const int32_t *>(batch_offsets->data), static_cast<const int32_t *>(batch_lens->data), partitions, q->ne[2], output);
    }
}
