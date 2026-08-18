#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstring>

static __global__ void gated_delta_net_wy_a_cuda(
        const float * __restrict__ k,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        float * __restrict__ A) {
    constexpr int S = 128;
    constexpr int T = 64;
    __shared__ float sk[T * S];
    __shared__ float sg[T];
    __shared__ float sb[T];

    const int group = blockIdx.x;
    const float * kg = k + (int64_t) group * T * S;
    const float * gg = g + (int64_t) group * T;
    const float * bg = beta + (int64_t) group * T;
    float * Ag = A + (int64_t) group * T * T;

    for (int i = threadIdx.x; i < T * S; i += blockDim.x) {
        sk[i] = kg[i];
    }
    if (threadIdx.x < T) {
        sg[threadIdx.x] = gg[threadIdx.x];
        sb[threadIdx.x] = bg[threadIdx.x];
    }
    __syncthreads();

    for (int index = threadIdx.x; index < T * T; index += blockDim.x) {
        const int i = index / T;
        const int j = index % T;
        float value = 0.0f;
        if (i > j) {
#pragma unroll 4
            for (int d = 0; d < S; ++d) {
                value += sk[i * S + d] * sk[j * S + d];
            }
            value *= sb[i] * __expf(sg[i] - sg[j]);
        }
        Ag[index] = value;
    }
}

static __global__ void gated_delta_net_wy_solve_cuda(
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ A,
        float * __restrict__ W,
        float * __restrict__ U) {
    constexpr int S = 128;
    constexpr int T = 64;
    __shared__ float sA[T * T];
    __shared__ float sy[T * S];

    const int group = blockIdx.x;
    const int kind = blockIdx.y;
    const int d = threadIdx.x;
    const float * Ag = A + (int64_t) group * T * T;
    for (int i = threadIdx.x; i < T * T; i += blockDim.x) {
        sA[i] = Ag[i];
    }
    __syncthreads();

    const float * input = kind == 0 ? k : v;
    const float * ig = input + (int64_t) group * T * S;
    const float * gg = g + (int64_t) group * T;
    const float * bg = beta + (int64_t) group * T;
    float * output = (kind == 0 ? W : U) + (int64_t) group * T * S;

    for (int i = 0; i < T; ++i) {
        float value = bg[i] * ig[i * S + d];
        if (kind == 0) {
            value *= __expf(gg[i]);
        }
        for (int j = 0; j < i; ++j) {
            value -= sA[i * T + j] * sy[j * S + d];
        }
        sy[i * S + d] = value;
        output[i * S + d] = value;
    }
}

static __global__ void gated_delta_net_chunk_copy_cuda(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int matrix_elements,
        int n_chunks,
        int chunk) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= matrix_elements) {
        return;
    }
    const int head = blockIdx.y;
    dst[(int64_t) head * matrix_elements + index] = src[(int64_t) head * matrix_elements * n_chunks + (int64_t) chunk * matrix_elements + index];
}

static __global__ void gated_delta_net_chunk_scale_state_cuda(
        float * __restrict__ state,
        const float * __restrict__ scale,
        int state_elements,
        int n_chunks,
        int chunk) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= state_elements) {
        return;
    }
    const int head = blockIdx.y;
    state[(int64_t) head * state_elements + index] *= scale[(int64_t) head * n_chunks + chunk];
}

static __global__ void gated_delta_net_chunk_pack_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        float * __restrict__ q_pack,
        float * __restrict__ k_pack,
        float * __restrict__ v_pack,
        float * __restrict__ g_cumsum,
        float * __restrict__ beta_pack,
        int S,
        int T,
        int n_tokens,
        int n_chunks,
        int H_k,
        int H_v) {
    const int group = blockIdx.x;
    const int head = group / n_chunks;
    const int chunk = group % n_chunks;
    const int sequence = head / H_v;
    const int v_head = head % H_v;
    const int k_head = v_head % H_k;
    const int begin = chunk * T;
    const int64_t st = (int64_t) S * T;
    const float scale = rsqrtf((float) S);

    for (int index = threadIdx.x; index < S * T; index += blockDim.x) {
        const int token_in_chunk = index / S;
        const int dim = index % S;
        const int token = begin + token_in_chunk;
        float qv = 0.0f;
        float kv = 0.0f;
        float vv = 0.0f;
        if (token < n_tokens) {
            qv = q[((int64_t) sequence * n_tokens * H_k + (int64_t) token * H_k + k_head) * S + dim] * scale;
            kv = k[((int64_t) sequence * n_tokens * H_k + (int64_t) token * H_k + k_head) * S + dim];
            vv = v[((int64_t) sequence * n_tokens * H_v + (int64_t) token * H_v + v_head) * S + dim];
        }
        q_pack[(int64_t) group * st + index] = qv;
        k_pack[(int64_t) group * st + index] = kv;
        v_pack[(int64_t) group * st + index] = vv;
    }
    if (threadIdx.x == 0) {
        float cumulative = 0.0f;
        for (int token_in_chunk = 0; token_in_chunk < T; ++token_in_chunk) {
            const int token = begin + token_in_chunk;
            float bv = 0.0f;
            if (token < n_tokens) {
                const int64_t index = (int64_t) sequence * n_tokens * H_v + (int64_t) token * H_v + v_head;
                cumulative += g[index];
                bv = beta[index];
            }
            g_cumsum[(int64_t) group * T + token_in_chunk] = cumulative;
            beta_pack[(int64_t) group * T + token_in_chunk] = bv;
        }
    }
}

static __global__ void gated_delta_net_chunk_transform_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ u,
        const float * __restrict__ g,
        float * __restrict__ q_g,
        float * __restrict__ kg_t,
        float * __restrict__ v_t,
        float * __restrict__ g_last,
        int S,
        int T) {
    const int group = blockIdx.x;
    const int64_t st = (int64_t) S * T;
    const float last = g[(int64_t) group * T + T - 1];
    if (threadIdx.x == 0) {
        g_last[group] = __expf(last);
    }
    for (int index = threadIdx.x; index < S * T; index += blockDim.x) {
        const int token = index / S;
        const int dim = index % S;
        const float gt = g[(int64_t) group * T + token];
        q_g[(int64_t) group * st + index] = q[(int64_t) group * st + index] * __expf(gt);
        kg_t[(int64_t) group * st + (int64_t) dim * T + token] = k[(int64_t) group * st + index] * __expf(last - gt);
        v_t[(int64_t) group * st + (int64_t) dim * T + token] = u[(int64_t) group * st + index];
    }
}

static __global__ void gated_delta_net_chunk_mask_kq_cuda(
        float * __restrict__ kq,
        const float * __restrict__ g,
        int T) {
    const int group = blockIdx.x;
    for (int index = threadIdx.x; index < T * T; index += blockDim.x) {
        const int key = index % T;
        const int query = index / T;
        float value = 0.0f;
        if (key <= query) {
            value = kq[(int64_t) group * T * T + key + query * T] * __expf(g[(int64_t) group * T + query] - g[(int64_t) group * T + key]);
        }
        kq[(int64_t) group * T * T + key + query * T] = value;
    }
}

static void gated_delta_net_init_tensor(
        ggml_tensor & tensor,
        void * data,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    memset(&tensor, 0, sizeof(tensor));
    tensor.type = GGML_TYPE_F32;
    tensor.data = data;
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = ne3;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = ne0 * sizeof(float);
    tensor.nb[2] = tensor.nb[1] * ne1;
    tensor.nb[3] = tensor.nb[2] * ne2;
}

static void gated_delta_net_chunk_gemm(
        cublasHandle_t handle,
        cublasOperation_t trans_a,
        cublasOperation_t trans_b,
        int m,
        int n,
        int k,
        float alpha,
        const float * a,
        int lda,
        int64_t stride_a,
        const float * b,
        int ldb,
        int64_t stride_b,
        float beta,
        float * c,
        int ldc,
        int64_t stride_c,
        int batch_count) {
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, trans_a, trans_b, m, n, k,
        &alpha, a, CUDA_R_32F, lda, stride_a,
        b, CUDA_R_32F, ldb, stride_b,
        &beta, c, CUDA_R_32F, ldc, stride_c,
        batch_count, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void gated_delta_net_chunk_scan_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_cd = dst->src[0];
    const ggml_tensor * v_t = dst->src[1];
    const ggml_tensor * kq = dst->src[2];
    const ggml_tensor * q_g = dst->src[3];
    const ggml_tensor * kg_t = dst->src[4];
    const ggml_tensor * g_last = dst->src[5];
    const ggml_tensor * state_in = dst->src[6];

    const int S = k_cd->ne[0];
    const int T = k_cd->ne[1];
    const int n_chunks = k_cd->ne[2];
    const int H = k_cd->ne[3];
    const int output_matrix = S * T;
    const int state_matrix = S * S;
    const int vprime_matrix = T * S;
    float * output = static_cast<float *>(dst->data);
    float * state = output + (int64_t) output_matrix * n_chunks * H;
    float * vprime = state + (int64_t) state_matrix * H;
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemcpyAsync(state, state_in->data, (size_t) state_matrix * H * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    const float * k_cd_data = static_cast<const float *>(k_cd->data);
    const float * v_t_data = static_cast<const float *>(v_t->data);
    const float * kq_data = static_cast<const float *>(kq->data);
    const float * q_g_data = static_cast<const float *>(q_g->data);
    const float * kg_t_data = static_cast<const float *>(kg_t->data);
    const float * g_last_data = static_cast<const float *>(g_last->data);
    constexpr int threads = 256;

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        gated_delta_net_chunk_copy_cuda<<<dim3((vprime_matrix + threads - 1) / threads, H), threads, 0, stream>>>(
            v_t_data, vprime, vprime_matrix, n_chunks, chunk);
        gated_delta_net_chunk_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            T, S, S, -1.0f,
            k_cd_data + (int64_t) chunk * output_matrix, S, (int64_t) output_matrix * n_chunks,
            state, S, state_matrix,
            1.0f, vprime, T, vprime_matrix, H);

        float * output_chunk = output + (int64_t) chunk * output_matrix;
        gated_delta_net_chunk_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            S, T, S, 1.0f,
            state, S, state_matrix,
            q_g_data + (int64_t) chunk * output_matrix, S, (int64_t) output_matrix * n_chunks,
            0.0f, output_chunk, S, (int64_t) output_matrix * n_chunks, H);
        gated_delta_net_chunk_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            S, T, T, 1.0f,
            vprime, T, vprime_matrix,
            kq_data + (int64_t) chunk * T * T, T, (int64_t) T * T * n_chunks,
            1.0f, output_chunk, S, (int64_t) output_matrix * n_chunks, H);

        gated_delta_net_chunk_scale_state_cuda<<<dim3((state_matrix + threads - 1) / threads, H), threads, 0, stream>>>(
            state, g_last_data, state_matrix, n_chunks, chunk);
        gated_delta_net_chunk_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            S, S, T, 1.0f,
            kg_t_data + (int64_t) chunk * vprime_matrix, T, (int64_t) vprime_matrix * n_chunks,
            vprime, T, vprime_matrix,
            1.0f, state, S, state_matrix, H);
    }
}

static void gated_delta_net_chunk_fused_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    const ggml_tensor * beta = dst->src[4];
    const ggml_tensor * state_in = dst->src[5];
    const int S = q->ne[0];
    const int H_k = q->ne[1];
    const int n_tokens = q->ne[2];
    const int n_sequences = q->ne[3];
    const int H_v = v->ne[1];
    const int H = H_v * n_sequences;
    const int T = ggml_get_op_params_i32(dst, 2);
    const int n_chunks = (n_tokens + T - 1) / T;
    const int groups = H * n_chunks;
    const int64_t st = (int64_t) S * T;
    const int64_t tt = (int64_t) T * T;
    const int64_t output_elements = groups * st;
    const int64_t state_elements = (int64_t) H * S * S;
    const int64_t scan_scratch = (int64_t) H * st;
    float * cursor = static_cast<float *>(dst->data) + output_elements + state_elements + scan_scratch;
    float * q_pack = cursor; cursor += groups * st;
    float * k_pack = cursor; cursor += groups * st;
    float * v_pack = cursor; cursor += groups * st;
    float * g_pack = cursor; cursor += (int64_t) groups * T;
    float * b_pack = cursor; cursor += (int64_t) groups * T;
    float * A = cursor; cursor += groups * tt;
    float * W = cursor; cursor += groups * st;
    float * U = cursor; cursor += groups * st;
    float * KQ = cursor; cursor += groups * tt;
    float * QG = cursor; cursor += groups * st;
    float * KGT = cursor; cursor += groups * st;
    float * VT = cursor; cursor += groups * st;
    float * GLast = cursor; cursor += groups;
    GGML_ASSERT(cursor <= static_cast<float *>(dst->data) + ggml_nelements(dst));

    cudaStream_t stream = ctx.stream();
    gated_delta_net_chunk_pack_cuda<<<groups, 256, 0, stream>>>(
        static_cast<const float *>(q->data), static_cast<const float *>(k->data), static_cast<const float *>(v->data),
        static_cast<const float *>(g->data), static_cast<const float *>(beta->data),
        q_pack, k_pack, v_pack, g_pack, b_pack, S, T, n_tokens, n_chunks, H_k, H_v);
    gated_delta_net_wy_a_cuda<<<groups, 256, 0, stream>>>(k_pack, g_pack, b_pack, A);
    gated_delta_net_wy_solve_cuda<<<dim3(groups, 2), 128, 0, stream>>>(k_pack, v_pack, g_pack, b_pack, A, W, U);

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    gated_delta_net_chunk_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        T, T, S, 1.0f,
        k_pack, S, st,
        q_pack, S, st,
        0.0f, KQ, T, tt, groups);
    gated_delta_net_chunk_mask_kq_cuda<<<groups, 256, 0, stream>>>(KQ, g_pack, T);
    gated_delta_net_chunk_transform_cuda<<<groups, 256, 0, stream>>>(
        q_pack, k_pack, U, g_pack, QG, KGT, VT, GLast, S, T);

    ggml_tensor kcd_tensor;
    ggml_tensor vt_tensor;
    ggml_tensor kq_tensor;
    ggml_tensor qg_tensor;
    ggml_tensor kgt_tensor;
    ggml_tensor glast_tensor;
    ggml_tensor state_tensor = *state_in;
    ggml_tensor scan_dst = *dst;
    gated_delta_net_init_tensor(kcd_tensor, W, S, T, n_chunks, H);
    gated_delta_net_init_tensor(vt_tensor, VT, T, S, n_chunks, H);
    gated_delta_net_init_tensor(kq_tensor, KQ, T, T, n_chunks, H);
    gated_delta_net_init_tensor(qg_tensor, QG, S, T, n_chunks, H);
    gated_delta_net_init_tensor(kgt_tensor, KGT, T, S, n_chunks, H);
    gated_delta_net_init_tensor(glast_tensor, GLast, 1, 1, n_chunks, H);
    scan_dst.src[0] = &kcd_tensor;
    scan_dst.src[1] = &vt_tensor;
    scan_dst.src[2] = &kq_tensor;
    scan_dst.src[3] = &qg_tensor;
    scan_dst.src[4] = &kgt_tensor;
    scan_dst.src[5] = &glast_tensor;
    scan_dst.src[6] = &state_tensor;
    gated_delta_net_chunk_scan_cuda(ctx, &scan_dst);
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    if (ggml_get_op_params_i32(dst, 1) == 3) {
        GGML_ASSERT(cache == nullptr);
        gated_delta_net_chunk_fused_cuda(ctx, dst);
        return;
    }
    if (ggml_get_op_params_i32(dst, 1) == 1) {
        const ggml_tensor * src_k = dst->src[0];
        const ggml_tensor * src_v = dst->src[1];
        const ggml_tensor * src_g = dst->src[2];
        const ggml_tensor * src_b = dst->src[3];
        GGML_ASSERT(src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32);
        GGML_ASSERT(src_g->type == GGML_TYPE_F32 && src_b->type == GGML_TYPE_F32);
        GGML_ASSERT(src_k->ne[0] == 128 && src_k->ne[1] == 64);
        GGML_ASSERT(ggml_are_same_shape(src_k, src_v));
        const int64_t n_groups = src_k->ne[2] * src_k->ne[3];
        const int64_t a_elements = n_groups * 64 * 64;
        const int64_t wu_elements = n_groups * 64 * 128;
        float * A = (float *) dst->data;
        float * W = A + a_elements;
        float * U = W + wu_elements;
        cudaStream_t stream = ctx.stream();
        gated_delta_net_wy_a_cuda<<<n_groups, 256, 0, stream>>>(
            (const float *) src_k->data, (const float *) src_g->data, (const float *) src_b->data, A);
        gated_delta_net_wy_solve_cuda<<<dim3(n_groups, 2), 128, 0, stream>>>(
            (const float *) src_k->data, (const float *) src_v->data,
            (const float *) src_g->data, (const float *) src_b->data, A, W, U);
        return;
    }

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
