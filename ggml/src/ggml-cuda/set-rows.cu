#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant-cuda.cuh"
#include <cstring>
#include <cerrno>

static void load_turbo4_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    const char *s = getenv("TURBO4_ALPHA");
    if (!s) return;
    char *end;
    errno = 0;
    float a = strtof(s, &end);
    if (end == s || errno != 0 || a <= 0.0f || a >= 10.0f) {
        fprintf(stderr, "TURBO4: invalid TURBO4_ALPHA='%s'\n", s);
    } else {
        cudaMemcpyToSymbol(d_turbo4_alpha, &a, sizeof(float));
        fprintf(stderr, "TURBO4: alpha=%.3f (device %d)\n", a, device);
    }
}

static void load_tcq_norm_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;

    // Context-adaptive decode-time alpha is the default. Force encode-time V alpha to 1.0
    // unless TURBO_TCQ_ENCODE_ALPHA=1 is explicitly set to use encode-time alpha instead.
    const char *encode_mode = getenv("TURBO_TCQ_ENCODE_ALPHA");
    if (!encode_mode) {
        float one = 1.0f;
        cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &one, sizeof(float));
        if (device == 0) fprintf(stderr, "TCQ: encode V alpha=1.0 (context-adaptive decode-time alpha active)\n");
        // Still allow K alpha override
        const char *s = getenv("TURBO_TCQ_ALPHA");
        if (s) {
            char *end;
            errno = 0;
            float a = strtof(s, &end);
            if (end != s && errno == 0 && a > 0.0f && a < 10.0f) {
                cudaMemcpyToSymbol(d_tcq_norm_alpha, &a, sizeof(float));
            }
        }
        return;
    }

    const char *s = getenv("TURBO_TCQ_ALPHA");
    const char *sv = getenv("TURBO_TCQ_ALPHA_V");
    if (!s && !sv) return;
    float alpha_k = 1.0f;
    bool k_set = false;
    if (s) {
        char *end;
        errno = 0;
        float a = strtof(s, &end);
        if (end == s || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid TURBO_TCQ_ALPHA='%s'\n", s);
        } else {
            alpha_k = a;
            k_set = true;
            cudaMemcpyToSymbol(d_tcq_norm_alpha, &alpha_k, sizeof(float));
        }
    }
    if (sv) {
        char *end;
        errno = 0;
        float a = strtof(sv, &end);
        if (end == sv || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid TURBO_TCQ_ALPHA_V='%s'\n", sv);
        } else {
            cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &a, sizeof(float));
            fprintf(stderr, "TCQ: norm alpha K=%.3f V=%.3f\n", alpha_k, a);
            return;
        }
    }
    // TURBO_TCQ_ALPHA set but not TURBO_TCQ_ALPHA_V: V matches K for backwards compat
    if (k_set) {
        cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &alpha_k, sizeof(float));
        fprintf(stderr, "TCQ: norm alpha K=V=%.3f\n", alpha_k);
    }
}

// TCQ error dump for autocorrelation analysis (TURBO_TCQ_DUMP_ERRORS=N)
// Only active on the first device that triggers it (diagnostic tool, not perf-critical)
static int    tcq_dump_n = 0;
static int    tcq_dump_device = -1;
static float * tcq_dump_x_host = nullptr;
static uint8_t * tcq_dump_out_host = nullptr;
static float * tcq_dump_x_dev = nullptr;
static uint8_t * tcq_dump_out_dev = nullptr;

static void tcq_error_dump_flush() {
    if (tcq_dump_n == 0 || tcq_dump_device < 0) return;
    ggml_cuda_set_device(tcq_dump_device);
    cudaMemcpy(tcq_dump_x_host, tcq_dump_x_dev, tcq_dump_n * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tcq_dump_out_host, tcq_dump_out_dev, tcq_dump_n * 128 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    FILE * f = fopen("/tmp/tcq_errors.bin", "wb");
    if (f) {
        int32_t header[1] = { tcq_dump_n };
        fwrite(header, sizeof(int32_t), 1, f);
        fwrite(tcq_dump_x_host, sizeof(float), tcq_dump_n * 128, f);
        fwrite(tcq_dump_out_host, sizeof(uint8_t), tcq_dump_n * 128, f);
        fclose(f);
        fprintf(stderr, "TCQ: dumped %d groups to /tmp/tcq_errors.bin\n", tcq_dump_n);
    }
    cudaFree(tcq_dump_x_dev);
    cudaFree(tcq_dump_out_dev);
    free(tcq_dump_x_host);
    free(tcq_dump_out_host);
}

static void init_tcq_error_dump(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    const char *s = getenv("TURBO_TCQ_DUMP_ERRORS");
    if (!s) return;
    int n = atoi(s);
    if (n <= 0 || n > 500000) return;
    // Only allocate dump buffers on the first device that requests them
    if (tcq_dump_device >= 0) {
        // Already allocated on another device — just set the symbol pointers for this device
        cudaMemcpyToSymbol(d_tcq_dump_max, &n, sizeof(int));
        return;
    }
    tcq_dump_n = n;
    tcq_dump_device = device;
    tcq_dump_x_host = (float *)malloc(n * 128 * sizeof(float));
    tcq_dump_out_host = (uint8_t *)malloc(n * 128 * sizeof(uint8_t));
    cudaMalloc(&tcq_dump_x_dev, n * 128 * sizeof(float));
    cudaMalloc(&tcq_dump_out_dev, n * 128 * sizeof(uint8_t));
    cudaMemcpyToSymbol(d_tcq_dump_x_buf, &tcq_dump_x_dev, sizeof(float*));
    cudaMemcpyToSymbol(d_tcq_dump_out_buf, &tcq_dump_out_dev, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_tcq_dump_max, &n, sizeof(int));
    atexit(tcq_error_dump_flush);
    fprintf(stderr, "TCQ: will dump errors for first %d groups to /tmp/tcq_errors.bin\n", n);
}

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

// Per-device backtrace buffer for Viterbi (replaces 32KB shared memory per block)
static uint8_t * tcq_bt_buf[GGML_CUDA_MAX_DEVICES] = {};
static int64_t   tcq_bt_buf_bytes[GGML_CUDA_MAX_DEVICES] = {};

static void ensure_tcq_bt_buf(int device, int64_t bytes_needed) {
    if (bytes_needed <= tcq_bt_buf_bytes[device]) return;
    if (tcq_bt_buf[device]) cudaFree(tcq_bt_buf[device]);
    cudaMalloc(&tcq_bt_buf[device], bytes_needed);
    tcq_bt_buf_bytes[device] = bytes_needed;
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO2_0) {
        GGML_ASSERT(ne00 % QK_TURBO2_GROUP == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO2_GROUP;
        const int num_blocks_grid = (ne_total_groups + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            k_set_rows_turbo2<idx_t><<<num_blocks_grid, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                src0_d, src1_d, (block_turbo2_0 *)dst->data,
                ne_total_groups, ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        GGML_ASSERT(ne00 % QK_TURBO3_GROUP == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO3_GROUP;
        const int num_blocks_grid = (ne_total_groups + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            k_set_rows_turbo3<idx_t><<<num_blocks_grid, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                src0_d, src1_d, (block_turbo3_0 *)dst->data,
                ne_total_groups, ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, iq_is_k, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        load_turbo4_alpha(ctx.device);
        set_rows_cuda_quant<idx_t, block_turbo4_0, QK_TURBO4, quantize_f32_turbo4_0_block>(
            src0_d, src1_d, (block_turbo4_0*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,
            nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TURBO3_TCQ) {
        GGML_ASSERT(ne00 % QK_TURBO3_TCQ == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO3_TCQ;
        // Runtime codebook loading: TURBO_TCQ_CB overrides compiled-in codebook (per-device)
        {
            static bool tcq_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            if (!tcq_cb_loaded[ctx.device]) {
                tcq_cb_loaded[ctx.device] = true;
                const char *cb_path = getenv("TURBO_TCQ_CB");
                if (cb_path) {
                    float cb[512];
                    FILE *f = fopen(cb_path, "rb");
                    if (f && fread(cb, sizeof(float), 512, f) == 512) {
                        fclose(f);
                        cudaMemcpyToSymbol(d_turbo3_tcq_codebook, cb, 512*sizeof(float));
                        fprintf(stderr, "TCQ encode: loaded codebook from %s (device %d)\n", cb_path, ctx.device);
                    } else {
                        if (f) fclose(f);
                        fprintf(stderr, "TCQ encode: FAILED to load codebook from %s\n", cb_path);
                    }
                }
                load_tcq_norm_alpha(ctx.device);
                init_tcq_error_dump(ctx.device);
            }
        }
        // TCQ Viterbi encode: 512 threads per block. The TCQ3 backtrace stores
        // one predecessor for each 64-state low-bit group per step.
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            static int tcq3_use_shared_bt[GGML_CUDA_MAX_DEVICES] = {};
            static bool tcq3_bt_checked[GGML_CUDA_MAX_DEVICES] = {};
            constexpr int tcq3_bt_shared_bytes = 128 * 64;
            if (!tcq3_bt_checked[ctx.device]) {
                tcq3_bt_checked[ctx.device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                const char * tcq_shared_bt_env = getenv("TURBO_TCQ_SHARED_BT");
                if (!tcq_shared_bt_env || atoi(tcq_shared_bt_env) != 0) {
                    int max_shared_optin = 0;
                    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx.device));
                    if (max_shared_optin >= tcq3_bt_shared_bytes) {
                        CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo3_tcq<idx_t>, tcq3_bt_shared_bytes);
                        tcq3_use_shared_bt[ctx.device] = 1;
                        fprintf(stderr, "TCQ encode: using shared-memory backtrace (%d bytes/block)\n", tcq3_bt_shared_bytes);
                    } else {
                        fprintf(stderr, "TCQ encode: shared-memory backtrace unavailable, only %d bytes/block are available\n", max_shared_optin);
                    }
                }
#endif
            }
            if (!tcq3_use_shared_bt[ctx.device]) {
                ensure_tcq_bt_buf(ctx.device, ne_total_groups * 128 * 64);
            }
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            const int shared_bytes = tcq3_use_shared_bt[ctx.device] ? tcq3_bt_shared_bytes : 0;
            k_set_rows_turbo3_tcq<idx_t><<<(int)ne_total_groups, 512, shared_bytes, stream>>>(
                src0_d, src1_d, (block_turbo3_tcq *)dst->data,
                ne_total_groups, tcq_bt_buf[ctx.device], tcq3_use_shared_bt[ctx.device], ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, iq_is_k, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO2_TCQ) {
        GGML_ASSERT(ne00 % QK_TURBO2_TCQ == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO2_TCQ;
        // Runtime codebook loading: TURBO_TCQ_CB2 overrides compiled-in 2-bit codebook (per-device)
        {
            static bool tcq2_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            if (!tcq2_cb_loaded[ctx.device]) {
                tcq2_cb_loaded[ctx.device] = true;
                const char *cb_path = getenv("TURBO_TCQ_CB2");
                if (cb_path) {
                    float cb[256];
                    FILE *f = fopen(cb_path, "rb");
                    if (f && fread(cb, sizeof(float), 256, f) == 256) {
                        fclose(f);
                        cudaMemcpyToSymbol(d_turbo2_tcq_codebook, cb, 256*sizeof(float));
                        fprintf(stderr, "TCQ2 encode: loaded 2-bit codebook from %s (device %d)\n", cb_path, ctx.device);
                    } else {
                        if (f) fclose(f);
                        fprintf(stderr, "TCQ2 encode: FAILED to load codebook from %s\n", cb_path);
                    }
                }
                load_tcq_norm_alpha(ctx.device);
                init_tcq_error_dump(ctx.device);
            }
        }
        // 2-bit TCQ Viterbi encode: 256 threads per block. Compressed backtrace
        // stores one predecessor per 64 low-state groups per step (same as turbo3_tcq).
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            static int tcq2_use_shared_bt[GGML_CUDA_MAX_DEVICES] = {};
            static bool tcq2_bt_checked[GGML_CUDA_MAX_DEVICES] = {};
            constexpr int tcq2_bt_shared_bytes = 128 * 64;
            if (!tcq2_bt_checked[ctx.device]) {
                tcq2_bt_checked[ctx.device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                const char * tcq_shared_bt_env = getenv("TURBO_TCQ_SHARED_BT");
                if (!tcq_shared_bt_env || atoi(tcq_shared_bt_env) != 0) {
                    int max_shared_optin = 0;
                    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx.device));
                    if (max_shared_optin >= tcq2_bt_shared_bytes) {
                        CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo2_tcq<idx_t>, tcq2_bt_shared_bytes);
                        tcq2_use_shared_bt[ctx.device] = 1;
                    }
                }
#endif
            }
            if (!tcq2_use_shared_bt[ctx.device]) {
                ensure_tcq_bt_buf(ctx.device, ne_total_groups * 128 * 64);
            }
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            const int shared_bytes = tcq2_use_shared_bt[ctx.device] ? tcq2_bt_shared_bytes : 0;
            k_set_rows_turbo2_tcq<idx_t><<<(int)ne_total_groups, 256, shared_bytes, stream>>>(
                src0_d, src1_d, (block_turbo2_tcq *)dst->data,
                ne_total_groups, tcq_bt_buf[ctx.device], tcq2_use_shared_bt[ctx.device], ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, iq_is_k, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


// InnerQ calibration state machine (driven by TURBO_INNERQ env var)
static int innerq_state = 0; // 0=uninit, 1=calibrating, 2=active, -1=disabled
static int innerq_tokens_seen = 0;
static constexpr int INNERQ_CALIBRATION_TOKENS = 100000; // count total set_rows tokens across all layers

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    // Post-rotation extraction: one-time init
    if (h_extract_state == 0 && (dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO3_TCQ || dst->type == GGML_TYPE_TURBO2_TCQ)) {
        static const char * env = getenv("TURBO_EXTRACT");
        if (env && atoi(env) > 0) {
            turbo_extract_init(atoi(env));
        } else {
            h_extract_state = -1;
        }
    }
    // Check if extraction buffer is full
    if (h_extract_state == 1) turbo_extract_check_done();

    // InnerQ: one-time init on first turbo SET_ROWS call
    if (innerq_state == 0 && (dst->type == GGML_TYPE_TURBO2_0 || dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO3_TCQ || dst->type == GGML_TYPE_TURBO2_TCQ)) {
        static const char * env = getenv("TURBO_INNERQ");
        if (env && atoi(env) > 0) {
            turbo_innerq_init();
            turbo_innerq_start_calibration();
            innerq_state = 1;
            fprintf(stderr, "InnerQ: calibration started (collecting %d tokens)\n", INNERQ_CALIBRATION_TOKENS);
        } else {
            turbo_innerq_init(); // identity scales
            innerq_state = -1;
        }
    }

    // Track calibration progress
    if (innerq_state == 1 && (dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO3_TCQ || dst->type == GGML_TYPE_TURBO2_TCQ)) {
        innerq_tokens_seen += dst->src[0]->ne[1];
        if (innerq_tokens_seen >= INNERQ_CALIBRATION_TOKENS) {
            turbo_innerq_finalize_calibration();
            innerq_state = 2;
            fprintf(stderr, "InnerQ: calibration complete, scales active\n");
        }
    }

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
