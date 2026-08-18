#!/usr/bin/env python3
"""
TurboQuant CUDA patch v2 for TheTom/llama-cpp-turboquant
Adds CUDA support for turbo3 + turbo4 with:
  - Norm caching in FA vec_dot (avoids redundant fp16→float conversions)
  - Mixed K/V type support (turbo3 K + q8_0 V, etc.)
Run from repo root: python3 apply_turbo_cuda_v2.py
"""
import sys, os

CUDA = "ggml/src/ggml-cuda"

def read(p):
    with open(p) as f: return f.read()
def write(p, c):
    with open(p, 'w') as f: f.write(c)
def find_line(lines, needle):
    s = needle.strip()
    for i, l in enumerate(lines):
        if s in l.strip(): return i
    return -1
def find_line_after(lines, needle, start=0):
    s = needle.strip()
    for i in range(start, len(lines)):
        if s in lines[i].strip(): return i
    return -1
def insert_after(path, needle, text, desc, occ=1):
    c = read(path); lines = c.split('\n'); count = 0
    for i, l in enumerate(lines):
        if needle.strip() in l.strip():
            count += 1
            if count == occ:
                for j, nl in enumerate(text.split('\n')): lines.insert(i+1+j, nl)
                write(path, '\n'.join(lines)); print(f"  OK: {desc}"); return
    print(f"  FAIL: {desc}"); sys.exit(1)
def insert_before(path, needle, text, desc, occ=1):
    c = read(path); lines = c.split('\n'); count = 0
    for i, l in enumerate(lines):
        if needle.strip() in l.strip():
            count += 1
            if count == occ:
                for j, nl in enumerate(text.split('\n')): lines.insert(i+j, nl)
                write(path, '\n'.join(lines)); print(f"  OK: {desc}"); return
    print(f"  FAIL: {desc}"); sys.exit(1)

if not os.path.isfile(f"{CUDA}/set-rows.cu"):
    print("ERROR: Run from the llama-cpp-turboquant repo root."); sys.exit(1)
if 'turbo-quant-cuda.cuh' in read(f"{CUDA}/set-rows.cu"):
    print("Patch already applied. Aborting."); sys.exit(1)

print("=== TurboQuant CUDA Patch v2 (turbo3 + turbo4 + mixed K/V) ===\n")

# =====================================================================
# STEP 0: Create new files
# =====================================================================
print("[0/7] Creating new files...")

write(f"{CUDA}/turbo-quant-cuda.cuh", r'''#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "ggml-common.h"

// === Shared constants ===
static __constant__ float d_turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

static __device__ __forceinline__
uint8_t turbo_find_nearest_3bit(float val) {
    if      (val < d_turbo_mid_3bit[0]) return 0;
    else if (val < d_turbo_mid_3bit[1]) return 1;
    else if (val < d_turbo_mid_3bit[2]) return 2;
    else if (val < d_turbo_mid_3bit[3]) return 3;
    else if (val < d_turbo_mid_3bit[4]) return 4;
    else if (val < d_turbo_mid_3bit[5]) return 5;
    else if (val < d_turbo_mid_3bit[6]) return 6;
    else                                return 7;
}

// === TURBO3: SET_ROWS kernel ===
template<typename idx_t>
static __global__ void k_set_rows_turbo3(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO3_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO3_GROUP;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    // NOTE: FWHT rotation omitted — no graph-level inverse on CUDA yet.
    // TODO: Add pre-rotate-Q in FA kernel + inverse-rotate output for optimal quality.
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO3;
        blk.norm = __float2half(grp_norm);
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk.signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t idx = turbo_find_nearest_3bit(x[off + j]);
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));
        }
    }
}

// === TURBO3: GET_ROWS dequantize ===
#define QR_TURBO3_0 2
static __device__ __forceinline__
void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.x = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
    { const int j = iqs + 16;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.y = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
}

// === TURBO4: 3-bit unpack helper ===
static __device__ __forceinline__
uint8_t turbo4_unpack_3bit(const uint8_t * qs, int j) {
    int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 48) raw |= (uint16_t)qs[byte_idx + 1] << 8;
    return (uint8_t)((raw >> bit_pos) & 0x7);
}

// === TURBO4: SET_ROWS quantize ===
static __device__ __forceinline__
void quantize_f32_turbo4_0_block(const float * src, block_turbo4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst->norm = __float2half(norm);
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    for (int j = 0; j < 48; j++) dst->qs[j] = 0;
    for (int j = 0; j < 16; j++) dst->signs[j] = 0;
    float recon[128];
    for (int j = 0; j < 128; j++) {
        uint8_t idx = turbo_find_nearest_3bit(x[j]);
        recon[j] = d_turbo_centroids_3bit[idx];
        int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
        dst->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < 48)
            dst->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
    }
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        float r = x[j] - recon[j]; rnorm_sq += r * r;
        if (r >= 0.0f) dst->signs[j / 8] |= (1 << (j % 8));
    }
    dst->rnorm = __float2half(sqrtf(rnorm_sq));
}

// === TURBO4: GET_ROWS dequantize ===
#define QR_TURBO4_0 2
static __device__ __forceinline__
void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const float qjl_scale = 1.2533141f / 128.0f * rnorm;
    { const int j = iqs;
      uint8_t idx = turbo4_unpack_3bit(x[ib].qs, j);
      float s = (x[ib].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
      v.x = (d_turbo_centroids_3bit[idx] + s * qjl_scale) * norm; }
    { const int j = iqs + 64;
      uint8_t idx = turbo4_unpack_3bit(x[ib].qs, j);
      float s = (x[ib].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
      v.y = (d_turbo_centroids_3bit[idx] + s * qjl_scale) * norm; }
}
''')
print("  Created: turbo-quant-cuda.cuh")

# Template instances: turbo3×turbo3, turbo4×turbo4, plus mixed types
for kt, vt in [("turbo3_0","turbo3_0"), ("turbo4_0","turbo4_0"),
               ("turbo3_0","q8_0"), ("turbo4_0","q8_0"),
               ("q8_0","turbo3_0"), ("q8_0","turbo4_0")]:
    KT = f"GGML_TYPE_{kt.upper()}"
    VT = f"GGML_TYPE_{vt.upper()}"
    fname = f"fattn-vec-instance-{kt}-{vt}.cu"
    write(f"{CUDA}/template-instances/{fname}",
        f'#include "../fattn-vec.cuh"\n'
        f'DECL_FATTN_VEC_CASE( 64, {KT}, {VT});\n'
        f'DECL_FATTN_VEC_CASE(128, {KT}, {VT});\n'
        f'DECL_FATTN_VEC_CASE(256, {KT}, {VT});\n')
    print(f"  Created: {fname}")

# =====================================================================
# STEP 1: set-rows.cu
# =====================================================================
print("\n[1/7] set-rows.cu")
f = f"{CUDA}/set-rows.cu"
insert_after(f, '#include "cpy-utils.cuh"', '#include "turbo-quant-cuda.cuh"', "add include")

TURBO_SET = """\
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        GGML_ASSERT(ne00 % QK_TURBO3_GROUP == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO3_GROUP;
        const int num_blocks_grid = (ne_total_groups + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            k_set_rows_turbo3<idx_t><<<num_blocks_grid, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                src0_d, src1_d, (block_turbo3_0 *)dst->data,
                ne_total_groups, ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_quant<idx_t, block_turbo4_0, QK_TURBO4, quantize_f32_turbo4_0_block>(
            src0_d, src1_d, (block_turbo4_0*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,
            nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);"""

content = read(f); lines = content.split('\n')
for i, l in enumerate(lines):
    if 'GGML_ABORT' in l and 'unsupported type' in l:
        insert_at = i - 1
        while insert_at >= 0 and '} else {' not in lines[insert_at]: insert_at -= 1
        if insert_at >= 0:
            for j, nl in enumerate(TURBO_SET.split('\n')): lines.insert(insert_at + j, nl)
            write(f, '\n'.join(lines)); print("  OK: add turbo3+turbo4 dispatch"); break
else:
    print("  FAIL: set-rows dispatch"); sys.exit(1)

# =====================================================================
# STEP 2: getrows.cu
# =====================================================================
print("\n[2/7] getrows.cu")
f = f"{CUDA}/getrows.cu"
insert_after(f, '#include "convert.cuh"', '#include "turbo-quant-cuda.cuh"', "add include")

TURBO_GET = """\
        case GGML_TYPE_TURBO3_0:
            get_rows_cuda_q<QK_TURBO3, QR_TURBO3_0, dequantize_turbo3_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TURBO4_0:
            get_rows_cuda_q<QK_TURBO4, QR_TURBO4_0, dequantize_turbo4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;"""
content = read(f); lines = content.split('\n')
for i, l in enumerate(lines):
    if 'default:' in l:
        ctx = '\n'.join(lines[max(0,i-2):i+3])
        if 'src0' in ctx or 'k-quants' in ctx:
            for j, nl in enumerate(TURBO_GET.split('\n')): lines.insert(i+j, nl)
            write(f, '\n'.join(lines)); print("  OK: add turbo3+turbo4 case"); break
else:
    print("  FAIL: getrows"); sys.exit(1)

# =====================================================================
# STEP 3: ggml-cuda.cu
# =====================================================================
print("\n[3/7] ggml-cuda.cu")
cuda_main = None
for name in os.listdir(CUDA):
    if name.endswith('.cu'):
        c = read(f"{CUDA}/{name}")
        if 'GGML_OP_SET_ROWS' in c and 'GGML_TYPE_IQ4_NL' in c:
            cuda_main = f"{CUDA}/{name}"; break
if not cuda_main: print("  FAIL: can't find main CUDA file"); sys.exit(1)
print(f"  Using: {cuda_main}")

content = read(cuda_main)
content = content.replace(
    'GGML_TYPE_IQ4_NL) &&',
    'GGML_TYPE_IQ4_NL ||\n                       op->type == GGML_TYPE_TURBO3_0 || op->type == GGML_TYPE_TURBO4_0) &&', 1)
write(cuda_main, content); print("  OK: SET_ROWS supports_op")

content = read(cuda_main); lines = content.split('\n')
for loc in [i for i, l in enumerate(lines) if 'case GGML_OP_GET_ROWS:' in l]:
    ctx = '\n'.join(lines[loc:loc+15])
    if 'src[0]->type' in ctx or 'src0->type' in ctx:
        for i in range(loc, min(loc+20, len(lines))):
            if 'GGML_TYPE_Q8_0' in lines[i] and 'case' in lines[i]:
                lines.insert(i+1, '                    case GGML_TYPE_TURBO3_0:')
                lines.insert(i+2, '                    case GGML_TYPE_TURBO4_0:')
                write(cuda_main, '\n'.join(lines)); print("  OK: GET_ROWS supports_op"); break
        break
else:
    print("  FAIL: GET_ROWS"); sys.exit(1)

# =====================================================================
# STEP 4: fattn-common.cuh — centroids, vec_dot with norm caching, dequantize_V
# =====================================================================
print("\n[4/7] fattn-common.cuh")
f = f"{CUDA}/fattn-common.cuh"

insert_after(f, '#include <cstdint>',
    '\nstatic __constant__ float d_turbo_centroids_3bit_fattn[8] = {\n'
    '    -0.190685f, -0.117832f, -0.065717f, -0.021460f,\n'
    '     0.021460f,  0.065717f,  0.117832f,  0.190685f\n'
    '};', "add centroids")

# turbo4 unpack helper + vec_dot for turbo3 and turbo4 (with norm caching)
VEC_DOT_ALL = '''
static __device__ __forceinline__
uint8_t turbo4_unpack_3bit_fattn(const uint8_t * qs, int j) {
    int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 48) raw |= (uint16_t)qs[byte_idx + 1] << 8;
    return (uint8_t)((raw >> bit_pos) & 0x7);
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo3_0 * K_t3 = (const block_turbo3_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int elem = (base_f2 + k_KQ_1) * 2;
            const int ib = elem / QK_TURBO3, j0 = elem % QK_TURBO3;
            // Norm caching: only reload when block changes
            const float norm = __half2float(K_t3[ib].norm);
            float k0, k1;
            { const int j = j0;
              const uint8_t low2 = (K_t3[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
              const uint8_t hi1  = (K_t3[ib].signs[j/8] >> (j%8)) & 0x1;
              k0 = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm; }
            { const int j = j0 + 1;
              const uint8_t low2 = (K_t3[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
              const uint8_t hi1  = (K_t3[ib].signs[j/8] >> (j%8)) & 0x1;
              k1 = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm; }
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += k0 * qf.x + k1 * qf.y;
        }
    }
    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo4_0 * K_t4 = (const block_turbo4_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    // Norm caching: turbo4 block = 128 elements = D, so one block per head.
    // Load norm/rnorm/qjl_scale once per block instead of per element pair.
    int prev_ib = -1;
    float norm = 0.0f, qjl_scale = 0.0f;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int elem = (base_f2 + k_KQ_1) * 2;
            const int ib = elem / QK_TURBO4, j0 = elem % QK_TURBO4;
            if (ib != prev_ib) {
                norm = __half2float(K_t4[ib].norm);
                const float rnorm = __half2float(K_t4[ib].rnorm);
                qjl_scale = 1.2533141f / 128.0f * rnorm;
                prev_ib = ib;
            }
            float k0, k1;
            { const int j = j0;
              float c = d_turbo_centroids_3bit_fattn[turbo4_unpack_3bit_fattn(K_t4[ib].qs, j)];
              float s = (K_t4[ib].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
              k0 = (c + s * qjl_scale) * norm; }
            { const int j = j0 + 1;
              float c = d_turbo_centroids_3bit_fattn[turbo4_unpack_3bit_fattn(K_t4[ib].qs, j)];
              float s = (K_t4[ib].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
              k1 = (c + s * qjl_scale) * norm; }
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += k0 * qf.x + k1 * qf.y;
        }
    }
    return sum;
}
'''
insert_before(f, 'template <typename Tds, int ni>', VEC_DOT_ALL, "add vec_dot turbo3+turbo4")

# dequantize_V for turbo3 + turbo4
DEQUANT_V = '''
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const int64_t ib = i0 / QK_TURBO3;
    const int     j0 = (int)(i0 % QK_TURBO3);
    const float norm = __half2float(x[ib].norm);
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int j = j0 + l;
        const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
        const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
        vals[l] = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm;
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const int64_t ib = i0 / QK_TURBO4;
    const int     j0 = (int)(i0 % QK_TURBO4);
    // Norm caching: load once per block
    const float norm = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const float qjl_scale = 1.2533141f / 128.0f * rnorm;
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int j = j0 + l;
        float c = d_turbo_centroids_3bit_fattn[turbo4_unpack_3bit_fattn(x[ib].qs, j)];
        float s = (x[ib].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
        vals[l] = (c + s * qjl_scale) * norm;
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}
'''
insert_before(f, 'template <ggml_type type_K, int D, int nthreads>', DEQUANT_V, "add dequantize_V turbo3+turbo4")

# Dispatch
insert_after(f, 'return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;',
    '    } else if constexpr (type_K == GGML_TYPE_TURBO3_0) {\n        return vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads>;\n'
    '    } else if constexpr (type_K == GGML_TYPE_TURBO4_0) {\n        return vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads>;',
    "add turbo to get_vec_dot_KQ")
insert_after(f, 'return dequantize_V_q8_0<T, ne>;',
    '    } else if constexpr (type_V == GGML_TYPE_TURBO3_0) {\n        return dequantize_V_turbo3_0<T, ne>;\n'
    '    } else if constexpr (type_V == GGML_TYPE_TURBO4_0) {\n        return dequantize_V_turbo4_0<T, ne>;',
    "add turbo to get_dequantize_V")

# =====================================================================
# STEP 5: fattn-vec.cuh
# =====================================================================
print("\n[5/7] fattn-vec.cuh")
f = f"{CUDA}/fattn-vec.cuh"
content = read(f)
content = content.replace(
    'type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q',
    '(type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO4_0) ? 128 / cpy_nb : nthreads_KQ_q')
content = content.replace(
    'type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q',
    '(type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO4_0) ? 128 / cpy_nb : nthreads_V_q')
content = content.replace(
    'type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4',
    '(type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO4_0) ? 2*cpy_ne : 4')
content = content.replace(
    'type_K != GGML_TYPE_F16;',
    'type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_TURBO3_0 && type_K != GGML_TYPE_TURBO4_0;')
write(f, content); print("  OK: nthreads/Q_q8_1/V_rows fixes")

# EXTERN_DECL macro: add turbo3 and turbo4
insert_after(f, 'extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0);',
    '    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_TURBO3_0); \\\n'
    '    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_TURBO4_0); \\',
    "add turbo to EXTERN_DECL macro")

# Add extern decls for turbo as K type + q8_0 as K type (for mixed)
content = read(f); lines = content.split('\n')
last_extern = -1
for i, l in enumerate(lines):
    if l.strip().startswith('EXTERN_DECL_FATTN_VEC_CASES('): last_extern = i
if last_extern >= 0:
    ins = ['',
        'EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_TURBO3_0)',
        'EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_TURBO3_0)',
        'EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_TURBO3_0)',
        '',
        'EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_TURBO4_0)',
        'EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_TURBO4_0)',
        'EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_TURBO4_0)',
    ]
    for j, nl in enumerate(ins): lines.insert(last_extern + 1 + j, nl)
    write(f, '\n'.join(lines)); print("  OK: add turbo extern decls")

# =====================================================================
# STEP 6: fattn.cu
# =====================================================================
print("\n[6/7] fattn.cu")
f = f"{CUDA}/fattn.cu"

# Type gate
content = read(f); lines = content.split('\n')
for i, l in enumerate(lines):
    if 'case GGML_TYPE_Q8_0:' in l:
        for j in range(i+1, min(i+3, len(lines))):
            if 'break;' in lines[j]:
                lines.insert(i+1, '        case GGML_TYPE_TURBO3_0:')
                lines.insert(i+2, '        case GGML_TYPE_TURBO4_0:')
                write(f, '\n'.join(lines)); print("  OK: type gate"); break
        break

# Force VEC for turbo types
content = read(f); lines = content.split('\n')
for i, l in enumerate(lines):
    if 'can_use_vector_kernel' in l and 'const bool' in l:
        ins = ['',
            '    // TurboQuant: only the vec kernel has turbo dequant support.',
            '    if (K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0 ||',
            '        K->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO4_0) {',
            '        if (Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0)',
            '            return BEST_FATTN_KERNEL_VEC;',
            '        return BEST_FATTN_KERNEL_NONE;',
            '    }', '']
        for j, nl in enumerate(ins): lines.insert(i + j, nl)
        write(f, '\n'.join(lines)); print("  OK: force VEC for turbo"); break

# Vec dispatch: add all turbo combinations
content = read(f)
q8_pat = 'FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)'
turbo_lines = (
    '\n    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)'
    '\n    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)'
    '\n    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)'
    '\n    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)'
    '\n    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)'
    '\n    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)')
content = content.replace(q8_pat, q8_pat + turbo_lines)
write(f, content); print("  OK: vec dispatch (all combos)")

# =====================================================================
# STEP 7: CMakeLists.txt
# =====================================================================
print("\n[7/7] CMakeLists.txt")
f = f"{CUDA}/CMakeLists.txt"
content = read(f); lines = content.split('\n')
f16_glob = find_line(lines, 'fattn-vec*f16-f16.cu')
append_line = find_line_after(lines, 'list(APPEND GGML_SOURCES_CUDA', f16_glob + 1)
new_lines = [
    '        file(GLOB   SRCS "template-instances/fattn-vec*turbo3_0*.cu")',
    '        list(APPEND GGML_SOURCES_CUDA ${SRCS})',
    '        file(GLOB   SRCS "template-instances/fattn-vec*turbo4_0*.cu")',
    '        list(APPEND GGML_SOURCES_CUDA ${SRCS})',
]
for j, nl in enumerate(new_lines): lines.insert(append_line + 1 + j, nl)
write(f, '\n'.join(lines)); print("  OK: add turbo globs")

# =====================================================================
print("\n=== Patch applied successfully ===")
print("\nBuild:")
print("  rm -rf build")
print("  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release")
print("  cmake --build build -j$(nproc)")
print("\nTest:")
print("  # turbo3 K+V (3.25 bits, 4.9x compression)")
print("  ./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo3 --cache-type-v turbo3 -fa")
print("  # turbo4 K+V (4.25 bits, 3.8x compression, QJL correction)")
print("  ./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo4 --cache-type-v turbo4 -fa")
print("  # mixed: turbo3 keys + q8_0 values (aggressive K, safe V)")
print("  ./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo3 --cache-type-v q8_0 -fa")
print("  # mixed: turbo4 keys + q8_0 values")
print("  ./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo4 --cache-type-v q8_0 -fa")
