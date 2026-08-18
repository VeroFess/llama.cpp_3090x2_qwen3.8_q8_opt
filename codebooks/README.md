# TCQ Codebook Binaries

Raw float32 arrays. 3-bit = 512 floats (2048 bytes), 2-bit = 256 floats (1024 bytes).
Load at runtime via `TURBO_TCQ_CB=<path>` (3-bit) or `TURBO_TCQ_CB2=<path>` (2-bit).

## Best codebooks

- **3-bit best (long ctx)**: `3bit/cb_50iter_finetuned.bin` — CUDA GLA fine-tuned, 50 iters from coset init. 52.8% MSE reduction. Best at 32K+, slight crossover at ~8K.
- **2-bit best (long ctx)**: `2bit/tcq_2bit_100iter_s99.bin` — numpy GLA, 100 iters, seed 99. 32.1% MSE reduction. Best at 64K.
- **2-bit best (short ctx)**: `2bit/tcq_2bit_cuda_200iter.bin` — CUDA GLA, 200 iters. 34.9% MSE reduction. Best at 2K-8K.

## 2-bit codebooks

| File | Method | MSE Red. | PPL @2K | PPL @64K |
|------|--------|----------|---------|----------|
| tcq_2bit_3iter_s99.bin | numpy 3-iter | 0.7% | 6.843 | 7.667 |
| tcq_2bit_10iter_s99.bin | numpy 10-iter | 13.0% | 6.842 | 7.749 |
| tcq_2bit_30iter_s99.bin | numpy 25.5% | 25.5% | 6.804 | 7.287 |
| tcq_2bit_50iter_s99.bin | numpy 50-iter | 28.5% | 6.781 | 7.301 |
| tcq_2bit_100iter_s99.bin | numpy 100-iter | 32.1% | 6.708 | **7.222** |
| tcq_2bit_cuda_200iter.bin | CUDA 200-iter | 34.9% | **6.658** | 7.305 |

## 3-bit codebooks

- `cb_50iter_finetuned.bin` — best, CUDA fine-tuned from coset init
- `tcq_3bit_numpy_s{7,42,99,123,999}.bin` — numpy GLA, 100 iters, various seeds
- `tcq_3bit_3iter_s{7,42,123,999}.bin` — numpy GLA, 3 iters (undertrained)
- `tcq_3bit_10iter_s42.bin` — numpy GLA, 10 iters
- `tcq_3bit_50iter_s42.bin` — numpy GLA, 50 iters
- `tcq_3bit_s99.bin`, `tcq_3bit_tb_s99.bin` — CUDA trained variants
- `tcq_3bit_left_s99.bin`, `tcq_3bit_left_tb_s99.bin` — left-shift trellis variants
