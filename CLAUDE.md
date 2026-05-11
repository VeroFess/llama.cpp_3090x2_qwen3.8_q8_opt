# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Fork of llama.cpp (`spiritbuun/buun-llama-cpp`) adding speculative decoding features:
- **DFlash**: Cross-attention drafter (DeltaNet-based, Qwen3.5/3.6 family). Captures target hidden states via eval callback, feeds them to a small drafter model through cross-attention layers.
- **TurboQuant**: Custom quantization types (turbo2/turbo3/turbo4) for KV cache compression. turbo4 = 4-bit KV with fused MMA kernels on Turing+.
- **DDTree**: Tree-based speculative decoding with SSM tree kernels (parent_ids on GPU).
- **CopySpec**: Suffix/copy-based speculation (model-free).

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)
```

Key binaries: `build/bin/llama-server`, `build/bin/llama-cli`, `build/bin/llama-bench`, `build/bin/llama-perplexity`

## Architecture

### Layers
- `ggml/` — tensor library, CUDA/Metal/CPU backends, quantization
- `src/` — llama.cpp core: model loading, context, graph building, sampling
- `src/models/` — per-architecture graph builders (one .cpp per model family)
- `common/` — shared utilities, speculative decoding orchestration
- `tools/server/` — HTTP server (slots, chat completions API)
- `include/llama.h` — public C API (~276 functions)

### DFlash-specific files
- `src/models/dflash_draft.cpp` — drafter graph builder (cross-attention + DeltaNet recurrent)
- `src/models/qwen35.cpp` — target model (hybrid attention + recurrent layers)
- `common/speculative.cpp` — `common_speculative_state_dflash` class: ring buffer, draft/verify loop, GPU ring interleave
- `ggml/src/ggml-cuda/cross-ring-interleave.cu` — GPU-side ring buffer management + interleave kernel
- `ggml/src/ggml-cuda/gated-delta-net.cu` — DeltaNet forward (flat + tree variants)
- `ggml/src/ggml-cuda/ssm-conv.cu` — SSM conv1d (flat + tree variants)

### Key patterns
- **proc_address**: Custom CUDA functions (GPU ring, tree kernels) are registered via `ggml_backend_cuda_reg_get_proc_address` and resolved at runtime. Not linked directly.
- **Tree verify**: `tree_bufs.parent_ids_gpu` enables batched tree kernels. Disabled automatically on multi-GPU (recurrent layers can't read cross-device).
- **Eval callback**: Target hidden states captured during forward pass via `llama_set_eval_callback`. Stored in CPU ring buffer, mirrored to GPU ring.
- **Ring buffer**: CPU `ring_buf[layer][slot*n_embd]` + GPU mirror. Cross-attention input built by interleaving layers for the context window.

## Test / Benchmark

```bash
# Perplexity (PPL)
build/bin/llama-perplexity -m model.gguf -f test.txt -c 4096

# Decode speed
build/bin/llama-bench -m model.gguf -p 0 -n 64 -t 1  # tg64

# Server with DFlash
build/bin/llama-server -m target.gguf --draft-model drafter.gguf \
  --draft-max 16 -ngl 99 -c 4096 --port 8080
```

Grep for `"Final estimate"` (PPL) and `"tg64"` (decode speed) in output.

## Multi-GPU Notes

- Tree verify (`parent_ids_gpu`) lives on GPU 0 only. Auto-disabled when `model.n_devices() > 1`.
- GPU ring buffer (`cross-ring-interleave.cu`) allocates without explicit `cudaSetDevice` — potential cross-device issue on multi-GPU setups.
- Uses `cudaStreamPerThread` throughout — each host thread gets its own stream per device.

## Git Conventions

- Branch per experiment: `experiment/SD-XXX-short-name`
- Never delete experiment branches
- Only merged, verified, clean code on `main`
