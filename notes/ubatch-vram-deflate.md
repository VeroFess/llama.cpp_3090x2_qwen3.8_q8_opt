# Ubatch VRAM Deflate — Investigation Notes

## Problem

`-ub` (ubatch) size determines permanent VRAM usage because llama.cpp's
`sched_reserve()` allocates compute buffers at init sized for the worst-case
graph (n_ubatch tokens) and never frees them.

- `-ub 512`: fast prefill, but ~2.5 GB more permanent VRAM
- `-ub 64` (DFlash auto-cap): slow prefill, low VRAM
- Users have to guess what fits — too high OOMs, too low wastes throughput

## Root Cause

`src/llama-context.cpp:428`:
```cpp
const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
```

`sched_reserve()` (line 414) builds a PP graph with n_ubatch tokens (line 585),
allocates VRAM for all intermediate tensors, and keeps it forever. The
attention intermediates scale O(n_tokens²) — 512 vs 64 is 64x in the
quadratic part.

`ggml_gallocr_reserve_n_impl()` in `ggml/src/ggml-alloc.c:913` only grows
buffers, never shrinks. Buffers persist in `galloc->buffers[i]`.

## Proposed Solution: Deflate After Prefill

`sched_reserve()` destroys the old scheduler via `sched.reset()` (unique_ptr)
and builds a new one — old compute buffers ARE freed. If we call it again
after prefill with a generation-sized graph, the large buffers get replaced
with tiny ones.

### Flow

1. New request → `sched_reserve(n_ubatch)` → process prompt → fast prefill
2. Prefill done → `sched_reserve(n_seqs)` → VRAM drops back down
3. Generation runs with minimal compute buffer
4. Next prompt → `sched_reserve(n_ubatch)` again

Cost of `sched_reserve()` is ~ms, negligible vs multi-second generation.

### Auto-probe (complementary)

Instead of users guessing `-ub`:
- Try user's requested ubatch, catch OOM, halve, retry
- Or use `cudaMemGetInfo` to estimate largest ubatch that fits pre-allocation

Combined: auto-probe finds max ubatch for prefill speed, deflate shrinks
it back for generation. Users set nothing, get optimal behavior.

## Open Questions / Potential Complications

- Server with multiple slots: one slot generating while another starts
  prefill — re-reserve would disrupt the generating slot's compute state
- `sched_reserve` rebuilds the entire scheduler (backend enumeration, graph
  result objects) — is there a lighter-weight path?
- Speculative decoding verify batch is bigger than pure TG (draft_max tokens)
  — deflated size needs to account for this
- Does re-reserving invalidate any cached graph metadata that affects
  correctness?
- Thread safety: server's slot processing is single-threaded per-batch but
  `sched_reserve` touches shared state
- DFlash-specific: draft model has its own context + scheduler — would need
  independent deflate
- KV cache state during re-reserve — does destroying the scheduler affect
  KV cache buffer ownership?

## Files

| File | Relevance |
|------|-----------|
| `src/llama-context.cpp:414-624` | `sched_reserve()`, graph reservation |
| `ggml/src/ggml-alloc.c:913-942` | Buffer allocation (grow-only) |
| `ggml/src/ggml-backend.cpp:1759` | `ggml_backend_sched_reserve` |
