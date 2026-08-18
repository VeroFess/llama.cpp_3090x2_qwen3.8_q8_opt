# Fork patch manifest

This file records the fork features that must survive upstream synchronization.
It is intentionally checked by `test-fork-features` so removal of a user-facing
codec or speculative backend is a test failure rather than a silent merge
regression.

## Integration anchors

| Item | Revision | Purpose |
| --- | --- | --- |
| Fork integration base | `4fc7c239fa2df20de00988424369f6bb7117888d` | Documented TurboQuant/TCQ, fused MTP and split-GPU deployment baseline. |
| Common upstream ancestor | `838374375cb8c467c33c263be4801c19d1089763` | llama.cpp state before the Qwen3.8 forward merge. |
| Pinned upstream merge | `34af94cd9ab277632e27caeec2d41de2fd091b31` | Qwen3.8 implementation and current server protocol baseline. |

## Required fork capabilities

### TurboQuant and TCQ

- GGML types `turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, and `turbo3_tcq`.
- CUDA encode/decode, flash-attention readers, codebook fingerprinting, and
  cache serialization guards.
- Codebook assets and the `TURBO_TCQ_CB` / `TURBO_TCQ_CB2` runtime contract.
- These codecs remain explicit capacity modes. The Qwen3.8 production profile
  defaults to `q8_0` and never selects a Turbo codec automatically.

### Qwen MTP

- Partial MTP-head loading from the primary GGUF.
- Fused target/draft execution and transactional rollback.
- Production default draft limit of two tokens.

### DFlash

- Shared multi-slot drafter context and per-sequence hidden-state routing.
- GPU cross ring, bounded sliding window, checkpoint save/restore, and tape
  replay on the CUDA device that owns each recurrent layer.
- Layer-split safety fixes; tensor split is rejected by the production profile.

### Hybrid recurrent memory

- Qwen hybrid attention/recurrent sequence operations remain atomic.
- Recurrent save/restore uses the backend that owns the layer and preserves
  exact state by default.
- Fork, rollback, cancellation, and slot reuse must not expose state from a
  different sequence generation.

### Hybrid Paged KV and prefix cache

- Q8 attention KV uses device-resident logical pages on the GPU that owns each attention layer.
- Full-state save/restore serializes unique shared pages once and reconstructs sequence sharing and tail CoW relationships.
- Exact token IDs, adapter fingerprints, radix lookup, segmented LRU, page generations, and refcounts guard prefix reuse.
- CPU KV swapping and pageable-host KV storage are not permitted.

### Two-stage CUDA pipeline

- The production boundary is selected from 4-layer groups and keeps MTP plus the output head on GPU 1.
- Systems without CUDA P2P use bounded pinned-host triple buffering, independent D2H/H2D streams, and CUDA events.
- Split-range submission overlaps GPU 0 and GPU 1 without per-layer cross-device ping-pong.
- CUDA Graphs use bounded shape-keyed caches and stable split input buffers.

### Scheduler, sampling, and protocols

- Deadline-fair continuous batching, chunked prefill, resident admission, and prefill starvation bounds remain enabled.
- Backend sampling is the default; raw logits copy is an explicit fallback.
- Qwen3.8 object and string tool arguments, preserved reasoning, Responses continuation, and Anthropic Messages remain supported.

### Target autotune and recovery

- Tuning caches use schema version 2 and bind model and tensor inventory hashes, source and fork revisions, CUDA and driver versions, GPU UUIDs, power limits, PCIe topology, KV codec, and resident sequence count.
- `--autotune quick` rejects unvalidated, tampered, wrong-model, conflicting, or unsafe caches.
- The Qwen production server has a decode/synchronize watchdog and exits after an unrecoverable CUDA compute error.

## Forward-port policy

Upstream changes are merged, never copied over the fork tree. Conflicts in
GGML types, CUDA attention, speculative decoding, model loading, or hybrid
memory are resolved by retaining the capabilities above on the current
upstream interfaces. Every synchronization records its upstream revision here
and runs `test-fork-features` plus the Qwen3.8 profile tests before merge
completion.
