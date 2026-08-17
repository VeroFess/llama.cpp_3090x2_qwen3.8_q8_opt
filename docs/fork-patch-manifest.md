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

## Forward-port policy

Upstream changes are merged, never copied over the fork tree. Conflicts in
GGML types, CUDA attention, speculative decoding, model loading, or hybrid
memory are resolved by retaining the capabilities above on the current
upstream interfaces. Every synchronization records its upstream revision here
and runs `test-fork-features` plus the Qwen3.8 profile tests before merge
completion.
