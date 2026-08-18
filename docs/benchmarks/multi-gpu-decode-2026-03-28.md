# Multi-GPU decode data point: 2x3090 + 3080 Ti

This report provides a multi-GPU data point for long-context decode performance. Most published results are for a single 3090; this setup uses 3 GPUs.

## Hardware
- 2x NVIDIA RTX 3090 (24 GB)
- 1x NVIDIA RTX 3080 Ti (12 GB)

## Model
- `unsloth/Qwen3.5-27B-GGUF:UD-Q5_K_XL`

## Configuration
Both runs used the same serving shape:
- `-ngl 99`
- `--tensor-split 2,2,1`
- `--ctx-size 674288`
- `--parallel 3`
- `--flash-attn on`

Method:
- Cold server start before each run
- Identical prompt file for all runs
- Identical OpenAI-compatible request

## Short prompt (~1k tokens)
Server-reported prompt length: **1,038 tokens**

```text
┌────────────────────────────────────┬──────────────────┬────────────┬────────────┐
│ Config                             │ Method           │ Prompt t/s │ Decode t/s │
├────────────────────────────────────┼──────────────────┼────────────┼────────────┤
│ upstream llama.cpp + q8_0          │ cold server run  │ 941.15     │ 31.10      │
├────────────────────────────────────┼──────────────────┼────────────┼────────────┤
│ this branch + q8_0                 │ cold server run  │ 940.51     │ 30.61      │
├────────────────────────────────────┼──────────────────┼────────────┼────────────┤
│ this branch + turbo3 LA-1          │ cold server run  │ 933.36     │ 29.56      │
└────────────────────────────────────┴──────────────────┴────────────┴────────────┘
```

## Long prompt (~200k tokens)
Server-reported prompt length: **200,094 tokens**

```text
┌────────────────────────────────────┬──────────────────┬────────────┬────────────┐
│ Config                             │ Method           │ Prompt t/s │ Decode t/s │
├────────────────────────────────────┼──────────────────┼────────────┼────────────┤
│ upstream llama.cpp + q8_0          │ cold server run  │ 612.91     │ 16.92      │
├────────────────────────────────────┼──────────────────┼────────────┼────────────┤
│ this branch + q8_0                 │ cold server run  │ 619.35     │ 11.15      │
├────────────────────────────────────┼──────────────────┼────────────┼────────────┤
│ this branch + turbo3 LA-1          │ cold server run  │ 609.43     │ 7.46       │
└────────────────────────────────────┴──────────────────┴────────────┴────────────┘
```

## Observations
Short-context performance matches upstream. Long-context decode does not.

On this 3-GPU setup, the branch appears to need more work on multi-GPU decode performance at large context:
- Branch `q8_0` decode is slower than upstream `q8_0`.
- Branch `turbo3 LA-1` decode is slower again.
- Prompt speed remains close in all cases, indicating a decode-side issue.

These results describe a specific multi-GPU configuration and do not necessarily reflect single-GPU performance.
