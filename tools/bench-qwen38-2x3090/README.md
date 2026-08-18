# Qwen3.8 serving benchmark

The runner records the full synthetic request and SSE response trace, server debug state, git state, GPU clocks, power limits, topology, P2P capability, and deployment axes in one JSON artifact. The CSV contains the comparable summary rows.

Run the core production suite:

```bash
python3 tools/bench-qwen38-2x3090/bench.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B-Q8_0.gguf \
  --model-sha256 a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348 \
  --source-commit dc57e5ecb4d412243f915202143f0d9adf482e67 \
  --server-command './build-qwen38-3090/bin/llama-server --model /models/Qwen3.8-27B-Q8_0.gguf --profile qwen38-27b-q8-2x3090 --host 127.0.0.1 --port 8080' \
  --variant production \
  --speculation adaptive \
  --cuda-graph on \
  --pipeline on \
  --kv-mode q8_0 \
  --split-boundary 36/28
```

Add `--strict-manifest` for formal runs. It fails the run when the declared Q8 KV mode, split boundary, pipeline depth, CUDA Graph activity, or pinned-host transfer counters disagree with the server's actual debug state, or when commit, model hash, command, topology, or P2P evidence is missing.

Use `--quick` for a short smoke run. Use `--suite full` to include the 32K, 128K, and near-262K single-request cases. Select one or more cases with repeated `--scenario` arguments. Synthetic code prompts are calibrated through `/tokenize` before timing; each trace records the target, measured content tokens, and calibration attempts. `--no-prompt-calibration` is available only for endpoint compatibility checks.

Available workloads:

- `single_short`
- `single_32k`
- `single_128k`
- `single_262k`
- `eight_decode`
- `mixed_prefill_decode`
- `eight_varied_prefill`
- `shared_prefix_fork`
- `coding_agent_tool_loop`

Configuration comparisons use the same workload with different deployment axes. Run separate server deployments and label each artifact with:

- `--speculation adaptive|mtp|dflash|none`
- `--cuda-graph on|off`
- `--pipeline on|off`
- `--kv-mode q8_0|turbo|tcq`
- `--split-boundary auto|36/28|...`
- `--variant legacy|upstream-eager|ablation-name|production`

An optional `--manifest-json` file can add deployment data that the HTTP and host interfaces do not expose. A failed request remains in the JSON trace and makes the runner exit non-zero.

Run the live protocol fixtures against the same production profile:

```bash
python3 -B tools/bench-qwen38-2x3090/protocol_fixtures.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B-Q8_0.gguf \
  --output-json qwen38-protocol-fixtures.json
```

The protocol artifact covers OpenAI object and JSON-string tool arguments, parallel and multi-step tool history, reasoning preservation on and off, Unicode and escaped arguments, expected malformed-argument rejection, byte-wise SSE consumption, Responses `previous_response_id`, and Anthropic/Claude Code Messages history.
