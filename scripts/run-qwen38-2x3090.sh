#!/usr/bin/env bash
set -euo pipefail

model_path="${QWEN38_MODEL:-/models/Qwen3.8-27B-Q8_0.gguf}"
server_bin="${QWEN38_SERVER:-./build-qwen38-3090/bin/llama-server}"

exec "${server_bin}" \
    --model "${model_path}" \
    --profile qwen38-27b-q8-2x3090 \
    --host "${QWEN38_HOST:-0.0.0.0}" \
    --port "${QWEN38_PORT:-8080}" \
    "$@"

