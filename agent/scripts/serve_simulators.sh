#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODEL_PATH="${1:?usage: serve_simulators.sh MODEL_PATH NAME [MODEL_PORT] [SHIM_PORT]}"
MODEL_NAME="${2:?}"
MODEL_PORT="${3:-8000}"
SHIM_PORT="${4:-8710}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
  --port "$MODEL_PORT" --trust-remote-code &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

until curl -sf "http://127.0.0.1:$MODEL_PORT/v1/models" >/dev/null; do sleep 5; done
echo "vLLM up on :$MODEL_PORT"

SHIM_UPSTREAM="http://127.0.0.1:$MODEL_PORT/v1" \
SHIM_MODEL="$MODEL_NAME" \
SHIM_STRIP_THINK=1 \
  python -m llm.shim --port "$SHIM_PORT"
