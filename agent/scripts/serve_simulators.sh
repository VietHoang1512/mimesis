#!/usr/bin/env bash
# Serve one user simulator, with the shim in front of it.
#
#   bash scripts/serve_simulators.sh /path/to/mimesis-9b  mimesis-9b  8000  8710
#
# Two processes, and the order matters:
#
#   vLLM on $MODEL_PORT   the simulator itself
#   shim on $SHIM_PORT    strips <think> before the gym reads message.content,
#                         and holds the stripped reasoning as SDPO's privileged
#                         channel, keyed by episode
#
# Point SIM_BASE_URL at the SHIM, never at vLLM directly. A gym that receives a
# raw reasoning block uses it verbatim as the user's utterance; the agent reads
# it, the judge grades against it, and the episode is corrupted with no error
# anywhere. See llm/shim.py.
#
# For a hosted simulator, skip vLLM and set SHIM_UPSTREAM to the provider.
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
