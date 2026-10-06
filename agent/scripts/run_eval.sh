#!/usr/bin/env bash
# Evaluate one agent checkpoint against one user simulator across the 15 gyms.
#
#   bash scripts/run_eval.sh /path/to/checkpoint my-agent
#
# Assumes a simulator is already being served with its shim
# (scripts/serve_simulators.sh) and that SIM_BASE_URL points at that shim.
#
# An FSDP training checkpoint is merged to HF format first; pass an HF directory
# and the merge is skipped. The merged agent is then served with vLLM and
# eval.py drives agent <-> gym turns against it.
#
# Repeat this per simulator to build the main table, then:
#   python eval/digest.py -o results/main_results.json
#   python eval/make_main_table.py --digest results/main_results.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$AGENT_ROOT"

CKPT="${1:?usage: run_eval.sh CHECKPOINT AGENT_NAME [AGENT_PORT]}"
AGENT_NAME="${2:?}"
AGENT_PORT="${3:-8100}"

# 15 evaluation sets. The eight travel* variants differ in how many preference
# slots the simulated user holds back; table3.py collapses them into one column.
ENVS="travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222 \
function intention persuasion tau telepathy turtle bamboogle"

HF_CKPT="$CKPT"
if [ -d "$CKPT/actor" ]; then
  HF_CKPT="$CKPT/actor_hf"
  echo "merging FSDP shards -> $HF_CKPT"
  python eval/merge.py merge --backend fsdp \
    --local_dir "$CKPT/actor" --target_dir "$HF_CKPT"
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$HF_CKPT" --served-model-name "$AGENT_NAME" \
  --port "$AGENT_PORT" --trust-remote-code &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$AGENT_PORT/v1/models" >/dev/null; do sleep 5; done

export EVAL_OUTPUT_DIR="$AGENT_EVAL_DIR"
export AGENT_BASE_URL="http://127.0.0.1:$AGENT_PORT/v1"
mkdir -p "$AGENT_EVAL_DIR/outputs"

RUN_NAME="${AGENT_NAME}__${SIM_MODEL_NAME}__J-self"
python eval/eval.py \
  --model_name "$AGENT_NAME" \
  --port "$AGENT_PORT" \
  --max_turns 16 \
  --pass_k 1 \
  --temperature 0 \
  --envs $ENVS \
  --save_name "$AGENT_EVAL_DIR/outputs/results_${RUN_NAME}"

echo "done -> $AGENT_EVAL_DIR/outputs/results_${RUN_NAME}_results.json"
