#!/bin/bash
#SBATCH --job-name=simarena
#SBATCH --nodes=1
##SBATCH --account=<your-account>
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

ANNOTATION_ID="${ANNOTATION_ID:-document_creation_annotations_full}"

set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "${SIMULATORARENA_ROOT:?clone microsoft/SimulatorArena and set SIMULATORARENA_ROOT -- see eval/simarena/SETUP.md}/simulation"

export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

ASSISTANT="${ASSISTANT:-gpt-5.5}"
VERSION="${VERSION:-zero-shot-cot-user-profile}"
PROFILE="${PROFILE:-preference_and_writing_interaction_style}"
NUM_CONV="${NUM_CONV:-}"

SIMS_GATEWAY="gemini-3.1-pro-preview claude-opus-5 gpt-5.5 gemini-3.6-flash claude-sonnet-5"
MODEL_ROOT=${MIMESIS_MODEL_DIR}
SIMS_OURS="mimesis-9b mimesis-4b mimesis-4b-step400"
SIMS_BASELINE="osim-8b osim-4b Ditto-8B humanlm-opinion sotopia-rl-qwen-2.5-7B-grpo \
Qwen3.5-9B Qwen3-8B Qwen3.5-4B"
SIMS_LOCAL=$(for m in $SIMS_OURS $SIMS_BASELINE; do printf 'local/%s ' "$m"; done)

case "${SIMS_SET:-}" in
  local)   SIMS="${SIMS:-$SIMS_LOCAL}" ;;
  all)     SIMS="${SIMS:-$SIMS_GATEWAY $SIMS_LOCAL}" ;;
esac

SIMS="${SIMS:-$SIMS_GATEWAY}"
SHARD="${SHARD:-}"; NSHARD="${NSHARD:-}"
if [ -n "$SHARD" ] && [ -n "$NSHARD" ]; then
  SIMS=$(echo $SIMS | tr ' ' '\n' | awk -v s="$SHARD" -v n="$NSHARD" 'NR%n==s')
  echo "[simarena] shard $SHARD/$NSHARD -> $(echo $SIMS | wc -w) simulator(s)"
fi
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-8200}"
export LOCAL_VLLM_PORT
VLLM_STARTUP_TRIES="${VLLM_STARTUP_TRIES:-900}"

NUM_ARG=(); [ -n "$NUM_CONV" ] && NUM_ARG=(--num_conversations "$NUM_CONV")
echo "[simarena] assistant=$ASSISTANT  version=$VERSION  profile=$PROFILE"
echo "[simarena] simulators: $SIMS"

wait_server_ready() {
  local pid="$1" url="$2" tries="${3:-300}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 2
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

resolve_chat_template() {
  local status
  status=$(python - "$1" <<'PY' 2>/dev/null
import sys
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    print("MISSING" if not getattr(tok, "chat_template", None) else "OK")
except Exception:
    print("OK")
PY
)
  [ "$status" = "MISSING" ] && echo "../../agents/tau_usi/chatml.jinja" || echo ""
}

FAILED=""
for sim in $SIMS; do
  out="output/${ANNOTATION_ID}/${sim}/${VERSION}-up-${PROFILE}_for_benchmarking.json"
  if [ -s "$out" ]; then
    echo "[simarena] skip (exists): $sim"; continue
  fi

  vpid=""
  if [[ "$sim" == local/* ]]; then
    base="${sim#local/}"
    path="$MODEL_ROOT/$base"
    tmpl=$(resolve_chat_template "$path")
    echo "[simarena] serving $base on :$LOCAL_VLLM_PORT ${tmpl:+(chat-template $tmpl)}"
    args=(--port "$LOCAL_VLLM_PORT" --served-model-name "$base"
          --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 32768)
    [ -n "$tmpl" ] && args+=(--chat-template "$tmpl")
    vllm serve "$path" "${args[@]}" > "logs/vllm_${base}.log" 2>&1 &
    vpid=$!
    if ! wait_server_ready "$vpid" "http://localhost:$LOCAL_VLLM_PORT/health" "$VLLM_STARTUP_TRIES"; then
      echo "[simarena] ERROR: vLLM never came up for $base (logs/vllm_${base}.log)"
      kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
      FAILED="$FAILED $sim"; continue
    fi
  fi

  echo "[simarena] === simulator=$sim  assistant=$ASSISTANT ==="
  python user_simulation_document_creation.py \
    --annotation_id="$ANNOTATION_ID" \
    --version="$VERSION" \
    --user_profile_version="$PROFILE" \
    --user_model="$sim" \
    --benchmarking \
    --allowed_models "$ASSISTANT" \
    "${NUM_ARG[@]}" 2>&1 | tee "logs/simarena_${sim//\//_}.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && FAILED="$FAILED $sim"

  [ -n "$vpid" ] && { kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null; }
done

if [ -n "$FAILED" ]; then
  echo "[simarena] FAILED:$FAILED"; exit 1
fi
echo "[simarena] all simulators done"
