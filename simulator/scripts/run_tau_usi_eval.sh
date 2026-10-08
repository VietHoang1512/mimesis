#!/bin/bash
#SBATCH --job-name=tau-usi
#SBATCH --nodes=1
##SBATCH --account=<your-account>
##SBATCH --qos=<your-qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "$MIMESIS_ROOT"
# ── Ports ─────────────────────────────────────────────────────────────────────
RUNTIME_PORT=${RUNTIME_PORT:-8005}
VLLM_PORT=${VLLM_PORT:-8100}

# ── Fixed agent + judge + data + runtime ──────────────────────────────────────
export TAU_USI_DATA_DIR="$PWD/data/tau-usi/data"
export RUNTIME_SERVICE_URL="http://localhost:$RUNTIME_PORT"
export TAU_USI_AGENT_MODEL="${TAU_USI_AGENT_MODEL:-gpt-5-5}"
export TAU_USI_AGENT_REASONING_EFFORT="${TAU_USI_AGENT_REASONING_EFFORT:-low}"
export OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-gpt-5-5}"
export JUDGE_MODEL_NAME="$OPENAI_MODEL_NAME"
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
unset OPENAI_BASE_URL
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.}"
export USER_SIM_ENABLE_THINKING="${USER_SIM_ENABLE_THINKING:-1}"
export TURNOFF_THINK=$([ "$USER_SIM_ENABLE_THINKING" = "1" ] && echo 0 || echo 1)
# ── What to run ───────────────────────────────────────────────────────────────
SMOKE=${SMOKE:-0}
DOMAINS_FULL="retail:0-114,airline:0-49"
if [ "$SMOKE" = "1" ]; then
  MODELS="${MODELS:-sunweiwei/Ditto-8B}"
  SEEDS="${SEEDS:-0}"
  DOMAINS="${DOMAINS:-retail:0-2,airline:0-1}"
else
  MODELS="${MODELS:-${MIMESIS_MODEL_DIR}/mimesis-9b}"
  SEEDS="${SEEDS:-0 1 2}"
  DOMAINS="${DOMAINS:-$DOMAINS_FULL}"
fi

# ── Rollout / serving knobs ───────────────────────────────────────────────────
WORKERS=${WORKERS:-16}
VLLM_TP=${VLLM_TP:-1}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.9}
VLLM_MAX_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_STARTUP_TRIES=${VLLM_STARTUP_TRIES:-900}
PROMPT_LEN=${PROMPT_LEN:-24000}
RESPONSE_LEN=${RESPONSE_LEN:-$([ "$USER_SIM_ENABLE_THINKING" = "1" ] && echo $((1024 * 16)) || echo 2048)}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

API_MODELS="${USER_SIM_API:-}"

if [ "$SMOKE" = "1" ]; then
  OUTDIR=${OUTDIR:-outputs/tau_usi_smoke}
elif [ "$DOMAINS" != "$DOMAINS_FULL" ]; then
  OUTDIR=${OUTDIR:-outputs/tau_usi_$(echo "$DOMAINS" | tr -c '[:alnum:]' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')}
else
  OUTDIR=${OUTDIR:-outputs/tau_usi}
fi
mkdir -p "$OUTDIR/logs"

# ── Start the runtime service ─────────────────────────────────────────────────
echo "[tau_usi] starting tau-bench runtime service on :$RUNTIME_PORT"
python -m agents.tau_usi.runtime_service --port "$RUNTIME_PORT" > "$OUTDIR/logs/runtime_service.log" 2>&1 &
RUNTIME_PID=$!
VLLM_PID=""
cleanup() {
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
  kill "$RUNTIME_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1" tries="${2:-180}" i
  for ((i = 0; i < tries; i++)); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

wait_server_ready() {
  local pid="$1" url="$2" tries="${3:-300}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 2
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if ! wait_http "http://localhost:$RUNTIME_PORT/health" 60; then
  echo "[tau_usi] ERROR: runtime service did not become healthy; see $OUTDIR/logs/runtime_service.log"
  exit 1
fi
echo "[tau_usi] runtime service healthy."

# ── Rollout + score one (model, seed) ─────────────────────────────────────────
declare -a FAILED=()
declare -a UNSCORED=()

rollout_and_score() {
  local model="$1" seed="$2" base label out
  base=$(basename "$model")
  label="${base}-repetition-${REPETITION_PENALTY}-thinking-${USER_SIM_ENABLE_THINKING:-0}-resp${RESPONSE_LEN}-seed${seed}"
  out="$OUTDIR/${label}_task_results.json"
  export EVAL_SEED=$seed
  echo "[tau_usi] === $label : rollout over $DOMAINS (workers=$WORKERS) ==="
  python -m agents.tau_usi.run_eval \
    --user-sim-model "$model" \
    --domains "$DOMAINS" \
    --workers "$WORKERS" \
    --prompt-length "$PROMPT_LEN" \
    --response-length "$RESPONSE_LEN" \
    --out "$out" 2>&1 | tee "$OUTDIR/logs/${label}_rollout.log"
  if [ ! -s "$out" ]; then
    echo "[tau_usi] ERROR: no rollout results at $out -- skipping score for $label"
    UNSCORED+=("$label")
    return
  fi
  echo "[tau_usi] === $label : scoring ==="
  python -m agents.tau_usi.usi_metric score "$out" --label "$label" --out-dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/logs/${label}_score.log"
}

if [ -n "$API_MODELS" ]; then
  unset OPENAI_AGENT_BASE_URL
  export OPENAI_AGENT_REASONING_EFFORT="${OPENAI_AGENT_REASONING_EFFORT:-low}"
  echo "[tau_usi] API user-sim mode: $API_MODELS"
  for model in $API_MODELS; do
    export OPENAI_AGENT_MODEL="$model"
    for seed in $SEEDS; do rollout_and_score "$model" "$seed"; done
  done
else
  unset OPENAI_AGENT_REASONING_EFFORT

  resolve_chat_template() {
    if [ -n "$CHAT_TEMPLATE" ]; then echo "$CHAT_TEMPLATE"; return; fi
    local status
    status=$(python - "$1" <<'PY' 2>/dev/null
import sys
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    print("MISSING" if not getattr(tok, "chat_template", None) else "OK")
except Exception:
    print("OK")  # can't tell (e.g. gated/auth); let vLLM resolve it
PY
)
    [ "$status" = "MISSING" ] && echo "agents/tau_usi/chatml.jinja" || echo ""
  }

  for model in $MODELS; do
    base=$(basename "$model")
    tmpl=$(resolve_chat_template "$model")
    [ -n "$tmpl" ] && echo "[tau_usi] '$model' ships no chat template -> using $tmpl"
    echo "[tau_usi] serving user-sim '$model' on :$VLLM_PORT (tp=$VLLM_TP)"
    vllm_args=(--port "$VLLM_PORT" --served-model-name "$model"
               --tensor-parallel-size "$VLLM_TP" --gpu-memory-utilization "$VLLM_GPU_UTIL"
               --max-model-len "$VLLM_MAX_LEN" $VLLM_EXTRA_ARGS)
    [ -n "$tmpl" ] && vllm_args+=(--chat-template "$tmpl")
    vllm serve "$model" "${vllm_args[@]}" > "$OUTDIR/logs/vllm_${base}.log" 2>&1 &
    VLLM_PID=$!

    if ! wait_server_ready "$VLLM_PID" "http://localhost:$VLLM_PORT/health" "$VLLM_STARTUP_TRIES"; then
      echo "[tau_usi] ERROR: vLLM for '$model' did not come up; see $OUTDIR/logs/vllm_${base}.log — skipping"
      kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""
      FAILED+=("$base")
      continue
    fi
    echo "[tau_usi] vLLM ready for '$model'."

    export OPENAI_AGENT_BASE_URL="http://localhost:$VLLM_PORT/v1"
    export OPENAI_AGENT_MODEL="$model"
    export OPENAI_AGENT_API_KEY="${OPENAI_AGENT_API_KEY:-EMPTY}"

    for seed in $SEEDS; do rollout_and_score "$model" "$seed"; done

    kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""
    unset OPENAI_AGENT_BASE_URL OPENAI_AGENT_MODEL
  done
fi

echo "[tau_usi] building summary table"
python -m agents.tau_usi.aggregate_usi --dir "$OUTDIR"
echo "[tau_usi] done. Per-run artifacts + usi_summary.md under $OUTDIR/"

if [ ${#FAILED[@]} -gt 0 ] || [ ${#UNSCORED[@]} -gt 0 ]; then
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[tau_usi] FAILED: ${#FAILED[@]} model(s) never served --"
    for b in "${FAILED[@]}"; do
      echo "[tau_usi]   $b  (see $OUTDIR/logs/vllm_$b.log)"
    done
  fi
  if [ ${#UNSCORED[@]} -gt 0 ]; then
    echo "[tau_usi] UNSCORED: ${#UNSCORED[@]} run(s) produced no rollout results --"
    for l in "${UNSCORED[@]}"; do
      echo "[tau_usi]   $l  (see $OUTDIR/logs/${l}_rollout.log)"
    done
  fi
  exit 1
fi
