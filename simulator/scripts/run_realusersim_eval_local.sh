#!/bin/bash
#SBATCH --job-name=rus-local
#SBATCH --nodes=1
##SBATCH --account=<your-account>
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# RealUserSim PT3 with LOCAL checkpoints as the user simulator. One vLLM server
# per checkpoint, served sequentially; the assistant and the judge still go
# through the hosted endpoint (gpt-5.5) so every row -- local or frontier -- is scored
# by the identical judge against the identical assistant.
#
# See run_realusersim_eval_api.sh for the gpt-4o protocol deviation. It applies
# equally here.
#
# Usage:
#   sbatch --qos=a100_genai_interns_high run_realusersim_eval_local.sh
#   MODELS="osim-8b" sbatch --qos=a100_dev run_realusersim_eval_local.sh
#   SHARD=0 NSHARD=4 sbatch --qos=a100_genai_interns_high run_realusersim_eval_local.sh
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "$MIMESIS_ROOT"
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
# Progress lines are the only way to see a stalled run; do not let them sit in a buffer.
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

MODEL_ROOT=${MODEL_ROOT:-${MIMESIS_MODEL_DIR}}
# Same roster as run_simarena.sh, so the two benchmarks cover an identical set.
MODELS_OURS="\
Qwen3.5-9B-CHATML-290213-original-chat-thoughttrace-fix-426855-all-428848 \
Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-fix-329200-all-427969 \
Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-fix-329200-all-427969-400 \
Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329201-adv-370850 \
Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329200-all-337915 \
Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329200-all-386391 \
Qwen3.5-4B-CHATML-279674-ditto-rl-all-288414 \
Qwen3.5-4B-CHATML-279674-ditto-rl-all-291189 \
Qwen3.5-4B-CHATML-291301-full-ditto-rl-all-300775 \
Qwen3.5-9B-CHATML-279675-ditto-rl-all-290192"
MODELS_BASELINE="osim-8b osim-4b Ditto-8B humanlm-opinion sotopia-rl-qwen-2.5-7B-grpo \
Qwen3.5-9B Qwen3-8B Qwen3.5-4B"
MODELS="${MODELS:-$MODELS_OURS $MODELS_BASELINE}"

# Optional round-robin sharding so N nodes can split the roster.
SHARD="${SHARD:-}"; NSHARD="${NSHARD:-}"
if [ -n "$SHARD" ] && [ -n "$NSHARD" ]; then
  MODELS=$(echo $MODELS | tr ' ' '\n' | awk -v s="$SHARD" -v n="$NSHARD" 'NR%n==s')
  echo "[rus_local] shard $SHARD/$NSHARD -> $(echo $MODELS | wc -w) model(s)"
fi

CONDITIONS="${CONDITIONS:-correct_profile task_only}"
SEEDS="${SEEDS:-0}"
ASSISTANT="${RUS_ASSISTANT:-gpt-5.5}"
JUDGE="${RUS_JUDGE:-gpt-5.5}"
OUTDIR="${OUTDIR:-outputs/realusersim}"
# Local generation is bounded by our own server, not a provider quota, but each
# case still waits on gpt-5.5 for its assistant turns -- so this is the ceiling
# for the whole pipeline, not just for vLLM.
WORKERS="${WORKERS:-16}"
JUDGE_WORKERS="${JUDGE_WORKERS:-6}"
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-8300}"
export LOCAL_VLLM_PORT
VLLM_STARTUP_TRIES="${VLLM_STARTUP_TRIES:-900}"
TP="${TP:-1}"

EXTRA=()
[ -n "${SPLITS:-}" ] && EXTRA+=(--splits $SPLITS)
[ -n "${LIMIT:-}" ]  && EXTRA+=(--limit "$LIMIT")
# Thinking mode is NOT forced: the checkpoint's own chat template decides, so each
# model produces its best response. realusersim_eval.py strips <think> and untagged
# prose CoT, and RAISES ReasoningOnlyTurn when a turn is reasoning with no message
# to extract -- that case is logged per-case rather than silently sent as empty.
# Set ENABLE_THINKING=0/1 explicitly only for an ablation.
THINK_ARG=(); [ -n "${ENABLE_THINKING:-}" ] && THINK_ARG=(--enable-thinking "$ENABLE_THINKING")

mkdir -p "$OUTDIR/logs" logs
echo "[rus_local] models: $MODELS"
echo "[rus_local] conditions: $CONDITIONS  seeds: $SEEDS  assistant=$ASSISTANT judge=$JUDGE"

wait_server_ready() {  # pid url tries -- bails the moment $pid dies
  local pid="$1" url="$2" tries="${3:-300}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 2
    curl -sf --noproxy '*' "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# Several MIMESIS SFT checkpoints ship no chat_template; vLLM serves them fine
# but every chat request then fails, silently producing empty turns. Same guard
# as run_turing_eval.sh / run_simarena.sh.
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
  [ "$status" = "MISSING" ] && echo "agents/tau_usi/chatml.jinja" || echo ""
}

FAILED=""
for base in $MODELS; do
  # Skip the whole checkpoint only if every condition x seed is already judged.
  need=0
  for cond in $CONDITIONS; do for seed in $SEEDS; do
    [ -s "$OUTDIR/local_${base}-${cond}-seed${seed}_judged.jsonl" ] || need=1
  done; done
  if [ "$need" = "0" ]; then echo "[rus_local] skip (done): $base"; continue; fi

  path="$MODEL_ROOT/$base"
  [ -d "$path" ] || { echo "[rus_local] MISSING checkpoint: $path"; FAILED="$FAILED $base"; continue; }
  tmpl=$(resolve_chat_template "$path")
  echo "[rus_local] serving $base on :$LOCAL_VLLM_PORT ${tmpl:+(chat-template $tmpl)}"
  args=(--port "$LOCAL_VLLM_PORT" --served-model-name "$base"
        --tensor-parallel-size "$TP" --gpu-memory-utilization 0.9 --max-model-len 32768)
  [ -n "$tmpl" ] && args+=(--chat-template "$tmpl")
  vllm serve "$path" "${args[@]}" > "logs/vllm_rus_${base}.log" 2>&1 &
  vpid=$!
  if ! wait_server_ready "$vpid" "http://localhost:$LOCAL_VLLM_PORT/health" "$VLLM_STARTUP_TRIES"; then
    echo "[rus_local] ERROR: vLLM never came up for $base (logs/vllm_rus_${base}.log)"
    kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
    FAILED="$FAILED $base"; continue
  fi

  for cond in $CONDITIONS; do
    for seed in $SEEDS; do
      stem="local_${base}-${cond}-seed${seed}"
      gen="$OUTDIR/${stem}_gen.jsonl"
      echo "[rus_local] === simulate $base / $cond / seed$seed ==="
      python realusersim_eval.py simulate \
        --sim-model "local/$base" --condition "$cond" --seed "$seed" \
        --assistant-model "$ASSISTANT" --workers "$WORKERS" \
        --out-dir "$OUTDIR" "${EXTRA[@]}" "${THINK_ARG[@]}" \
        2>&1 | tee -a "$OUTDIR/logs/${stem}_sim.log"

      [ -s "$gen" ] || { echo "[rus_local] no generations for $stem"; FAILED="$FAILED $stem"; continue; }
      echo "[rus_local] === judge $stem ==="
      python realusersim_eval.py judge --gen "$gen" \
        --judge-model "$JUDGE" --workers "$JUDGE_WORKERS" \
        2>&1 | tee -a "$OUTDIR/logs/${stem}_judge.log"
    done
  done

  kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
done

[ -n "$FAILED" ] && echo "[rus_local] FAILED:$FAILED"
python realusersim_eval.py aggregate --dir "$OUTDIR"
