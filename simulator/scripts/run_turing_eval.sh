#!/bin/bash
#SBATCH --job-name=turing
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
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"

export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# ── Heldout set ───────────────────────────────────────────────────────────────
TEST_PARQUET="${TEST_PARQUET:-turing-rl/data/prism/prism_history_s42_sft40_grpo60/test.parquet}"

# ── What to run ───────────────────────────────────────────────────────────────
SMOKE=${SMOKE:-0}
MODEL_ROOT=${MIMESIS_MODEL_DIR}
if [ "$SMOKE" = "1" ]; then
  MODELS="${MODELS:-$MODEL_ROOT/mimesis-9b}"
  LIMIT="${LIMIT:-20}"
else
  MODELS="${MODELS:-\
$MODEL_ROOT/mimesis-9b}"
  LIMIT="${LIMIT:-}"
fi

SEEDS="${SEEDS:-0 1 2}"

ADOPT_LEGACY=${ADOPT_LEGACY:-1}

# ── Knobs ─────────────────────────────────────────────────────────────────────
OUTDIR=${OUTDIR:-outputs/turing}
if [ "$SMOKE" = "1" ]; then
  OUTDIR=${OUTDIR_SMOKE:-outputs/turing_smoke}
elif [ -n "$LIMIT" ]; then
  OUTDIR="${OUTDIR%/}_n$LIMIT"
fi
GEN_WORKERS=${GEN_WORKERS:-32}
SCORE_WORKERS=${SCORE_WORKERS:-4}
MAX_TOKENS=${MAX_TOKENS:-16384}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.9}
VLLM_MAX_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_STARTUP_TRIES=${VLLM_STARTUP_TRIES:-900}
BASE_PORT=${BASE_PORT:-8100}
N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "${N_GPUS:-0}" -lt 1 ] && N_GPUS=1

mkdir -p "$OUTDIR/logs"
LIMIT_ARG=(); [ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")

N_MODELS=$(echo $MODELS | wc -w)
echo "[turing] $N_MODELS model(s), $N_GPUS GPU(s), heldout=$TEST_PARQUET ${LIMIT:+(first $LIMIT rows)}"
echo "[turing] phase 1 generate (parallel) -> phase 2 score (serial, throttled)"

wait_server_ready() {
  local pid="$1" url="$2" tries="${3:-300}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 2
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
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

declare -a GEN_FILES=()
idx=0
N_NEW=0
N_SKIP=0
for model in $MODELS; do
  base=$(basename "$model")

  legacy="$OUTDIR/${base}_gen.json"
  legacy_sum="$OUTDIR/${base}_turing_summary.json"
  if [ "$ADOPT_LEGACY" = "1" ] && [ -s "$legacy" ] && [ ! -s "$OUTDIR/${base}-seed0_gen.json" ]; then
    echo "[turing] adopting unseeded $base as seed0"
    mv "$legacy" "$OUTDIR/${base}-seed0_gen.json"
    [ -s "$legacy_sum" ] && mv "$legacy_sum" "$OUTDIR/${base}-seed0_turing_summary.json"
    [ -s "$OUTDIR/${base}_turing_items.json" ] && \
      mv "$OUTDIR/${base}_turing_items.json" "$OUTDIR/${base}-seed0_turing_items.json"
  fi

  need=""
  for seed in $SEEDS; do
    gen="$OUTDIR/${base}-seed${seed}_gen.json"
    GEN_FILES+=("$gen")
    if [ -s "$gen" ]; then
      echo "[turing] skip generate (exists): $(basename "$gen")"
      N_SKIP=$((N_SKIP + 1))
    else
      need="$need $seed"
    fi
  done
  [ -z "$need" ] && { idx=$((idx + 1)); continue; }
  N_NEW=$((N_NEW + $(echo $need | wc -w)))

  gpu=$((idx % N_GPUS))
  port=$((BASE_PORT + gpu))
  tmpl=$(resolve_chat_template "$model")
  (
    [ -n "$tmpl" ] && echo "[turing] '$base' ships no chat template -> using $tmpl"
    echo "[turing] gpu$gpu :$port serving $base (seeds:$need)"
    vllm_args=(--port "$port" --served-model-name "$base"
               --tensor-parallel-size 1 --gpu-memory-utilization "$VLLM_GPU_UTIL"
               --max-model-len "$VLLM_MAX_LEN")
    [ -n "$tmpl" ] && vllm_args+=(--chat-template "$tmpl")
    CUDA_VISIBLE_DEVICES=$gpu vllm serve "$model" "${vllm_args[@]}" \
      > "$OUTDIR/logs/vllm_${base}.log" 2>&1 &
    vpid=$!
    if ! wait_server_ready "$vpid" "http://localhost:$port/health" "$VLLM_STARTUP_TRIES"; then
      echo "[turing] ERROR: vLLM for '$base' never came up; see $OUTDIR/logs/vllm_${base}.log"
      kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
      exit 1
    fi
    for seed in $need; do
      python turing_eval.py generate \
        --test-parquet "$TEST_PARQUET" \
        --model "$base" --base-url "http://localhost:$port/v1" \
        --workers "$GEN_WORKERS" --max-tokens "$MAX_TOKENS" --seed "$seed" \
        "${LIMIT_ARG[@]}" --out "$OUTDIR/${base}-seed${seed}_gen.json" \
        2>&1 | tee "$OUTDIR/logs/${base}-seed${seed}_generate.log"
    done
    kill "$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
  ) &

  idx=$((idx + 1))
  [ $((idx % N_GPUS)) -eq 0 ] && wait
done
wait
echo "[turing] phase 1 done ($N_NEW attempted, $N_SKIP already complete)."

declare -a FAILED=()
for gen in "${GEN_FILES[@]}"; do
  [ -s "$gen" ] || FAILED+=("$(basename "$gen" _gen.json)")
done

declare -a UNSCORED=()
for gen in "${GEN_FILES[@]}"; do
  [ -s "$gen" ] || { echo "[turing] no generations, skipping score: $gen"; continue; }
  base=$(basename "$gen" _gen.json)
  if [ -s "$OUTDIR/${base}_turing_summary.json" ]; then
    echo "[turing] skip score (exists): $base"; continue
  fi
  echo "[turing] === scoring $base ==="
  python turing_eval.py score --gen "$gen" --out-dir "$OUTDIR" \
    --workers "$SCORE_WORKERS" 2>&1 | tee "$OUTDIR/logs/${base}_score.log"
  [ -s "$OUTDIR/${base}_turing_summary.json" ] || UNSCORED+=("$base")
done

echo "[turing] building leaderboard"
python turing_eval.py aggregate --dir "$OUTDIR"
echo "[turing] done. Per-run artifacts + turing_summary.md under $OUTDIR/"

if [ ${#FAILED[@]} -gt 0 ] || [ ${#UNSCORED[@]} -gt 0 ]; then
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[turing] FAILED: ${#FAILED[@]}/$N_MODELS model(s) produced no generations --"
    for b in "${FAILED[@]}"; do
      echo "[turing]   $b  (see $OUTDIR/logs/vllm_$b.log)"
    done
  fi
  if [ ${#UNSCORED[@]} -gt 0 ]; then
    echo "[turing] UNSCORED: ${#UNSCORED[@]} model(s) generated but the judge failed --"
    for b in "${UNSCORED[@]}"; do
      echo "[turing]   $b  (see $OUTDIR/logs/${b}_score.log)"
    done
    echo "[turing]   generations are on disk; requeue to resume at scoring."
  fi
  exit 1
fi
