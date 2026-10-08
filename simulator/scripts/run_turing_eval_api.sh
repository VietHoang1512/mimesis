#!/bin/bash
#SBATCH --job-name=turing-api
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

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

MODELS="${MODELS:-gpt-5.5 claude-opus-5 claude-sonnet-5 gemini-3.6-flash gemini-3.1-pro-preview}"
REASONING_EFFORT="${OPENAI_AGENT_REASONING_EFFORT:-low}"

TEST_PARQUET="${TEST_PARQUET:-turing-rl/data/prism/prism_history_s42_sft40_grpo60/test.parquet}"

SEEDS="${SEEDS:-0 1 2}"

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  LIMIT="${LIMIT:-20}"
  OUTDIR=${OUTDIR:-outputs/turing_smoke}
else
  LIMIT="${LIMIT:-}"
  OUTDIR=${OUTDIR:-outputs/turing}
fi

GEN_WORKERS=${GEN_WORKERS:-8}
SCORE_WORKERS=${SCORE_WORKERS:-4}
MAX_TOKENS=${MAX_TOKENS:-4096}

mkdir -p "$OUTDIR/logs"
LIMIT_ARG=(); [ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")

N_MODELS=$(echo $MODELS | wc -w)
echo "[turing_api] $N_MODELS model(s): $MODELS"
echo "[turing_api] heldout=$TEST_PARQUET ${LIMIT:+(first $LIMIT rows)}"

for model in $MODELS; do
  for seed in $SEEDS; do
    base=$(echo "$model" | tr '/:' '__')-seed${seed}
    gen="$OUTDIR/${base}_gen.json"

    if [ -s "$gen" ]; then
      echo "[turing_api] skip generate (exists): $gen"
    else
      echo "[turing_api] === generating $model seed=$seed ==="
      python turing_eval.py generate \
        --test-parquet "$TEST_PARQUET" \
        --api-model "$model" --reasoning-effort "$REASONING_EFFORT" \
        --workers "$GEN_WORKERS" --max-tokens "$MAX_TOKENS" \
        "${LIMIT_ARG[@]}" --out "$gen" 2>&1 | tee "$OUTDIR/logs/${base}_generate.log"
    fi

    [ -s "$gen" ] || { echo "[turing_api] no generations, skipping score: $base"; continue; }
    if [ -s "$OUTDIR/${base}_turing_summary.json" ]; then
      echo "[turing_api] skip score (exists): $base"; continue
    fi
    echo "[turing_api] === scoring $base ==="
    python turing_eval.py score --gen "$gen" --out-dir "$OUTDIR" \
      --workers "$SCORE_WORKERS" 2>&1 | tee "$OUTDIR/logs/${base}_score.log"
  done
done

echo "[turing_api] building leaderboard"
python turing_eval.py aggregate --dir "$OUTDIR"
echo "[turing_api] done. Per-run artifacts + turing_summary.md under $OUTDIR/"
