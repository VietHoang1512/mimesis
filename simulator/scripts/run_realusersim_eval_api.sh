#!/bin/bash
#SBATCH --job-name=rus-api
#SBATCH --nodes=1
##SBATCH --account=<your-account>
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
export PYTHONUNBUFFERED=1

MODELS="${MODELS:-gpt-5.5 claude-opus-5 claude-sonnet-5 gemini-3.6-flash gemini-3.1-pro-preview}"

SHARD="${SHARD:-}"; NSHARD="${NSHARD:-}"
if [ -n "$SHARD" ] && [ -n "$NSHARD" ]; then
  MODELS=$(echo $MODELS | tr ' ' '\n' | awk -v s="$SHARD" -v n="$NSHARD" 'NR%n==s')
  echo "[rus_api] shard $SHARD/$NSHARD -> $(echo $MODELS | wc -w) model(s)"
fi
CONDITIONS="${CONDITIONS:-correct_profile task_only}"
SEEDS="${SEEDS:-0}"
ASSISTANT="${RUS_ASSISTANT:-gpt-5.5}"
JUDGE="${RUS_JUDGE:-gpt-5.5}"
export RUS_ASSISTANT RUS_JUDGE

WORKERS="${WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-6}"

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  OUTDIR="${OUTDIR:-outputs/realusersim_smoke}"
  EXTRA=(--splits mixed_domain --limit "${LIMIT:-10}")
else
  OUTDIR="${OUTDIR:-outputs/realusersim}"
  EXTRA=()
  [ -n "${SPLITS:-}" ] && EXTRA+=(--splits $SPLITS)
  [ -n "${LIMIT:-}" ]  && EXTRA+=(--limit "$LIMIT")
fi
mkdir -p "$OUTDIR/logs" logs

echo "[rus_api] simulators: $MODELS"
echo "[rus_api] conditions: $CONDITIONS   seeds: $SEEDS"
echo "[rus_api] assistant=$ASSISTANT  judge=$JUDGE  -> $OUTDIR"

for model in $MODELS; do
  for cond in $CONDITIONS; do
    for seed in $SEEDS; do
      stem="$(echo "$model" | tr '/:' '__')-${cond}-seed${seed}"
      gen="$OUTDIR/${stem}_gen.jsonl"

      echo "[rus_api] === simulate $model / $cond / seed$seed ==="
      python realusersim_eval.py simulate \
        --sim-model "$model" --condition "$cond" --seed "$seed" \
        --assistant-model "$ASSISTANT" --workers "$WORKERS" \
        --out-dir "$OUTDIR" "${EXTRA[@]}" 2>&1 | tee -a "$OUTDIR/logs/${stem}_sim.log"

      [ -s "$gen" ] || { echo "[rus_api] no generations for $stem, skipping judge"; continue; }
      echo "[rus_api] === judge $stem ==="
      python realusersim_eval.py judge --gen "$gen" \
        --judge-model "$JUDGE" --workers "$JUDGE_WORKERS" \
        2>&1 | tee -a "$OUTDIR/logs/${stem}_judge.log"
    done
  done
done

echo "[rus_api] building leaderboard"
python realusersim_eval.py aggregate --dir "$OUTDIR"
