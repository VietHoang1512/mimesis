#!/bin/bash
#SBATCH --job-name=turing-api
#SBATCH --nodes=1
##SBATCH --account=<your-account>
##SBATCH --qos=a100_dev
##SBATCH --qos=<your-qos>   # was: a100_genai_interns_high
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
# Turing-test eval of API models AS THE USER-SIMULATOR. The Turing analogue of
# run_tau_usi_eval_api.sh, evaluating the same five models so the two benchmarks
# stay directly comparable.
#
# Unlike run_turing_eval.sh there is NO vLLM and NO GPU: generation goes through
# the hosted endpoint transport in api.py (WITH_X2P=1), which routes per family --
# gpt-* -> Azure Responses, claude-* -> /v1/messages, gemini-* -> Vertex. The
# judge is the same gpt-5.6 on the judge endpoint. (If your qos requires a GPU allocation,
# uncomment the two lines below.)
##SBATCH --gres=gpu:1
##SBATCH --gpus-per-node=1
#
# Both phases here are API-bound and share rate limits, so unlike the local
# script there is nothing to fan out: models run one after another.
#
# Prereq -- build the heldout set once:
#   cd turing-rl && bash bash_scripts/data/generate_data.sh prism gpt-5.6
#
# Usage:
#   sbatch run_turing_eval_api.sh                                  # 5 models
#   MODELS="gpt-5.5" bash run_turing_eval_api.sh                   # single model
#   SMOKE=1 MODELS="gemini-3.6-flash" bash run_turing_eval_api.sh  # 20-target canary
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "$MIMESIS_ROOT"
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"

# --export=ALL forwards the SUBMITTING shell's https_proxy, which points at that
# host's own outbound proxy listener and is dead on any compute node. api.py is immune (it
# passes --proxy explicitly to curl), but the JUDGE is not: turing-rl's client
# reaches the configured endpoint through urllib, which honours the variable and then
# burns all 8 retries on URLError. That is why frontier runs sat at 0/10 scored
# on every QOS while the local script -- which already has this line -- scored
# normally. Compute nodes reach the judge endpoint directly, so drop the stale values.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# Same five models as run_tau_usi_eval_api.sh / run_eval_api.sh.
MODELS="${MODELS:-gpt-5.5 claude-opus-5 claude-sonnet-5 gemini-3.6-flash gemini-3.1-pro-preview}"
# Reasoning effort for the Responses/Vertex user-sims (ignored by /v1/messages
# models, whose API has no equivalent field).
REASONING_EFFORT="${OPENAI_AGENT_REASONING_EFFORT:-low}"

TEST_PARQUET="${TEST_PARQUET:-turing-rl/data/prism/prism_history_s42_sft40_grpo60/test.parquet}"

# Seeds. These models take no --seed (the Responses API rejects one, Anthropic
# /v1/messages has none), but they DO sample at the provider default temperature,
# so each run is a genuinely independent draw -- verified 3/3 distinct on an
# identical prompt for gpt-5.5, claude-opus-5 and gemini-3.1-pro. Seeds here
# therefore give real variance; they are simply not reproducible, so re-running
# "seed1" yields a new draw rather than the same one. Artifacts are suffixed
# -seed<N> so turing_eval.py aggregate groups them (it strips /-seed\d+$/).
SEEDS="${SEEDS:-0 1 2}"

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  LIMIT="${LIMIT:-20}"
  OUTDIR=${OUTDIR:-outputs/turing_smoke}
else
  LIMIT="${LIMIT:-}"
  OUTDIR=${OUTDIR:-outputs/turing}
fi

# Generation is API-bound like the judge, so keep concurrency modest: these
# providers throttle, and a burst here just buys 429s and retries.
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
