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
#
# RealUserSim PT3 with FRONTIER models as the user simulator. No GPU: the
# simulator, the assistant and the judge all go through the hosted endpoint transport
# in api.py, which routes per family (gpt-* Azure Responses, claude-*
# /v1/messages, gemini-* Vertex).
#
# PROTOCOL DEVIATION -- read before using the numbers.
# RealUserSim runs gpt-4o as the assistant AND the judge. Neither is reachable
# here: the Responses gateway fronts only the gpt-5-5 Azure deployment, and
# the judge endpoint answers gpt-4o-genai with "key does not have access to the model"
# (both verified). Assistant and judge are therefore pinned to gpt-5.5. Every
# simulator sees the identical assistant and identical judge, so comparisons
# BETWEEN the rows are clean -- but the absolute Fidelity Index is not on the
# same scale as the published 45.3 / 24.2, which stay an external reference.
#
# Prereq (already done; compute nodes reach HF, the login node does not):
#   python -c "from huggingface_hub import snapshot_download as d; \
#     d('Salesforce/RealUserSim', repo_type='dataset', local_dir='data/RealUserSim')"
#
# Usage:
#   sbatch --qos=cpu_lowest run_realusersim_eval_api.sh
#   MODELS="claude-opus-5" CONDITIONS="correct_profile task_only" \
#     sbatch --qos=cpu_lowest run_realusersim_eval_api.sh
#   SMOKE=1 bash run_realusersim_eval_api.sh          # 10 cases, mixed_domain
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "$MIMESIS_ROOT"
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
# --export=ALL forwards the SUBMITTING shell's https_proxy, which points at that
# host's own outbound proxy listener and is dead on any compute node. api.py is immune (it
# passes --proxy explicitly to curl) but the OpenAI SDK path is not. Same line as
# run_turing_eval_api.sh, added there after frontier runs sat at 0/10 for hours.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
# Progress lines are the only way to see a stalled run; do not let them sit in a buffer.
export PYTHONUNBUFFERED=1

MODELS="${MODELS:-gpt-5.5 claude-opus-5 claude-sonnet-5 gemini-3.6-flash gemini-3.1-pro-preview}"

# Round-robin sharding, matching run_realusersim_eval_local.sh. This was MISSING
# here: SHARD/NSHARD were accepted by the caller's --export and then silently
# ignored, so two "shards" each ran the WHOLE roster and appended to the same
# *_gen.jsonl concurrently -- jobs 471630/471631 burned 21h doing that and left 27
# duplicate case_ids in one file. An unread variable is indistinguishable from a
# working one at submit time, which is why this has to live in the script.
SHARD="${SHARD:-}"; NSHARD="${NSHARD:-}"
if [ -n "$SHARD" ] && [ -n "$NSHARD" ]; then
  MODELS=$(echo $MODELS | tr ' ' '\n' | awk -v s="$SHARD" -v n="$NSHARD" 'NR%n==s')
  echo "[rus_api] shard $SHARD/$NSHARD -> $(echo $MODELS | wc -w) model(s)"
fi
# correct_profile is the headline condition; task_only is the published 24.2
# control that isolates the profile's contribution. shuffled_profile answers the
# harder question -- whether a gain comes from matching THIS user or merely from
# being told to imitate someone -- and is opt-in because it triples the bill.
CONDITIONS="${CONDITIONS:-correct_profile task_only}"
SEEDS="${SEEDS:-0}"
ASSISTANT="${RUS_ASSISTANT:-gpt-5.5}"
JUDGE="${RUS_JUDGE:-gpt-5.5}"
export RUS_ASSISTANT RUS_JUDGE

# All three roles share the provider quota (claude-* in particular sit in the
# up-anthropic bucket, 475/60s and 2450/600s, that throttled the SOUL evals), and
# one PT3 case is ~11 sequential calls. Keep this modest.
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

      # simulate and judge are both resumable (they skip case_ids already on
      # disk), so re-running after a preemption is always safe and never
      # duplicates work.
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
