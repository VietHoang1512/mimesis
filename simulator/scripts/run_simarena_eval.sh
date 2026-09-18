#!/bin/bash
#SBATCH --job-name=simarena-eval
#SBATCH --nodes=1
##SBATCH --account=<your-account>
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# SimulatorArena document-creation JUDGED stage, for simulators whose
# conversations already exist under simulation/output/<annotation_id>/.
#
# Pipeline (mirrors evaluation/document_creation/evaluate_simulators.sh):
#   0. terminate_conversation_document_creation.py  -- marks where each simulated
#      conversation actually ended; every later step reads it, and the shipped
#      driver hard-exits without it.
#   1. extract document   -> evaluation_outputs/extracted_document/
#   2. rate document      -> evaluation_outputs/document_rating/
#   3. rate interaction   -> evaluation_outputs/interaction_rating/
# Step 2 REQUIRES step 1; the order below is not cosmetic.
#
# TWO SUBSTITUTIONS, both forced:
#   * the shipped runner submits to OpenAI's Batch API (api.openai.com), which is
#     VPN-gated here and hangs rather than failing. simarena_run_batch.py consumes
#     the identical batch_prompts/*.jsonl and writes the identical
#     evaluation_outputs/ structure, but fans out through api.py.
#   * the evaluator is gpt-5-mini upstream; only gpt-5.5 exists on this gateway.
#     SIMARENA_EVALUATOR remaps it (utils.py) so the hardcoded name in the
#     termination script is covered too.
#
# We deliberately STOP before show_simulator_performance.py. That script computes
# correlation against HUMAN ratings looked up BY assistant model; our assistant is
# a substitute nobody rated, so every instance resolves to None and the table comes
# back empty. Ratings are aggregated by simarena_aggregate.py instead, as a
# simulator-preference ranking with no human-correlation claim.
#
# Usage:
#   sbatch --qos=a100_genai_interns_high run_simarena_eval.sh
#   SIMS="gpt-5.5" bash run_simarena_eval.sh          # single simulator
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
ROOT="$MIMESIS_ROOT"
cd "$ROOT/SimulatorArena/simulation"

export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PYTHONUNBUFFERED=1
export SIMARENA_EVALUATOR="${SIMARENA_EVALUATOR:-gpt-5.5}"

ANNOTATION_ID="${ANNOTATION_ID:-document_creation_annotations}"
VERSION="${VERSION:-zero-shot-cot-user-profile}"
PROFILE="${PROFILE:-preference_and_writing_interaction_style}"
FILE_STEM="${VERSION}-up-${PROFILE}_for_benchmarking"
# RealUserSim is on the same gpt-5.5 deployment; 19 concurrent jobs there dropped
# throughput 2.7x. Stay modest.
WORKERS="${WORKERS:-8}"

# Every simulator that has generations, unless overridden.
if [ -z "${SIMS:-}" ]; then
  SIMS=$(cd "output/$ANNOTATION_ID" && find . -name "${FILE_STEM}.json" \
         | sed 's|^\./||;s|/[^/]*\.json$||' | sort)
fi
echo "[simeval] annotation=$ANNOTATION_ID  evaluator=$SIMARENA_EVALUATOR  workers=$WORKERS"
echo "[simeval] simulators: $(echo $SIMS | wc -w)"

# `-s` is not enough: a failed termination run still writes "{}", which is 2 bytes
# and non-empty, so a size check silently accepts it and every later stage skips.
# Require actual keys.
nonempty_json() { python -c "
import json,sys
try: sys.exit(0 if json.load(open(sys.argv[1])) else 1)
except Exception: sys.exit(1)" "$1" 2>/dev/null; }

FAILED=""
for sim in $SIMS; do
  # batch-prompt scripts identify a run by file_name; the simulator is a directory
  # level above, so it has to be folded into the name they see.
  FN="$sim/$FILE_STEM"
  echo ""
  echo "[simeval] ================= $sim ================="

  # --- step 0: termination detection -------------------------------------
  TERM="terminated_conversations/$ANNOTATION_ID/$FN.json"
  if nonempty_json "$ROOT/SimulatorArena/simulation/$TERM"; then
    echo "[simeval] skip termination (exists)"
  else
    # --simulation_path is resolved as simulation/output/<annotation_id>/<arg>,
    # so it must be the path RELATIVE to that, not from the repo root.
    python terminate_conversation_document_creation.py \
      --annotation_id "$ANNOTATION_ID" \
      --simulation_path "$FN.json" \
      2>&1 | grep -v "^\[api.py\] POST" | tail -5
  fi

  if ! nonempty_json "$ROOT/SimulatorArena/simulation/$TERM"; then
    echo "[simeval] ERROR: no terminated_conversations for $sim -- skipping"
    FAILED="$FAILED $sim/terminate"; continue
  fi

  # --- steps 1-3 ----------------------------------------------------------
  cd "$ROOT/SimulatorArena/evaluation/document_creation/scripts"
  for stage in extracted_document document_rating interaction_rating; do
    OUT="../evaluation_outputs/$ANNOTATION_ID/$stage/$FN.json"
    if nonempty_json "$OUT"; then echo "[simeval] skip $stage (exists)"; continue; fi
    # Ratings read the EXTRACTED document. generate_batch_prompts_for_rating.py
    # will happily emit 51 prompts without it and rate an empty document, so the
    # dependency has to be enforced here rather than trusted.
    if [ "$stage" != "extracted_document" ] && \
       ! nonempty_json "../evaluation_outputs/$ANNOTATION_ID/extracted_document/$FN.json"; then
      echo "[simeval] ERROR: $stage needs extracted_document first -- skipping"
      FAILED="$FAILED $sim/$stage"; continue
    fi

    case "$stage" in
      extracted_document)
        python generate_batch_prompts_for_document_extraction.py \
          --file_name "$FN" --annotation_id "$ANNOTATION_ID" \
          --evaluator_model "$SIMARENA_EVALUATOR" 2>&1 | tail -3 ;;
      *)
        python generate_batch_prompts_for_rating.py \
          --file_name "$FN" --annotation_id "$ANNOTATION_ID" \
          --aspect "${stage%_rating}" \
          --evaluator_model "$SIMARENA_EVALUATOR" 2>&1 | tail -3 ;;
    esac

    # The batch scripts always write to evaluation_outputs/<stage>/<FN>.json with no
    # dataset in the path, and <FN> is the same for every annotation set -- so results
    # must be relocated under <annotation_id>/ or the next dataset overwrites this one.
    OUTDIR_NS="../evaluation_outputs/$ANNOTATION_ID/$stage/$(dirname "$FN")"
    mkdir -p "$OUTDIR_NS"

    BATCH="../batch_prompts/$stage/$FN.jsonl"
    if [ ! -s "$BATCH" ]; then echo "[simeval] no prompts for $stage"; continue; fi
    python "$ROOT/simarena_run_batch.py" --batch-file "$BATCH" \
      --model "$SIMARENA_EVALUATOR" --workers "$WORKERS" \
      2>&1 | grep -v "^\[api.py\] POST"
    FLAT="../evaluation_outputs/$stage/$FN.json"
    [ -s "$FLAT" ] && mv "$FLAT" "$OUTDIR_NS/" && echo "[simeval]   -> $OUTDIR_NS/"
    nonempty_json "../evaluation_outputs/$ANNOTATION_ID/$stage/$FN.json" \
      || FAILED="$FAILED $sim/$stage"
  done
  cd "$ROOT/SimulatorArena/simulation"
done

[ -n "$FAILED" ] && echo "[simeval] FAILED:$FAILED"
cd "$ROOT"
python simarena_aggregate.py --annotation-id "$ANNOTATION_ID"
