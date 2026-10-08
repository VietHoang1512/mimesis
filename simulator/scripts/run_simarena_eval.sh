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
WORKERS="${WORKERS:-8}"

if [ -z "${SIMS:-}" ]; then
  SIMS=$(cd "output/$ANNOTATION_ID" && find . -name "${FILE_STEM}.json" \
         | sed 's|^\./||;s|/[^/]*\.json$||' | sort)
fi
echo "[simeval] annotation=$ANNOTATION_ID  evaluator=$SIMARENA_EVALUATOR  workers=$WORKERS"
echo "[simeval] simulators: $(echo $SIMS | wc -w)"

nonempty_json() { python -c "
import json,sys
try: sys.exit(0 if json.load(open(sys.argv[1])) else 1)
except Exception: sys.exit(1)" "$1" 2>/dev/null; }

FAILED=""
for sim in $SIMS; do
  FN="$sim/$FILE_STEM"
  echo ""
  echo "[simeval] ================= $sim ================="

  # --- step 0: termination detection -------------------------------------
  TERM="terminated_conversations/$ANNOTATION_ID/$FN.json"
  if nonempty_json "$ROOT/SimulatorArena/simulation/$TERM"; then
    echo "[simeval] skip termination (exists)"
  else
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
