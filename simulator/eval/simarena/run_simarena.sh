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
#
# SimulatorArena document-creation: evaluate the models in our paper table AS USER
# SIMULATORS, with the ASSISTANT held fixed at gpt-5.5.
#
# PROTOCOL DEVIATION -- read before using the numbers.
# SimulatorArena's own simulator metric correlates LLM-judge ratings of simulated
# conversations against HUMAN ratings of conversations with 9 specific assistants
# (gpt-4o, gpt-4-turbo, claude-3-5-sonnet-20240620, llama-3-1-70b/8b,
# mistral-large-2407, phi-3-medium/small, gpt-4o-mini). NONE of those are reachable
# here: the hosted endpoint returns DeploymentNotFound for all nine, and the judge endpoint has only
# gpt-4o / gpt-4o-mini and the key is not entitled to them. Substituting the
# assistant makes correlation-against-human-ratings meaningless, because those
# ratings are ratings OF those systems.
#
# So this does NOT reproduce their metric. It holds the assistant fixed at gpt-5.5
# and compares simulators to each other on the judge's document/interaction
# ratings. Internally consistent; not comparable to published SimulatorArena
# numbers, and it carries no human-correlation validity claim.
#
# WHICH SUBSET -- this is a protocol question, not a cost knob.
# The paper has TWO different evaluations on two different sets:
#   sections 4-5, evaluating SIMULATORS  -> n=459 (the results table header says so)
#   section 6,    evaluating ASSISTANTS  -> 51 document topics (17 x 3 types)
# We are doing the FORMER, so 459 is the aligned choice. The shipped
# `*_for_benchmarking.json` holds the 51 and is meant for the latter; using it
# here was a mistake (it answers "rank assistants on fixed contexts", not "rank
# simulators"). Benchmarking mode -- the only path that lets us pin a substitute
# assistant, since non-benchmarking mode replays the assistant recorded in each
# annotation and all nine of those are unreachable -- hard-codes the filename to
# "<annotation_id>_for_benchmarking.json", so the full set is exposed by copying
# it to that name rather than by changing code.
#   51-file : 51 records, 51 distinct workers (one conversation each)
#   459-file: 459 records, 75 distinct workers (~6 conversations each)
# So 459 buys 9x the instances but only 1.5x the users; treat the worker as the
# clustering unit when reasoning about its error bars, not the record.
ANNOTATION_ID="${ANNOTATION_ID:-document_creation_annotations_full}"

# Usage:
#   sbatch --qos=cpu_lowest run_simarena.sh                     # gateway sims, no GPU
#   SIMS="local/osim-8b" sbatch --qos=a100_genai_interns_high run_simarena.sh
#   ANNOTATION_ID=document_creation_annotations bash run_simarena.sh   # old 51-topic set
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-${MIMESIS_CONDA_ENV:-mimesis}2}"
cd "${SIMULATORARENA_ROOT:?clone microsoft/SimulatorArena and set SIMULATORARENA_ROOT -- see eval/simarena/SETUP.md}/simulation"

export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
# Same --export=ALL hazard as run_turing_eval*.sh: the submitting shell's
# https_proxy points at ITS OWN outbound proxy listener, dead on a compute node. api.py is
# immune (explicit --proxy to curl) but anything using httpx/urllib is not.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# Thinking mode is NOT forced. Each checkpoint's own chat template decides, so a
# reasoning model may reason and a non-reasoning one need not -- every model gets to
# produce its best response rather than one we constrained. mimesis_transport's
# _strip_reasoning() removes <think> blocks and untagged prose CoT before the turn
# reaches the assistant or the judge, and warns when a turn was reasoning-only.
# Set ODYSSIM_ENABLE_THINKING=0/1 explicitly to override for an ablation.

ASSISTANT="${ASSISTANT:-gpt-5.5}"
VERSION="${VERSION:-zero-shot-cot-user-profile}"
PROFILE="${PROFILE:-preference_and_writing_interaction_style}"
NUM_CONV="${NUM_CONV:-}"          # empty = all 51 in the benchmarking subset

# The paper table, as simulators. Gateway names go through api.py; "local/<name>"
# is served by vllm on LOCAL_VLLM_PORT (see the serve block below).
SIMS_GATEWAY="gemini-3.1-pro-preview claude-opus-5 gpt-5.5 gemini-3.6-flash claude-sonnet-5"
MODEL_ROOT=${MIMESIS_MODEL_DIR}
# All ten MIMESIS-trained checkpoints, not just the two in the paper table -- the
# turing / tau-USI-D1-D4 tables carry the full set and the thoughttrace vs ditto-rl
# lines rank differently on different metrics, so dropping eight would hide that.
SIMS_OURS="\
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
SIMS_BASELINE="osim-8b osim-4b Ditto-8B humanlm-opinion sotopia-rl-qwen-2.5-7B-grpo \
Qwen3.5-9B Qwen3-8B Qwen3.5-4B"
# The serving branch dispatches on a "local/" prefix (`if [[ "$sim" == local/* ]]`),
# so these names MUST carry it. Without the prefix they are silently treated as
# gateway model names and every call fails -- which is why earlier runs had to pass
# the fully-prefixed list by hand.
SIMS_LOCAL=$(for m in $SIMS_OURS $SIMS_BASELINE; do printf 'local/%s ' "$m"; done)

# SIMS_SET selects a roster without the caller having to spell one out:
#   gateway (default) | local | all
case "${SIMS_SET:-}" in
  local)   SIMS="${SIMS:-$SIMS_LOCAL}" ;;
  all)     SIMS="${SIMS:-$SIMS_GATEWAY $SIMS_LOCAL}" ;;
esac

SIMS="${SIMS:-$SIMS_GATEWAY}"
# Round-robin sharding so N jobs can split the roster. 459 topics x 23 simulators
# is ~10.5k conversations and ~170k model calls, far too much for one job -- but
# every role funnels through the same gpt-5.5 deployment, and 19 concurrent jobs
# previously cut throughput 2.7x. Keep the fan-out to ~8.
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

wait_server_ready() {  # pid url tries -- bails the moment $pid dies
  local pid="$1" url="$2" tries="${3:-300}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 2
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# Several MIMESIS SFT checkpoints ship no chat_template; vLLM serves them fine but
# every chat request then fails, silently producing empty turns. Same guard as
# run_turing_eval.sh.
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
