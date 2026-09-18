#!/bin/bash
#SBATCH --job-name=eval-api
#SBATCH --nodes=1
##SBATCH --account=<your-account>
##SBATCH --qos=a100_dev
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Cluster-specific SLURM settings. The published runs used Meta's internal
# partitions; replace the account/qos below with your own, or launch the
# `python3 train_ppo.py` / eval invocation directly without sbatch.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source /opt/conda/etc/profile.d/conda.sh
conda activate ${MIMESIS_CONDA_ENV:-mimesis}
cd "$MIMESIS_ROOT"
# Eval remote API models AS THE USER-SIMULATOR under test (gpt-5.5, claude-opus-4.7,
# gemini-3.6-flash, ...). The tested model AND the judges both route through the hosted endpoint
# via api.py: the outbound proxy injects auth and WITH_X2P=1 (above) makes the proxy reachable
# on compute nodes, so no API key / base URL is needed. The OpenAI-SDK path (VPN-gated
# the configured endpoint / external endpoints) is avoided by leaving OPENAI_AGENT_BASE_URL unset.
#
# verl always builds a local vLLM engine even in api mode, so a small tokenizer model
# (ACTOR_MODEL_PATH, default Qwen/Qwen3-4B-Base) still boots on GPU for token
# bookkeeping only — it does not affect scores.
# Offline, not online: external egress from these nodes is currently blocked (the outbound proxy
# answers CONNECT with 403 for huggingface.co, and api.wandb.ai is unreachable), so
# wandb.init() blocks for 90s and then raises CommError inside Tracking.__init__ -- which
# happens in ray_trainer.fit() BEFORE any eval work, so the whole run dies. Job 430300 lost
# seed0 that way after an hour of vLLM startup.
#
# Nothing about the scores depends on wandb: aggregate.py parses the tee'd stdout log. Runs
# are still recorded under $WANDB_DIR (recipe/ditto/eval.sh) and can be pushed once egress
# is back:  wandb sync outputs/wandb/offline-run-*
# Set WANDB_MODE=online explicitly to restore the old behaviour.
export WANDB_MODE="${WANDB_MODE:-offline}"
# Judge model. Deliberately the OLD "gpt-5-5" spelling: api.py's normalize_model()
# rewrites it to gpt-5.5 before the request, while EXPERIMENT_NAME below embeds this
# string, so changing it here would rename every output log and W&B run and break
# comparability with the existing rows.
export OPENAI_MODEL_NAME="gpt-5-5"            # judge model (independent of tested model)
export JUDGE_MODEL_NAME=$OPENAI_MODEL_NAME
export WANDB_PROJECT="MIMESIS-eval"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export SAVE_CONVERSATIONS=1

# In-flight rollouts. recipe/ditto/eval.sh defaults to 512, which is sized for a local vLLM
# engine; an API user-simulator is metered by the provider instead. The Anthropic upstream
# rejected the 2026-08-11 runs with
#   rate limit exceeded for u-...::up-anthropic::w-60s:  517/475 in 60s
#   rate limit exceeded for u-...::up-anthropic::w-600s: 2451/2450 in 600s
# and claude-opus-5 / claude-sonnet-5 lost 27% of their rollouts to zero-turn conversations
# (they scored 46.6 / 44.5 against claude-opus-4.8's 62.4 -- an artifact, not a capability).
#
# The 600s window binds: 2450/600s = 4.08 req/s sustained. At the ~30s median call latency
# recorded in api.py, in-flight N yields N/30 req/s, so N must stay under ~120. 96 leaves
# headroom for latency swings. This slows a run from ~22 to ~50 min -- the run is now
# quota-bound rather than latency-bound, which is the actual ceiling either way.
export MAX_CONCURRENT_ROLLOUTS="${MAX_CONCURRENT_ROLLOUTS:-96}"

# Seeds to evaluate over. Each seed is a separate eval run with its own log /
# outputs / conversations dir (suffixed "-seed<seed>"); average across seeds to
# reduce noise. Neither the hosted endpoint path takes a seed (the Responses API rejects it
# outright, Anthropic /v1/messages has none), so EVAL_SEED only separates the runs.
# Override for a single-seed run: SEEDS=0 sbatch run_eval_api.sh
SEEDS="${SEEDS:-0 1 2}"

# Models to evaluate as the user simulator, spelled as the hosted endpoint registry names.
# CallAPI routes each family automatically (api.py): gpt-* -> Azure Responses,
# claude-* -> Anthropic /v1/messages, gemini-* -> Vertex :generateContent.
#
#   (default)                           the curated cross-provider set below
#   MODELS=all                          every text model on every gateway (19)
#   MODELS="gpt-5.5 gemini-2.5-pro"     an explicit list
#
# Sets below are what is actually deployed, verified 2026-08-07 by calling each one.
# Two changes from the old hardcoded list: grok-4 and glm-5.2 have no route on any
# gateway any more (gpt-5.5 is the ONLY routable OpenAI model, so gpt-5-4 / codex
# names would just fail loudly), and gemini is BACK -- it was dropped during the
# the shared API client era but the `confucius-vertex-gemini` upstream serves 9 text models.
MODELS_OPENAI="gpt-5.5"
MODELS_CLAUDE="claude-opus-5 claude-opus-4.8 claude-opus-4.7 claude-opus-4.6 claude-opus-4.5 \
claude-sonnet-5 claude-sonnet-4.6 claude-sonnet-4.5 claude-haiku-4.5"
# Text generation only. The *-image and *-computer-use deployments are not user
# simulators, and the -latest aliases are deliberately excluded: they do not report
# a resolved version, so what they point at can move under a run and quietly break
# comparability between rows.
MODELS_GEMINI="gemini-3.6-flash gemini-3.5-flash gemini-3.5-flash-lite \
gemini-3.1-pro-preview gemini-3.1-flash-lite gemini-3-flash-preview \
gemini-2.5-pro gemini-2.5-flash gemini-2.5-flash-lite"

# Default: one or two representatives per provider. Assigned with ${MODELS:-...} so
# the documented env override actually takes effect -- the previous version hardcoded
# the list, so `MODELS=... sbatch run_eval_api.sh` silently evaluated something else.
MODELS="${MODELS:-gpt-5.5 }"
[ "$MODELS" = "all" ] && MODELS="$MODELS_OPENAI $MODELS_CLAUDE $MODELS_GEMINI"

# An API eval used to cost ~22 min per seed at 512-way concurrency (measured: the glm-5.2
# usersim logs ran 23:36 -> 23:56 -> 00:20). At the quota-safe MAX_CONCURRENT_ROLLOUTS above
# the run is throughput-bound by the provider instead, so budget ~50 min/run. A full sweep
# is therefore sized in HOURS and can run past the 48h SLURM wall clock, so print the plan
# up front instead of discovering it three models in: MODELS=all x 3 seeds is 57 runs, ~48h.
MIN_PER_RUN=${MIN_PER_RUN:-50}
N_MODELS=$(echo $MODELS | wc -w)
N_SEEDS=$(echo $SEEDS | wc -w)
N_RUNS=$((N_MODELS * N_SEEDS))
echo "[run_eval_api] $N_MODELS model(s) x $N_SEEDS seed(s) = $N_RUNS run(s)"
echo "[run_eval_api] concurrency: $MAX_CONCURRENT_ROLLOUTS in-flight rollouts"
echo "[run_eval_api] rough estimate at ~${MIN_PER_RUN} min/run: $((N_RUNS * MIN_PER_RUN / 60))h $((N_RUNS * MIN_PER_RUN % 60))m"
echo "[run_eval_api] models: $MODELS"

# ── Transport ─────────────────────────────────────────────────────────────────
# A model name ending in -genai is a the judge endpoint entitlement (claude-5-1-fable-gcp-genai,
# gemini-3-1-pro-preview-genai, ...) and must go to the configured endpoint/compat, NOT the AI
# Gateway. Routing cannot be sniffed from the name downstream: is_messages_model()
# matches any claude-*, so fable would be sent to the Anthropic gateway where it is
# not deployed on the configured endpoint.
#
# This also buys quota independence. the judge endpoint is metered separately from the gateway's
# up-anthropic bucket -- measured 80/80 concurrent fable calls, zero throttling, 1.6s
# median, while a Claude eval held up-anthropic pinned at 478/475 -- so a -genai run
# can proceed in parallel with the claude-opus-5 / claude-sonnet-5 evals.
case "$MODELS" in
  *-genai*) export OPENAI_AGENT_TRANSPORT=judge ;;
  *)        unset  OPENAI_AGENT_TRANSPORT ;;
esac
echo "[run_eval_api] transport: ${OPENAI_AGENT_TRANSPORT:-the shared API client (the hosted endpoint)}"

# ── Preflight: can this node actually reach the transport we need? ────────────
# Some a100 nodes come up with a dead the outbound proxy: the forward proxy accepts the TCP
# connection but never answers, so every request dies on curl exit 28 after its full
# timeout. Job 430407 landed on a100-060-137 and logged 256 such timeouts and ZERO
# successful calls of any kind -- judge included -- burning an hour of vLLM startup per
# seed before failing. One check up front turns that into a fast failure you can
# immediately resubmit past.
#
# Goes through api.py rather than a hand-rolled curl on purpose: the request needs
# the proxy AND the provider-specific version
# headers, and a preflight that reimplements those will drift from the real path and
# report false failures (it did while this was being written).
#
# NOTE the judge (gpt-5.5, Responses route) always goes through the hosted endpoint, even on
# a the judge endpoint run, so the gateway is checked in BOTH cases -- a node with a dead proxy
# cannot score a the judge endpoint run either.
echo "[run_eval_api] preflight from $(hostname)..."
if ! timeout 200 python -c "
import asyncio, os, sys, api

async def main():
    # Judge transport: needed by every run regardless of the model under test.
    txt = await api.chat_async('Say hi.', model='gpt-5.5', max_tokens=32,
                               timeout=45, max_retries=2)
    print(f'  the hosted endpoint (judge) replied: {txt[:40]!r}')
    if not txt.strip():
        return 1
    if os.getenv('OPENAI_AGENT_TRANSPORT') == 'judge':
        m = '$(echo $MODELS | awk '{print $1}')'
        txt = await api.chat_async('Say hi.', model=m, max_tokens=32,
                                           timeout=45, max_retries=2)
        print(f'  the judge endpoint ({m}) replied: {txt[:40]!r}')
        if not txt.strip():
            return 1
    return 0

try:
    sys.exit(asyncio.run(main()))
except Exception as e:
    print(f'  {type(e).__name__}: {str(e)[:220]}', file=sys.stderr)
    sys.exit(1)
"; then
  echo "[run_eval_api] PREFLIGHT FAILED on $(hostname)." >&2
  echo "[run_eval_api] If this is a transport/entitlement error the message above says so;" >&2
  echo "[run_eval_api] otherwise it is a bad node -- resubmit to land elsewhere:" >&2
  echo "[run_eval_api]   MODELS=\"$MODELS\" SEEDS=\"$SEEDS\" sbatch --exclude=$(hostname) run_eval_api.sh" >&2
  exit 1
fi
echo "[run_eval_api] preflight OK"

FAILED=""
# Tokenizer/bookkeeping model for the local vLLM engine verl builds even in api mode.
# Prefer the cached snapshot DIRECTORY over the hub id: vLLM resolves a hub id through
# snapshot_download(), which under HF_HUB_OFFLINE refuses any snapshot missing a single
# file -- and this cache lacks .gitattributes / LICENSE / README.md, none of which any
# inference path reads. Job 430253 lost all 6 runs to exactly that IncompleteSnapshotError.
# vLLM's get_model_path() short-circuits on an existing path, so this skips the check
# entirely. Falls back to the hub id if the cache is ever absent.
QWEN_SNAPSHOT=$(ls -d "$PWD/outputs/hf_cache/hub/models--Qwen--Qwen3-4B-Base/snapshots/"*/ 2>/dev/null | head -1)
QWEN_SNAPSHOT="${QWEN_SNAPSHOT%/}"          # verl/utils/fs.py:238 asserts no trailing slash
TOKENIZER_MODEL="${QWEN_SNAPSHOT:-Qwen/Qwen3-4B-Base}"
echo "[run_eval_api] tokenizer model: $TOKENIZER_MODEL"

for model in $MODELS
do
  unset OPENAI_AGENT_BASE_URL                 # force the hosted endpoint transport in CallAPI
  export OPENAI_AGENT_MODEL=$model            # the model under test
  export OPENAI_AGENT_REASONING_EFFORT=low
  export ACTOR_MODEL_PATH=$TOKENIZER_MODEL    # local tokenizer model only
  for seed in $SEEDS
  do
    export EVAL_SEED=$seed
    export EXPERIMENT_NAME="${model}-eval-usersim-$OPENAI_MODEL_NAME-seed$seed"
    # Resume guard. a100_genai_shared is PREEMPTIBLE, and without this a preemption
    # restart re-runs every seed from scratch -- job 468915 was preempted 8h in and
    # began redoing seed 0, which had finished 11h earlier. At ~4h/seed that is a
    # whole day of recompute to arrive back where it started. Keyed on the scores
    # file, which is written last, so a half-finished seed is correctly redone.
    if [ -s "outputs/$EXPERIMENT_NAME-eval.scores.json" ]; then
      echo "[run_eval_api] skip (already scored): $EXPERIMENT_NAME"
      continue
    fi
    echo "[run_eval_api] === $model seed=$seed -> outputs/$EXPERIMENT_NAME-eval.log"
    bash recipe/ditto/eval.sh api | tee outputs/$EXPERIMENT_NAME-eval.log
    # ${PIPESTATUS[0]} is eval.sh's status, not tee's. Without this a model that
    # dies (a retired name, a gateway outage) would abort nothing and silently
    # produce an empty row; worse, in a 27-run sweep one failure must not cost the
    # other 26, so record it and carry on.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[run_eval_api] FAILED: $model seed=$seed"
      FAILED="$FAILED $model:seed$seed"
      continue
    fi
    python aggregate.py outputs/$EXPERIMENT_NAME-eval.log
  done
done

if [ -n "$FAILED" ]; then
  echo "[run_eval_api] completed with failures:$FAILED"
else
  echo "[run_eval_api] all $N_RUNS run(s) completed"
fi
