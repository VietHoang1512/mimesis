#!/bin/bash
#SBATCH --job-name=eval
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

source /opt/conda/etc/profile.d/conda.sh
conda activate ${MIMESIS_CONDA_ENV:-mimesis}
cd "$MIMESIS_ROOT"
# Judges (call_openai / call_openai_parse) go through the hosted endpoint via api.py, which
# needs no API key or base URL: the outbound proxy injects auth, and WITH_X2P=1 (above) makes
# the proxy reachable on compute nodes. The old the configured endpoint passthrough vars were
# VPN-gated (the source of the 403s), so they're removed. Note the the shared API client
# pipelines this used to name are gone -- they now 403 every model -- so api.py
# points at the per-upstream gateways instead; nothing to change here.
# wandb egress is gated from compute nodes, and trainer.logger includes "wandb"
# (recipe/ditto/eval.sh:148), so an online init blocks 90s and then raises
# CommError -- which kills the run BEFORE any rollout. Job 472660 exited 0:0 with
# all 29 datasets N/A for exactly this reason, so the failure is silent unless you
# read stderr. run_eval_api.sh already defaults offline; match it.
# Nothing about the scores depends on wandb: aggregate.py parses the tee'd stdout
# log. Runs are still recorded under $WANDB_DIR and can be pushed once egress is
# back:  wandb sync outputs/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-offline}"
# Judge model. api.py's normalize_model() rewrites this to gpt-5.5; the old spelling
# is kept because EXPERIMENT_NAME embeds it and renaming would orphan existing rows.
export OPENAI_MODEL_NAME="gpt-5-5"
export JUDGE_MODEL_NAME=$OPENAI_MODEL_NAME
export WANDB_PROJECT="MIMESIS-eval"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
# Thinking + chat template (read by the Agent class, agents/utils.py).
#   TURNOFF_THINK=1 (default) disables the model's <think> reasoning; =0 keeps it.
#   USE_MODEL_CHAT_TEMPLATE=0 (default) forces the plain ChatML template the MIMESIS
#     models were SFT'd with; =1 keeps the model's OWN template (e.g. Qwen3.5's native
#     thinking template, whose generation prompt pre-fills <think>).
# To eval a THINKING model set BOTH: USE_MODEL_CHAT_TEMPLATE=1 TURNOFF_THINK=0 sbatch run_eval.sh
export TURNOFF_THINK="${TURNOFF_THINK:-0}"
export USE_MODEL_CHAT_TEMPLATE="${USE_MODEL_CHAT_TEMPLATE:-0}"

# Generation budget per turn (recipe/ditto/eval.sh reads MAX_RESPONSE_LENGTH). A thinking
# model spends most of its budget on the reasoning block, and eval.sh's 8k default cut
# ~7% of samples off mid-<think>: no </think> is emitted, so clean_response returns "" and
# those score exactly 0 — measured at 41% of humanllm and 13% of mistakes on the
# thoughttrace-337915 checkpoint, worth ~3 points of Overall. run_rl.sh already uses 16k
# for the same reason, so validation-in-wandb and this eval disagreed by that much.
# Tracks TURNOFF_THINK so no-think runs keep the old 8k and stay comparable to existing rows.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$([ "$TURNOFF_THINK" = "0" ] && echo $((1024 * 16)) || echo $((1024 * 8)))}"

# Tags baked into EXPERIMENT_NAME so runs with different thinking / chat-template
# settings get their own log, outputs dir and W&B run instead of colliding.
THINK_TAG=$([ "$TURNOFF_THINK" = "1" ] && echo "nothink" || echo "think")
TMPL_TAG=$([ "$USE_MODEL_CHAT_TEMPLATE" = "1" ] && echo "modeltmpl" || echo "chatml")

# Save the full conversation of every eval rollout (all 27 tasks) as JSON under
# outputs/$EXPERIMENT_NAME/conversations/ for later analysis. Set to 0 to disable,
# or set CONVERSATION_LOG_DIR to redirect elsewhere. See agents/utils.py.
export SAVE_CONVERSATIONS=1

# Seeds to evaluate over. Each seed is a full, independently-seeded eval run
# (EVAL_SEED -> vLLM per-request SamplingParams.seed, see agents/utils.py) with
# its own log / outputs dir / conversations dir, suffixed "-seed<seed>". Average
# across seeds to reduce sampling noise. Override for a single-seed run:
#   SEEDS=0 sbatch run_eval.sh
SEEDS="${SEEDS:-0 1 2}"

# Checkpoints to evaluate, as absolute paths. Overridable so a single missing model
# can be filled in without editing this file; the default is unchanged.
MODELS="${MODELS:-${MIMESIS_MODEL_DIR}/Qwen3-4B}"

for model in $MODELS
do
  export ACTOR_MODEL_PATH=$model
  for seed in $SEEDS
  do
    export EVAL_SEED=$seed
    export EXPERIMENT_NAME="$(basename "$ACTOR_MODEL_PATH")-eval-$OPENAI_MODEL_NAME-$THINK_TAG-$TMPL_TAG-seed$seed"
    # Resume guard, same as run_eval_api.sh. a100_genai_shared is PREEMPTIBLE and a
    # seed costs ~4h, so without this a preemption near the end throws away the whole
    # run -- job 468915 was preempted 8h in and restarted from seed 0. Keyed on the
    # scores file, written last, so a half-finished seed is still correctly redone.
    # An all-None scores file (the wandb-timeout failure mode) is NOT skipped: the
    # guard requires at least one non-null dataset.
    if [ -s "outputs/$EXPERIMENT_NAME-eval.scores.json" ] && \
       python -c "import json,sys; d=json.load(open('outputs/$EXPERIMENT_NAME-eval.scores.json')); \
sys.exit(0 if any(v is not None for v in d.values()) else 1)" 2>/dev/null; then
      echo "[run_eval] skip (already scored): $EXPERIMENT_NAME"
      continue
    fi
    echo "[run_eval] seed=$seed conversations -> outputs/$EXPERIMENT_NAME/conversations/"
    bash recipe/ditto/eval.sh local | tee outputs/$EXPERIMENT_NAME-eval.log
    python aggregate.py outputs/$EXPERIMENT_NAME-eval.log
  done
done
