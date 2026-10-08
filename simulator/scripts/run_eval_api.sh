#!/bin/bash
#SBATCH --job-name=eval-api
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

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source /opt/conda/etc/profile.d/conda.sh
conda activate ${MIMESIS_CONDA_ENV:-mimesis}
cd "$MIMESIS_ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OPENAI_MODEL_NAME="gpt-5-5"
export JUDGE_MODEL_NAME=$OPENAI_MODEL_NAME
export WANDB_PROJECT="MIMESIS-eval"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export SAVE_CONVERSATIONS=1

export MAX_CONCURRENT_ROLLOUTS="${MAX_CONCURRENT_ROLLOUTS:-96}"

SEEDS="${SEEDS:-0 1 2}"

MODELS_OPENAI="gpt-5.5"
MODELS_CLAUDE="claude-opus-5 claude-opus-4.8 claude-opus-4.7 claude-opus-4.6 claude-opus-4.5 \
claude-sonnet-5 claude-sonnet-4.6 claude-sonnet-4.5 claude-haiku-4.5"
MODELS_GEMINI="gemini-3.6-flash gemini-3.5-flash gemini-3.5-flash-lite \
gemini-3.1-pro-preview gemini-3.1-flash-lite gemini-3-flash-preview \
gemini-2.5-pro gemini-2.5-flash gemini-2.5-flash-lite"

MODELS="${MODELS:-gpt-5.5 }"
[ "$MODELS" = "all" ] && MODELS="$MODELS_OPENAI $MODELS_CLAUDE $MODELS_GEMINI"

MIN_PER_RUN=${MIN_PER_RUN:-50}
N_MODELS=$(echo $MODELS | wc -w)
N_SEEDS=$(echo $SEEDS | wc -w)
N_RUNS=$((N_MODELS * N_SEEDS))
echo "[run_eval_api] $N_MODELS model(s) x $N_SEEDS seed(s) = $N_RUNS run(s)"
echo "[run_eval_api] concurrency: $MAX_CONCURRENT_ROLLOUTS in-flight rollouts"
echo "[run_eval_api] rough estimate at ~${MIN_PER_RUN} min/run: $((N_RUNS * MIN_PER_RUN / 60))h $((N_RUNS * MIN_PER_RUN % 60))m"
echo "[run_eval_api] models: $MODELS"

case "$MODELS" in
  *-genai*) export OPENAI_AGENT_TRANSPORT=judge ;;
  *)        unset  OPENAI_AGENT_TRANSPORT ;;
esac
echo "[run_eval_api] transport: ${OPENAI_AGENT_TRANSPORT:-default OpenAI-compatible client}"

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
QWEN_SNAPSHOT=$(ls -d "$PWD/outputs/hf_cache/hub/models--Qwen--Qwen3-4B-Base/snapshots/"*/ 2>/dev/null | head -1)
QWEN_SNAPSHOT="${QWEN_SNAPSHOT%/}"
TOKENIZER_MODEL="${QWEN_SNAPSHOT:-Qwen/Qwen3-4B-Base}"
echo "[run_eval_api] tokenizer model: $TOKENIZER_MODEL"

for model in $MODELS
do
  unset OPENAI_AGENT_BASE_URL
  export OPENAI_AGENT_MODEL=$model
  export OPENAI_AGENT_REASONING_EFFORT=low
  export ACTOR_MODEL_PATH=$TOKENIZER_MODEL
  for seed in $SEEDS
  do
    export EVAL_SEED=$seed
    export EXPERIMENT_NAME="${model}-eval-usersim-$OPENAI_MODEL_NAME-seed$seed"
    if [ -s "outputs/$EXPERIMENT_NAME-eval.scores.json" ]; then
      echo "[run_eval_api] skip (already scored): $EXPERIMENT_NAME"
      continue
    fi
    echo "[run_eval_api] === $model seed=$seed -> outputs/$EXPERIMENT_NAME-eval.log"
    bash recipe/ditto/eval.sh api | tee outputs/$EXPERIMENT_NAME-eval.log
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
