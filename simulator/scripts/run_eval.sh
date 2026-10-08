#!/bin/bash
#SBATCH --job-name=eval
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

source /opt/conda/etc/profile.d/conda.sh
conda activate ${MIMESIS_CONDA_ENV:-mimesis}
cd "$MIMESIS_ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OPENAI_MODEL_NAME="gpt-5-5"
export JUDGE_MODEL_NAME=$OPENAI_MODEL_NAME
export WANDB_PROJECT="MIMESIS-eval"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
export TURNOFF_THINK="${TURNOFF_THINK:-0}"
export USE_MODEL_CHAT_TEMPLATE="${USE_MODEL_CHAT_TEMPLATE:-0}"

export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$([ "$TURNOFF_THINK" = "0" ] && echo $((1024 * 16)) || echo $((1024 * 8)))}"

THINK_TAG=$([ "$TURNOFF_THINK" = "1" ] && echo "nothink" || echo "think")
TMPL_TAG=$([ "$USE_MODEL_CHAT_TEMPLATE" = "1" ] && echo "modeltmpl" || echo "chatml")

export SAVE_CONVERSATIONS=1

SEEDS="${SEEDS:-0 1 2}"

MODELS="${MODELS:-${MIMESIS_MODEL_DIR}/Qwen3-4B}"

for model in $MODELS
do
  export ACTOR_MODEL_PATH=$model
  for seed in $SEEDS
  do
    export EVAL_SEED=$seed
    export EXPERIMENT_NAME="$(basename "$ACTOR_MODEL_PATH")-eval-$OPENAI_MODEL_NAME-$THINK_TAG-$TMPL_TAG-seed$seed"
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
