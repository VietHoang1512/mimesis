#!/bin/bash
#SBATCH --job-name=mimesis-thoughttrace
#SBATCH --nodes=1
##SBATCH --account=<your-account>
##SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -xeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MIMESIS_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${MIMESIS_CONDA_ENV:-mimesis}"

export PYTHONPATH="$MIMESIS_ROOT:${PYTHONPATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$MIMESIS_OUTPUT_DIR}"
MODEL_PATH="${MODEL_PATH:-$MIMESIS_MODEL_DIR/mimesis-9b-midtrain}"
DATASET_PATH="${DATASET_PATH:-$MIMESIS_DATA_DIR/thoughttrace/ThoughtTrace.jsonl}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$MIMESIS_ROOT/sft/qwen3_5_think_training.jinja}"
DS_CONFIG="${DS_CONFIG:-$MIMESIS_ROOT/sft/ds_zero3.json}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$MODEL_PATH")-thoughttrace-${SLURM_JOB_ID:-local}}"
RUN_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"

# ── Hyperparameters (as published) ────────────────────────────────────────────
n_gpus="${N_GPUS:-8}"
lr="${LR:-1e-7}"
num_epochs="${NUM_EPOCHS:-3}"
max_length="${MAX_LENGTH:-32768}"
micro_batch_size="${MICRO_BATCH_SIZE:-1}"
grad_accum="${GRAD_ACCUM:-4}"
warmup_steps="${WARMUP_STEPS:-10}"
save_strategy="${SAVE_STRATEGY:-steps}"
save_steps="${SAVE_STEPS:-20}"
save_total_limit="${SAVE_TOTAL_LIMIT:-10}"
save_only_model="${SAVE_ONLY_MODEL:-True}"
eval_ratio="${EVAL_RATIO:-0}"
seed="${SEED:-42}"

# ── Setup ─────────────────────────────────────────────────────────────────────
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
mkdir -p "$HF_HOME" "$WANDB_DIR" "$RUN_DIR" logs

export LOGGING_LEVEL="${LOGGING_LEVEL:-ERROR}"

export FLA_TILELANG="${FLA_TILELANG:-0}"

export TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-/tmp/triton_cache_${USER:-$(id -un)}}"
export TRITON_CACHE_DIR="$TRITON_CACHE_ROOT/$(hostname -s)"
export TORCHINDUCTOR_CACHE_DIR="$TRITON_CACHE_ROOT/inductor-$(hostname -s)"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# ── Script / SFTConfig arguments ──────────────────────────────────────────────
ARGS=(
  --dataset_path "$DATASET_PATH"
  --drop_trailing_context True
  --eval_ratio "$eval_ratio"
  --dataset_num_proc 16

  --model_name_or_path "$MODEL_PATH"
  --model_dtype bfloat16
  --attn_implementation flash_attention_2
  --gradient_checkpointing True

  --chat_template_path "$CHAT_TEMPLATE"
  --assistant_only_loss True
  --max_length "$max_length"
  --packing False

  --learning_rate "$lr"
  --lr_scheduler_type cosine
  --warmup_steps "$warmup_steps"
  --num_train_epochs "$num_epochs"
  --per_device_train_batch_size "$micro_batch_size"
  --gradient_accumulation_steps "$grad_accum"
  --max_grad_norm 1.0
  --bf16 True
  --seed "$seed"

  --output_dir "$RUN_DIR"
  --run_name "$EXPERIMENT_NAME"
  --logging_steps 1
  --save_strategy "$save_strategy"
  --save_steps "$save_steps"
  --save_total_limit "$save_total_limit"
  --save_only_model "$save_only_model"
  --report_to wandb
)

if [[ "${INSPECT_ONLY:-0}" == "1" ]]; then
  python -m sft.trl_sft "${ARGS[@]}" \
    --inspect_only True --report_to none --bf16 False --use_cpu True "$@"
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-mimesis-thoughttrace}"

NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTHONUNBUFFERED=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node="$n_gpus" \
    -m sft.trl_sft "${ARGS[@]}" --deepspeed "$DS_CONFIG" "$@"
