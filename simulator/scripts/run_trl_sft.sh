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

# Stage 2 of 3 — ThoughtTrace SFT. Finetunes the mid-training checkpoint from
# run_sft.sh on 2,155 conversations a human annotated turn by turn, via TRL's
# SFTTrainer on a single node of 8 A100s (DeepSpeed ZeRO-3).
#
# This is the stage that introduces reasoning: conversations are rendered under
# the native Qwen3.5 thinking template and each human turn is preceded by a
# <think> block built from the annotator's stated `reasons`. The role swap and
# that construction live in sft/trl_sft.py.
#
# Expected data layout:
#   huggingface-cli download cmu-lti/mimesis-thoughttrace \
#     --repo-type dataset --local-dir "$MIMESIS_DATA_DIR/thoughttrace"
#
# Replace the account/qos above with your own, or drop sbatch and launch
# directly on an interactive allocation.
#
# Usage:
#   sbatch scripts/run_trl_sft.sh                      # 1 node x 8 GPU, as published
#   bash scripts/run_trl_sft.sh                        # same, on an interactive alloc
#   INSPECT_ONLY=1 bash scripts/run_trl_sft.sh         # render + verify loss mask, no GPU
#   NUM_EPOCHS=1 LR=5e-6 sbatch scripts/run_trl_sft.sh
#   SAVE_STEPS=50 SAVE_TOTAL_LIMIT=5 sbatch scripts/run_trl_sft.sh
set -xeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MIMESIS_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${MIMESIS_CONDA_ENV:-mimesis}"

export PYTHONPATH="$MIMESIS_ROOT:${PYTHONPATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$MIMESIS_OUTPUT_DIR}"
# Output of stage 1, merged to HuggingFace format (see README.md).
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
# Rendered conversations measure p50 2.0k / p99 10.7k / max 22.7k tokens, so 32k
# truncates nothing. max_length only caps; it does not reserve memory.
max_length="${MAX_LENGTH:-32768}"
# 1 sequence per GPU x 8 GPUs x 4 accum = effective batch of 32 conversations,
# i.e. ~68 optimizer steps per epoch over the 2,155 conversations.
micro_batch_size="${MICRO_BATCH_SIZE:-1}"
grad_accum="${GRAD_ACCUM:-4}"
warmup_steps="${WARMUP_STEPS:-10}"
# Checkpoint every 20 optimizer steps, keeping the last 10 — ~3 checkpoints per
# epoch, ~10 over the default 3 epochs.
save_strategy="${SAVE_STRATEGY:-steps}"
save_steps="${SAVE_STEPS:-20}"
save_total_limit="${SAVE_TOTAL_LIMIT:-10}"
# Weights only, no optimizer/scheduler state — mirrors run_sft.sh's
# `checkpoint.save_contents=[model,extra]`. Keeps each checkpoint ~8GB instead of
# ~55GB (ZeRO-3 shards fp32 Adam state across ranks and writes all of it).
# Trade-off: you can resume the model but not the optimizer. Set to False if you
# need bit-exact resumption.
save_only_model="${SAVE_ONLY_MODEL:-True}"
eval_ratio="${EVAL_RATIO:-0}"
seed="${SEED:-42}"

# ── Setup ─────────────────────────────────────────────────────────────────────
# HF_HOME is exported by env.sh; keep wandb output beside this run.
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
mkdir -p "$HF_HOME" "$WANDB_DIR" "$RUN_DIR" logs

export LOGGING_LEVEL="${LOGGING_LEVEL:-ERROR}"

# Qwen3.5 runs most of its layers on fla's linear attention. Prefer fla's triton
# kernels over the slower tilelang reference path.
export FLA_TILELANG="${FLA_TILELANG:-0}"

# Keep triton's autotuning cache node-local; concurrent ranks race on a shared one.
export TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-/tmp/triton_cache_${USER:-$(id -un)}}"
export TRITON_CACHE_DIR="$TRITON_CACHE_ROOT/$(hostname -s)"
export TORCHINDUCTOR_CACHE_DIR="$TRITON_CACHE_ROOT/inductor-$(hostname -s)"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# ── Script / SFTConfig arguments ──────────────────────────────────────────────
# NB: --include_reactions is deliberately absent. It defaults to False, so the
# <think> blocks are built from the annotator's `reasons` only. That is what the
# published run used; passing it would train on different data.
ARGS=(
  # Data — the role swap and <think> construction live in sft/trl_sft.py.
  --dataset_path "$DATASET_PATH"
  --drop_trailing_context True
  --eval_ratio "$eval_ratio"
  --dataset_num_proc 16

  # Model
  --model_name_or_path "$MODEL_PATH"
  --model_dtype bfloat16
  --attn_implementation flash_attention_2
  --gradient_checkpointing True

  # Loss is taken only over the human's turns, which the {% generation %} block
  # in the training template marks. Verify with INSPECT_ONLY=1 before a long run.
  # Works despite Qwen3.5 being a VLM checkpoint only because trl_sft.py hands
  # SFTTrainer a plain tokenizer instead of the Qwen3VLProcessor.
  --chat_template_path "$CHAT_TEMPLATE"
  --assistant_only_loss True
  --max_length "$max_length"
  --packing False

  # Optimisation
  --learning_rate "$lr"
  --lr_scheduler_type cosine
  --warmup_steps "$warmup_steps"
  --num_train_epochs "$num_epochs"
  --per_device_train_batch_size "$micro_batch_size"
  --gradient_accumulation_steps "$grad_accum"
  --max_grad_norm 1.0
  --bf16 True
  --seed "$seed"

  # Trainer / logging / checkpointing
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
  # Single process, no GPU, no DeepSpeed: render an example, check the loss
  # mask is non-empty and covers the human's turns, print token lengths.
  # bf16 is off because TrainingArguments refuses to validate it without a GPU.
  python -m sft.trl_sft "${ARGS[@]}" \
    --inspect_only True --report_to none --bf16 False --use_cpu True "$@"
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-mimesis-thoughttrace}"

# Trailing "$@" lets any SFTConfig field be overridden ad hoc, e.g.
#   bash scripts/run_trl_sft.sh --max_steps 5 --report_to none
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTHONUNBUFFERED=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node="$n_gpus" \
    -m sft.trl_sft "${ARGS[@]}" --deepspeed "$DS_CONFIG" "$@"
