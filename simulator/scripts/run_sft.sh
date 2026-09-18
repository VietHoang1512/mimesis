#!/bin/bash
#SBATCH --job-name=mimesis-midtrain
#SBATCH --nodes=4
##SBATCH --account=<your-account>
##SBATCH --qos=<your-qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=60:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Stage 1 of 3 — mid-training. Adapts the Qwen3.5-9B-ChatML base on the
# 62-corpus human-conversation mixture, producing the checkpoint that
# run_trl_sft.sh (stage 2) and run_rl.sh (stage 3) build on.
#
# Runs verl's SPMD trainer (`verl.trainer.sft_trainer`) under torchrun, routing
# data through sft/dataset.py::OdysSimSFTDataset. Every conversation is
# role-swapped so the human side occupies the generated position, and the loss
# mask covers only those turns — see sft/sft_data.py.
#
# Expected data layout:
#   huggingface-cli download cmu-lti/osim-mid-training \
#     --repo-type dataset --local-dir "$MIMESIS_DATA_DIR/osim_mid_training"
#
# Replace the account/qos above with your own, or drop sbatch and launch
# directly on an interactive allocation.
#
# Usage:
#   sbatch scripts/run_sft.sh                        # 4 nodes x 8 GPU, as published
#   bash scripts/run_sft.sh                          # single-node dev run
#   TOTAL_TRAINING_STEPS=2 bash scripts/run_sft.sh   # smoke test
#   MODEL_PATH=/path/to/base bash scripts/run_sft.sh
#   TRAIN_FILES="$MIMESIS_DATA_DIR/osim_mid_training/wildchat/train_shard_*.parquet:1.5" \
#     bash scripts/run_sft.sh                        # custom shards, with oversampling ratio
set -xeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MIMESIS_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${MIMESIS_CONDA_ENV:-mimesis}"

# verl is vendored as a package inside the repo root, so the root alone makes
# both `verl` and `sft` importable.
export PYTHONPATH="$MIMESIS_ROOT:${PYTHONPATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$MIMESIS_OUTPUT_DIR}"
MODEL_PATH="${MODEL_PATH:-$MIMESIS_MODEL_DIR/Qwen3.5-9B-CHATML}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimesis-9b-midtrain-${SLURM_JOB_ID:-local}}"

data_dir="${DATA_DIR:-$MIMESIS_DATA_DIR/osim_mid_training}"

discover_split_files() {
  local explicit="$1"
  local split="$2"
  local fallback="$3"

  if [[ -n "$explicit" ]]; then
    printf "%s" "$explicit"
    return
  fi

  local patterns=(
    "$data_dir/${split}_shard_*.parquet"
    "$data_dir/${split}-*.parquet"
    "$data_dir/${split}/*.parquet"
    "$data_dir/*${split}*.parquet"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    if compgen -G "$pattern" > /dev/null; then
      printf "%s" "$pattern"
      return
    fi
  done

  printf "%s" "$fallback"
}

check_files_arg() {
  local files_arg="$1"
  local name="$2"
  local token path

  for token in $files_arg; do
    path="$token"
    if [[ "$token" =~ :[0-9]+([.][0-9]+)?$ ]]; then
      path="${token%:*}"
    fi
    if compgen -G "$path" > /dev/null || [[ -f "$path" ]]; then
      continue
    fi
    echo "$name does not match any parquet files: $path" >&2
    echo "Set $name explicitly, or download the dataset into DATA_DIR=$data_dir." >&2
    exit 1
  done
}

train_files="$(discover_split_files "${TRAIN_FILES:-}" train "$data_dir/*/train_shard_*.parquet")"
check_files_arg "$train_files" "TRAIN_FILES"

# ── Hyperparameters (as published) ────────────────────────────────────────────
actor_lr="${ACTOR_LR:-3e-5}"
actor_lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS:-0}"
max_prompt_length="${MAX_PROMPT_LENGTH:-$((1024 * 16))}"
max_response_length="${MAX_RESPONSE_LENGTH:-$((1024 * 8))}"
max_length=$((max_prompt_length + max_response_length))
max_token_len_per_gpu=$((max_length * 2))

train_batch_size="${TRAIN_BATCH_SIZE:-4096}"
micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-8}"
usp_size="${USP_SIZE:-1}"
n_gpus="${N_GPUS:-8}"
n_nodes="${SLURM_NNODES:-1}"
total_training_steps="${TOTAL_TRAINING_STEPS:-5000}"
test_freq="${TEST_FREQ:--1}"
save_freq="${SAVE_FREQ:-200}"
# DataLoader workers per GPU. Keep low — each worker holds an independent copy
# of the dataset's parquet handles + row-group cache; too many will OOM.
num_workers="${NUM_WORKERS:-2}"
# Row groups kept resident per shard, LRU. 1 is enough for shuffled sequential
# access (each shard's rows are visited in random order but bounded by RG size).
row_group_cache_size="${ROW_GROUP_CACHE_SIZE:-1}"

# ── Setup ─────────────────────────────────────────────────────────────────────
# HF_HOME and WANDB_DIR are exported by env.sh.
mkdir -p "$HF_HOME" "$WANDB_DIR" "$OUTPUT_DIR/$EXPERIMENT_NAME" logs

# Consumed by sft/dataset.py. Mid-training supervises surface behaviour only;
# the thinking template is introduced in stage 2 (run_trl_sft.sh).
export TURNOFF_THINK="${TURNOFF_THINK:-1}"
export LOGGING_LEVEL="${LOGGING_LEVEL:-ERROR}"

# Qwen3.5 runs most of its layers on fla's linear attention. Prefer fla's triton
# kernels over the slower tilelang reference path.
export FLA_TILELANG="${FLA_TILELANG:-0}"

# Keep triton's autotuning cache node-local. The default (~/.triton/cache) often
# sits on a shared filesystem, where concurrent ranks race on the same cache file.
# TRITON_CACHE_ROOT is exported so srun-launched child shells re-evaluate
# `hostname -s` on their own node.
export TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-/tmp/triton_cache_${USER:-$(id -un)}}"

# ── Hydra overrides ───────────────────────────────────────────────────────────
HYDRA_ARGS=(
  # Data
  "data.train_files=$train_files"
  "data.val_files=null"
  "data.train_batch_size=$train_batch_size"
  "data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu"
  "data.max_token_len_per_gpu=$max_token_len_per_gpu"
  "data.max_length=$max_length"
  "data.use_dynamic_bsz=False"
  "data.pad_mode=no_padding"
  "data.truncation=error"
  "data.num_workers=$num_workers"
  # Route through our custom dataset (loads sft/dataset.py::OdysSimSFTDataset).
  "data.custom_cls.path=$MIMESIS_ROOT/sft/dataset.py"
  "data.custom_cls.name=OdysSimSFTDataset"
  # Extra knobs consumed only by OdysSimSFTDataset — need "+" because they are
  # not declared in sft_trainer_engine.yaml's data schema.
  "+data.row_group_cache_size=$row_group_cache_size"
  "+data.max_prompt_length=$max_prompt_length"
  "+data.max_response_length=$max_response_length"

  # Model
  "model.path=$MODEL_PATH"
  "model.use_remove_padding=True"
  "model.enable_gradient_checkpointing=True"
  # Fused kernels and torch.compile re-trace on every distinct sequence length,
  # which is costly here. flash-attn still handles the heavy attention path.
  "model.use_fused_kernels=False"

  # Engine (FSDP)
  "engine=fsdp"
  "engine.strategy=fsdp"
  "engine.model_dtype=bfloat16"
  "engine.ulysses_sequence_parallel_size=$usp_size"
  "engine.use_torch_compile=False"

  # Optimizer
  "optim.lr=$actor_lr"
  "optim.lr_scheduler_type=cosine"
  "optim.lr_warmup_steps=$actor_lr_warmup_steps"
  # Pin the scheduler's horizon to trainer.total_training_steps so cosine decays
  # to min-lr over that many iterations, not over len(dataloader).
  "optim.total_training_steps=$total_training_steps"

  # Trainer / logging / checkpointing
  "trainer.n_gpus_per_node=$n_gpus"
  "trainer.nnodes=$n_nodes"
  "trainer.total_epochs=1"
  "trainer.total_training_steps=$total_training_steps"
  "trainer.save_freq=$save_freq"
  "trainer.test_freq=$test_freq"
  "trainer.max_ckpt_to_keep=10"
  # Skip optimizer state to keep checkpoints ~half the size. Resume still
  # restores model+extra (scheduler, RNG), just not the optimizer.
  "checkpoint.save_contents=[model,extra]"
  "trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME"
  "trainer.project_name=mimesis"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.logger=[console,wandb]"
)

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && [[ "$n_nodes" -gt 1 ]]; then
  # Multi-node: torchrun rendezvous on the first node.
  nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
  head_node="${nodes_array[0]}"
  head_ip=$(srun --overlap --nodes=1 --ntasks=1 --mem=0 -w "$head_node" hostname --ip-address)
  if [[ "$head_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_ip"
    head_ip=${ADDR[0]}
  fi
  rdzv_port="${RDZV_PORT:-29500}"
  rdzv_id="${SLURM_JOB_ID:-$RANDOM}"

  NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTHONUNBUFFERED=1 \
    srun --overlap --mem=0 \
      bash -c '
        export TRITON_CACHE_DIR="$TRITON_CACHE_ROOT/$(hostname -s)"
        export TORCHINDUCTOR_CACHE_DIR="$TRITON_CACHE_ROOT/inductor-$(hostname -s)"
        mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        exec torchrun \
          --nnodes="'"$n_nodes"'" \
          --nproc_per_node="'"$n_gpus"'" \
          --rdzv_id="'"$rdzv_id"'" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="'"$head_ip:$rdzv_port"'" \
          -m verl.trainer.sft_trainer '"${HYDRA_ARGS[*]@Q}"'
      '
else
  export TRITON_CACHE_DIR="$TRITON_CACHE_ROOT/$(hostname -s)"
  export TORCHINDUCTOR_CACHE_DIR="$TRITON_CACHE_ROOT/inductor-$(hostname -s)"
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTHONUNBUFFERED=1 \
    torchrun --standalone --nnodes=1 --nproc_per_node="$n_gpus" \
      -m verl.trainer.sft_trainer "${HYDRA_ARGS[@]}"
fi
