#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --nodes=2
##SBATCH --account=<your-account>
##SBATCH --qos=<your-qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=80:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
cd "$MIMESIS_ROOT"
source /opt/conda/etc/profile.d/conda.sh
conda activate ${MIMESIS_CONDA_ENV:-mimesis}

set -e

# ── Task ──────────────────────────────────────────────────────────────────────
TASK=all

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR=${MIMESIS_OUTPUT_DIR}
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimesis-9b-rl-${TASK}-${SLURM_JOB_ID}}"
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"
export API_DUMP_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/api_failures"


osim_dir=${MIMESIS_DATA_DIR}/sim_rl_data/

actor_model_path=${MIMESIS_OUTPUT_DIR}/thoughttrace-sft-9b
resolve_task_files() {
  case "$1" in
    sotopia)            train_rel=sotopia_clean_rl.parquet;        val_rel=sotopia_hard_val.parquet ;;
    coser)              train_rel=coser_rl_train.parquet;          val_rel=coser_val.parquet ;;
    lifechoices)        train_rel=lifechoices_hard_rl.parquet;     val_rel=lifechoices_val.parquet ;;
    userllm)            train_rel=userllm_rl_train.parquet;        val_rel=userllm_val.parquet ;;
    mirrorbench)        train_rel=mirrorbench_rl_train.parquet;    val_rel=mirrorbench_val.parquet ;;
    fantom)             train_rel=fantom_rl_train.parquet;         val_rel=fantom_val.parquet ;;
    hitom)              train_rel=hitom_rl_train.parquet;          val_rel=hitom_val.parquet ;;
    paratomi)           train_rel=paratomi_rl_train.parquet;       val_rel=paratomi_val.parquet ;;
    mistakes)           train_rel=mistakes_rl_train.parquet;       val_rel=mistakes_val.parquet ;;
    twinvoice)          train_rel=twinvoice_rl_train.parquet;      val_rel=twinvoice_val.parquet ;;
    social_r1)          train_rel=social_r1_rl.parquet;            val_rel=social_r1_val.parquet ;;
    behaviorchain)      train_rel=behaviorchain_rl_train.parquet;  val_rel=behaviorchain_val.parquet ;;
    sim_math)           train_rel=sim_math_rl.parquet;             val_rel=sim_math_val.parquet ;;
    sim_doc)            train_rel=sim_doc_rl.parquet;              val_rel=sim_doc_val.parquet ;;
    humanual_book)      train_rel=humanual_rl_book.parquet;        val_rel=humanual_book_val.parquet ;;
    humanual_chat)      train_rel=humanual_rl_chat.parquet;        val_rel=humanual_chat_val.parquet ;;
    humanual_email)     train_rel=humanual_rl_email.parquet;       val_rel=humanual_email_val.parquet ;;
    humanual_news)      train_rel=humanual_rl_news.parquet;        val_rel=humanual_news_val.parquet ;;
    humanual_opinion)   train_rel=humanual_rl_opinion.parquet;     val_rel=humanual_opinion_val.parquet ;;
    humanual_politics)  train_rel=humanual_rl_politics.parquet;    val_rel=humanual_politics_val.parquet ;;
    alignx)             train_rel=alignx_rl_8k.parquet;            val_rel=alignx_demo_val.parquet ;;
    socsci210)          train_rel=socsci210_rl_2k.parquet;         val_rel=socsci210_val.parquet ;;
    humanllm)           train_rel=humanllm_rl_train.parquet;       val_rel=humanllm_val.parquet ;;
    usersim_behavior)   train_rel=usersim_behavior_rl_train.parquet; val_rel=usersim_behavior_val.parquet ;;
    *) echo "Unknown TASK: $1" >&2; exit 1 ;;
  esac
}

ALL_TASKS="
sotopia coser lifechoices userllm mirrorbench fantom hitom paratomi
mistakes twinvoice social_r1 behaviorchain sim_math sim_doc humanual_book
humanual_chat humanual_email humanual_news humanual_opinion humanual_politics
alignx socsci210 humanllm usersim_behavior
"

join_hydra_list() { local a=(); for f in "$@"; do a+=("'$f'"); done; local IFS=,; echo "[${a[*]}]"; }

if [ "$TASK" = "all" ]; then
  train_arr=(); val_arr=()
  for t in $ALL_TASKS; do
    resolve_task_files "$t"
    train_arr+=("$osim_dir/$train_rel")
    val_arr+=("$osim_dir/$val_rel")
  done
  train_files="${TRAIN_FILES:-$(join_hydra_list "${train_arr[@]}")}"
  val_files="${VAL_FILES:-$(join_hydra_list "${val_arr[@]}")}"
else
  resolve_task_files "$TASK"
  train_files="${TRAIN_FILES:-$osim_dir/$train_rel}"
  val_files="${VAL_FILES:-$osim_dir/$val_rel}"
fi

# ── Hyperparameters ───────────────────────────────────────────────────────────
default_agent_loop="agent_hub"
agent_version="default"

loss_mode="vanilla"
clip_ratio_low=0.2
clip_ratio_high=0.28

actor_lr=8e-6
export MIN_THINK_TOKENS=4
max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 16))
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 2))

usp_size=1
train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=8
n_resp_per_prompt_val=1
infer_tp=1

lora_rank=32
lora_alpha=64
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
total_steps=500
save_freq=50
resume_mode=auto

# ── Setup ─────────────────────────────────────────────────────────────────────
export OPENAI_MODEL_NAME="gpt-5-5"
export JUDGE_MODEL_NAME=$OPENAI_MODEL_NAME
export LLAMA_API_KEY="${LLAMA_API_KEY:?set LLAMA_API_KEY (see scripts/env.sh)}"

export TURNOFF_THINK="${TURNOFF_THINK:-0}"
export USE_MODEL_CHAT_TEMPLATE="${USE_MODEL_CHAT_TEMPLATE:-1}"
export MIN_THINK_TOKENS="${MIN_THINK_TOKENS:-4}"
export HF_HOME=$OUTPUT_DIR/hf_cache
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p $OUTPUT_DIR/hf_cache $OUTPUT_DIR/wandb $OUTPUT_DIR/$EXPERIMENT_NAME

# ── Node-local compile caches (multi-node correctness) ────────────────────────
_cache_tag="${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="/tmp/triton-cache-$_cache_tag"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-cache-$_cache_tag"
export VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# ── Ray cluster (multi-node under SLURM) ──────────────────────────────────────
if [ -n "$SLURM_JOB_ID" ] && [ "$NNODES" -gt 1 ]; then
  ray_num_cpus="${SLURM_CPUS_PER_TASK:-96}"
  export RAY_NUM_NODES=$NNODES

  nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

  if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
      head_node_ip=${ADDR[1]}
    else
      head_node_ip=${ADDR[0]}
    fi
    echo "IPv6 address detected. Using IPv4: $head_node_ip"
  fi

  port=6379
  ip_head=$head_node_ip:$port
  export RAY_ADDRESS=$ip_head
  export MASTER_ADDR=$head_node_ip
  export MASTER_PORT=$port

  echo "Starting Ray HEAD at $head_node ($ip_head)"
  srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "$ray_num_cpus" --num-gpus "$GPUS_PER_NODE" --block &
  sleep 10

  worker_num=$((NNODES - 1))
  for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
      ray start --address "$ip_head" \
      --num-cpus "$ray_num_cpus" --num-gpus "$GPUS_PER_NODE" --block &
    sleep 5
  done
  sleep 10

  echo "==== Ray cluster resources ===="
  ray status --address "$ip_head" || true
  echo "==============================="
fi

# ── Train ─────────────────────────────────────────────────────────────────────
export LOGGING_LEVEL=ERROR
NCCL_DEBUG=WARN python3 train_ppo.py \
  hydra.run.dir=$OUTPUT_DIR/hydra \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=$default_agent_loop \
  actor_rollout_ref.rollout.agent.num_workers=64 \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.train_batch_size=$train_batch_size \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$actor_model_path \
  actor_rollout_ref.actor.freeze_vision_tower=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$usp_size \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length + 1024)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.n=$n_resp_per_prompt \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  algorithm.use_kl_in_reward=False \
  +algorithm.agent_version=$agent_version \
  trainer.n_gpus_per_node=$GPUS_PER_NODE \
  trainer.nnodes=$NNODES \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=MIMESIS-RL \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.val_before_train=True \
  trainer.save_freq=$save_freq \
  trainer.resume_mode=$resume_mode \
  trainer.max_actor_ckpt_to_keep=10 \
  trainer.max_critic_ckpt_to_keep=10 \
  trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
  trainer.test_freq=50 \
  trainer.total_training_steps=$total_steps \
  trainer.total_epochs=10000
