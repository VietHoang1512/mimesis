#!/usr/bin/env bash
# Shared RL launcher. Not run directly -- run_sdpo.sh and run_grpo.sh source it
# after setting the handful of variables that differ between the two arms.
#
# Keeping one invocation means the control really is the control: the SDPO arm
# and the GRPO arm differ in the SDPO_* block and nothing else.
#
# Every value below is the one used for the run reported in the paper. See the
# hyper-parameter table for the same numbers in prose.
set -euo pipefail

: "${EXPERIMENT_NAME:?set by the calling script}"
: "${SDPO_COEF:?set by the calling script (0 disables SDPO)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
cd "$AGENT_ROOT"

# --- data ---------------------------------------------------------------------
TRAIN_FILES="${TRAIN_FILES:-$AGENT_DATA_DIR/rl_split/train.parquet}"
VAL_FILES="${VAL_FILES:-$AGENT_DATA_DIR/rl_split/val.parquet}"
for f in "$TRAIN_FILES" "$VAL_FILES"; do
  [ -f "$f" ] || { echo "missing $f -- run scripts/prepare_data.sh first" >&2; exit 1; }
done

# --- cluster ------------------------------------------------------------------
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# --- optimisation -------------------------------------------------------------
actor_lr=1e-6
train_batch_size=64
ppo_mini_batch_size=8          # -> 8 gradient updates per rollout batch
micro_bsz=2
n_resp_per_prompt=8            # GRPO group size
total_epochs=5
save_freq=5
test_freq=5

# --- sequence / multi-turn ----------------------------------------------------
max_prompt_length=1152
max_response_length=8192
max_turns=16
gamma=0.8                      # per TURN, not per token
turn_level_method=R2G
trajectory_score_method=R2G

# --- systems ------------------------------------------------------------------
infer_tp=1
gpu_mem_util=0.50
param_offload=False
optimizer_offload=False
act_offload=False

# --- SDPO ---------------------------------------------------------------------
# SDPO_COEF=0 makes the auxiliary term vanish, which is exactly the GRPO control.
export SDPO_COEF
export SDPO_ESTIMATOR="${SDPO_ESTIMATOR:-sdar}"
export SDPO_GATE_BETA="${SDPO_GATE_BETA:-5.0}"
export SDPO_TURNS_PER_EPISODE="${SDPO_TURNS_PER_EPISODE:-16}"
export SDPO_NORM="${SDPO_NORM:-masked}"
export SDPO_HINT_STYLE="${SDPO_HINT_STYLE:-prospective}"
export SDPO_HINT_INLINE="${SDPO_HINT_INLINE:-0}"
export SDPO_HINT_BASE_URL="${SDPO_HINT_BASE_URL:-$SIM_BASE_URL}"
export SDPO_HINT_MODEL="${SDPO_HINT_MODEL:-$SIM_MODEL_NAME}"
# 1 blanks both privileged channels before the hint is written -- the
# no-privileged-information ablation, not the full method.
export SDPO_ABLATE_PRIVILEGED="${SDPO_ABLATE_PRIVILEGED:-0}"

echo "=== $EXPERIMENT_NAME ==="
echo "  actor      : $ACTOR_MODEL_PATH"
echo "  simulator  : $SIM_MODEL_NAME via $SIM_BASE_URL"
echo "  sdpo       : coef=$SDPO_COEF estimator=$SDPO_ESTIMATOR beta=$SDPO_GATE_BETA"
echo "               style=$SDPO_HINT_STYLE k=$SDPO_TURNS_PER_EPISODE ablate=$SDPO_ABLATE_PRIVILEGED"
echo "  cluster    : ${NNODES}x${GPUS_PER_NODE} GPU"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_multiturn' \
    algorithm.adv_estimator=grpo_multiturn \
    algorithm.gamma=$gamma \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=$act_offload \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bsz \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.sdpo_coef="$SDPO_COEF" \
    actor_rollout_ref.actor.sdpo_estimator="$SDPO_ESTIMATOR" \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem_util \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bsz \
    actor_rollout_ref.rollout.multi_turn.max_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.model_name="$MULTITURN_MODEL_NAME" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/interact_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="$turn_level_method" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="$trajectory_score_method" \
    actor_rollout_ref.hybrid_engine=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='mimesis-agent' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$AGENT_OUTPUT_DIR/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.val_before_train=False \
    trainer.total_epochs=$total_epochs "$@"
