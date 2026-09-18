#!/bin/bash
# Eval-only run across the full SOUL eval suite (27 tasks).
#
# Usage:
#   bash recipe/ditto/eval.sh local   # eval a local checkpoint via vLLM
#   bash recipe/ditto/eval.sh api     # eval an API model (Claude / GPT / ...)

MODE="${1:-${MODE:-local}}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ditto-eval-$MODE}"
WANDB_PROJECT="${WANDB_PROJECT:-ditto}"
eval_dir=${MIMESIS_DATA_DIR}/sim_eval_data/
val_files="[\
$eval_dir/sotopia_hard_val.parquet,\
$eval_dir/coser_val.parquet,\
$eval_dir/lifechoices_val.parquet,\
$eval_dir/userllm_val.parquet,\
$eval_dir/mirrorbench_val.parquet,\
$eval_dir/fantom_val.parquet,\
$eval_dir/hitom_val.parquet,\
$eval_dir/paratomi_val.parquet,\
$eval_dir/mistakes_val.parquet,\
$eval_dir/twinvoice_val.parquet,\
$eval_dir/social_r1_val.parquet,\
$eval_dir/behaviorchain_val.parquet,\
$eval_dir/sim_math_val.parquet,\
$eval_dir/sim_doc_val.parquet,\
$eval_dir/humanual_book_val.parquet,\
$eval_dir/humanual_chat_val.parquet,\
$eval_dir/humanual_email_val.parquet,\
$eval_dir/humanual_news_val.parquet,\
$eval_dir/humanual_opinion_val.parquet,\
$eval_dir/humanual_politics_val.parquet,\
$eval_dir/alignx_demo_val.parquet,\
$eval_dir/alignx_pair_val.parquet,\
$eval_dir/alignx_ugc_val.parquet,\
$eval_dir/alignx_arbitrary_val.parquet,\
$eval_dir/alignx_history16_val.parquet,\
$eval_dir/socsci210_val.parquet,\
$eval_dir/humanllm_val.parquet]"

# ── Mode: pick which model to evaluate ────────────────────────────────────────
case "$MODE" in
  local)
    # Evaluate a local checkpoint (HF hub id or local path) via vLLM.
    default_agent_loop="agent_hub"
    actor_model_path="${ACTOR_MODEL_PATH:-sunweiwei/Ditto-8B}"
    ;;
  api)
    # Evaluate a remote API model via the OpenAI-compatible agent loop.
    # Caller is expected to export OPENAI_AGENT_MODEL / BASE_URL / API_KEY
    # (see README for OpenAI / Claude / Gemini / ... examples).
    default_agent_loop="openai_agent"
    # actor_model_path is still required for tokenizer; keep a small local model.
    # verl always builds a local vLLM engine even in api mode, so this small model
    # boots on GPU purely for tokenization / token bookkeeping (not scored).
    actor_model_path="${ACTOR_MODEL_PATH:-Qwen/Qwen3-4B-Base}"
    ;;
  *) echo "Unknown MODE: $MODE (expected: local | api)" >&2; exit 1 ;;
esac

# ── Hyperparameters ───────────────────────────────────────────────────────────
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 8))}
# Generation cap per turn. 8k suits a no-think model, but a THINKING model spends most
# of it on the reasoning block: run past the cap and generation stops without ever
# emitting </think>, so clean_response (agents/utils.py:1036) returns "" and the sample
# scores 0 — a silent floor that looks like a bad model, not a truncated one. run_eval.sh
# raises this to 16k whenever TURNOFF_THINK=0, matching run_rl.sh.
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}

n_resp_per_prompt_val=1
infer_tp=2
# In-flight agent loops (divided across rollout.agent.num_workers by the harness, see
# verl/experimental/agent_loop/agent_loop.py:476). 512 is right for a local vLLM engine,
# which is the only thing bounding it there, but an API user-simulator is metered by the
# provider instead: at 512 the Anthropic upstream returned
#   rate limit exceeded ... up-anthropic::w-60s: 517/475 in 60s
#   rate limit exceeded ... up-anthropic::w-600s: 2451/2450 in 600s
# and the 2026-08-11 claude-opus-5 / claude-sonnet-5 evals lost 27% of rollouts to
# zero-turn conversations. Overridable so API runs can drop below quota without moving
# the local default (run_eval_api.sh sets it).
max_concurrent_rollouts=${MAX_CONCURRENT_ROLLOUTS:-512}

export n_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Using $n_gpus GPU" "$CUDA_VISIBLE_DEVICES"
# ── Setup ─────────────────────────────────────────────────────────────────────
export HF_HOME=$OUTPUT_DIR/hf_cache
# Resolve models from the local cache only. Egress to huggingface.co is blocked here (the
# outbound proxy answers CONNECT with 403), and recent transformers calls list_repo_templates()
# to look for a chat template even when every file is already cached -- that call is not
# offline-safe, so it hangs until httpx.ConnectTimeout and takes the whole run with it.
# Job 430209 lost all 6 runs that way, dying in hf_tokenizer() right after config
# validation. run_eval_api.sh is the exposed one because its ACTOR_MODEL_PATH is a hub id
# (Qwen/Qwen3-4B-Base, tokenizer only) rather than a filesystem path.
# If a model is genuinely missing this now fails with a clear cache-miss instead of a
# 15-minute network hang; populate the cache from a node with egress first.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p $OUTPUT_DIR/hf_cache $OUTPUT_DIR/wandb $OUTPUT_DIR/$EXPERIMENT_NAME

# ── Eval ──────────────────────────────────────────────────────────────────────
export LOGGING_LEVEL=ERROR

# ── Work around vLLM 0.12.0 "undefined symbol: cublasGemmEx" ───────────────────
# PyTorch loads its bundled cuBLAS into a local symbol scope, but vLLM's _C
# extension references cublasGemmEx unversioned and expects it in the global
# namespace. Preload cuBLAS so the symbol resolves (Ray workers inherit this).
# _cublas_dir="$(python3 -c 'import os, nvidia.cublas.lib as m; print(os.path.dirname(m.__file__))' 2>/dev/null)"
# if [ -n "$_cublas_dir" ]; then
#   export LD_PRELOAD="$_cublas_dir/libcublas.so.12:$_cublas_dir/libcublasLt.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
# fi

NCCL_DEBUG=WARN python3 train_ppo.py \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=$default_agent_loop \
  data.train_files="$val_files" \
  data.val_files="$val_files" \
  data.train_batch_size=64 \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$actor_model_path \
  '+actor_rollout_ref.model.override_config={attn_implementation: sdpa}' \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.gpu_memory_utilization=.6 \
  actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length + 1024)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  +actor_rollout_ref.rollout.agent.max_concurrent_rollouts=$max_concurrent_rollouts \
  algorithm.use_kl_in_reward=False \
  trainer.n_gpus_per_node=$n_gpus \
  trainer.nnodes=1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
  trainer.total_training_steps=1
