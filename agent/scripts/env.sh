#!/usr/bin/env bash
# Shared environment for every driver in scripts/. Source it, do not run it.
#
# The published runs used cluster-internal absolute paths; those are gone. Point
# these four at your own storage and every script below works unchanged.
#
#   AGENT_MODEL_DIR    base checkpoints and your SFT/RL outputs
#   AGENT_DATA_DIR     rl_split/ (train) and the per-gym eval parquets
#   AGENT_OUTPUT_DIR   run outputs, checkpoints, wandb, HF cache
#   AGENT_EVAL_DIR     where evaluation writes results/ and reward caches
#
# The user simulator and the judge are reached over an OpenAI-compatible API.
# Set SIM_BASE_URL / JUDGE_BASE_URL to any such endpoint -- vLLM, SGLang, or a
# hosted provider. See llm/README.md.

export AGENT_ROOT="${AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AGENT_MODEL_DIR="${AGENT_MODEL_DIR:-$AGENT_ROOT/models}"
export AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_ROOT/data}"
export AGENT_OUTPUT_DIR="${AGENT_OUTPUT_DIR:-$AGENT_ROOT/outputs}"
export AGENT_EVAL_DIR="${AGENT_EVAL_DIR:-$AGENT_OUTPUT_DIR/eval}"

# --- the policy being trained -------------------------------------------------
# Stage 1 (scripts/run_sft.sh) produces this. Training starts from it, not from
# the stock base model: the RL curves in the paper all begin at an SFT warm start.
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$AGENT_MODEL_DIR/sft/qwen3-8b}"

# --- the simulated user -------------------------------------------------------
# Serve your simulator (scripts/serve_simulators.sh), then put the shim in front
# of it so reasoning is stripped and captured. SIM_BASE_URL is the SHIM, not the
# model server -- the gyms must never see a raw <think> block. See llm/shim.py.
export SIM_MODEL_NAME="${SIM_MODEL_NAME:-mimesis-9b}"
export SIM_BASE_URL="${SIM_BASE_URL:-http://127.0.0.1:8710/v1}"

# verl's gym configs read OPENAI_BASE_URL at __post_init__ time, so it has to be
# exported before Ray starts on a multi-node run.
export OPENAI_BASE_URL="$SIM_BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export MULTITURN_MODEL_NAME="$SIM_MODEL_NAME"

# --- the judge ----------------------------------------------------------------
# Unset by default: the simulator grades its own interaction, which is the
# upstream behaviour and what every number in the paper used. Export these only
# if you want a separate judge model.
#   export JUDGE_MODEL_NAME=... JUDGE_BASE_URL=... JUDGE_API_KEY=...

# --- gym behaviour ------------------------------------------------------------
# Default gym timeout is 10 s, which silently turns a slow simulator turn into a
# canned fallback worth 0 reward. 35 s fits inside the rollout's own step budget.
export USERRL_SIM_TIMEOUT="${USERRL_SIM_TIMEOUT:-35}"
export USERRL_GYM_SEED="${USERRL_GYM_SEED:-0}"

mkdir -p "$AGENT_OUTPUT_DIR"
export HF_HOME="${HF_HOME:-$AGENT_OUTPUT_DIR/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-$AGENT_OUTPUT_DIR/wandb}"

# Hydra config root. Both training scripts pass this unchanged.
export CONFIG_PATH="$AGENT_ROOT/examples/sglang_multiturn/config"
