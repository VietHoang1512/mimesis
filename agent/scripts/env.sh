#!/usr/bin/env bash

export AGENT_ROOT="${AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AGENT_MODEL_DIR="${AGENT_MODEL_DIR:-$AGENT_ROOT/models}"
export AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_ROOT/data}"
export AGENT_OUTPUT_DIR="${AGENT_OUTPUT_DIR:-$AGENT_ROOT/outputs}"
export AGENT_EVAL_DIR="${AGENT_EVAL_DIR:-$AGENT_OUTPUT_DIR/eval}"

# --- the policy being trained -------------------------------------------------
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$AGENT_MODEL_DIR/sft/qwen3-8b}"

# --- the simulated user -------------------------------------------------------
export SIM_MODEL_NAME="${SIM_MODEL_NAME:-mimesis-9b}"
export SIM_BASE_URL="${SIM_BASE_URL:-http://127.0.0.1:8710/v1}"

export OPENAI_BASE_URL="$SIM_BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export MULTITURN_MODEL_NAME="$SIM_MODEL_NAME"

# --- gym behaviour ------------------------------------------------------------
export USERRL_SIM_TIMEOUT="${USERRL_SIM_TIMEOUT:-35}"
export USERRL_GYM_SEED="${USERRL_GYM_SEED:-0}"

mkdir -p "$AGENT_OUTPUT_DIR"
export HF_HOME="${HF_HOME:-$AGENT_OUTPUT_DIR/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-$AGENT_OUTPUT_DIR/wandb}"

export CONFIG_PATH="$AGENT_ROOT/examples/sglang_multiturn/config"
