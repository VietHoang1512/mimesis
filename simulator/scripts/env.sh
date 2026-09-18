#!/usr/bin/env bash
# Shared environment for every driver in scripts/. Source it, do not run it.
#
# The published runs used Meta-internal absolute paths; those are gone. Point
# these four at your own storage and every script below works unchanged.
#
#   MIMESIS_MODEL_DIR   base checkpoints and your SFT/RL outputs
#   MIMESIS_DATA_DIR    sim_rl_data/ (train) and sim_eval_data/ (SOUL val parquets)
#   MIMESIS_OUTPUT_DIR  run outputs, checkpoints, wandb, HF cache
#   LLAMA_API_KEY       credential for the judge/partner model endpoint
#
# The judge and conversational partner are reached over an OpenAI-compatible
# API. Set OPENAI_AGENT_BASE_URL (and OPENAI_AGENT_API_KEY) to any such endpoint
# -- vLLM, SGLang, or a hosted provider. See llm/README.md.

export MIMESIS_ROOT="${MIMESIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MIMESIS_MODEL_DIR="${MIMESIS_MODEL_DIR:-$MIMESIS_ROOT/models}"
export MIMESIS_DATA_DIR="${MIMESIS_DATA_DIR:-$MIMESIS_ROOT/data}"
export MIMESIS_OUTPUT_DIR="${MIMESIS_OUTPUT_DIR:-$MIMESIS_ROOT/outputs}"

# Judge / partner model. OPENAI_MODEL_NAME is read by agents/utils.py at import
# time, so it has to be exported before Ray starts on a multi-node run.
export OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-gpt-5-5}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-$OPENAI_MODEL_NAME}"

mkdir -p "$MIMESIS_OUTPUT_DIR"
export HF_HOME="${HF_HOME:-$MIMESIS_OUTPUT_DIR/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-$MIMESIS_OUTPUT_DIR/wandb}"
