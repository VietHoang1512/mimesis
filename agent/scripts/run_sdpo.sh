#!/usr/bin/env bash
# Stage 2, our method: multi-turn GRPO plus the CSD self-distillation term.
#
#   SDPO_HINT_BASE_URL=http://127.0.0.1:8711/v1 bash scripts/run_sdpo.sh
#
# The coach runs on its own shim, separate from the simulated user's shim
# (see agent/README.md, "Train"). Arguments after the script name are passed to
# hydra, so a short smoke run is:
#   SDPO_HINT_BASE_URL=http://127.0.0.1:8711/v1 bash scripts/run_sdpo.sh trainer.total_training_steps=2
set -euo pipefail
: "${SDPO_HINT_BASE_URL:?point SDPO_HINT_BASE_URL at the coach shim, e.g. http://127.0.0.1:8711/v1 (see agent/README.md)}"
export USERRL_SDPO=1   # enables the CSD term in the rollout and the actor
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sdpo-a0.01}"
export SDPO_COEF="${SDPO_COEF:-0.01}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_train_common.sh"
