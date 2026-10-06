#!/usr/bin/env bash
# Stage 2 -- our method: multi-turn GRPO plus the SDPO self-distillation term.
#
#   bash scripts/run_sdpo.sh
#
# Anything after the script name is passed through to hydra, so a short smoke
# run is:  bash scripts/run_sdpo.sh trainer.total_training_steps=2
set -euo pipefail
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sdpo-a0.01}"
export SDPO_COEF="${SDPO_COEF:-0.01}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_train_common.sh"
