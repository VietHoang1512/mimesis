#!/usr/bin/env bash
# Stage 2 control -- plain multi-turn GRPO.
set -euo pipefail
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-ctl}"
export SDPO_COEF=0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_train_common.sh"
