#!/usr/bin/env bash
# Stage 2 control -- plain multi-turn GRPO.
#
#   bash scripts/run_grpo.sh
#
# Identical to run_sdpo.sh in every respect except SDPO_COEF=0, which makes the
# auxiliary term vanish. Running the control through the same binary with one
# value changed is the tightest control available; a separately-written baseline
# would differ in ways that are hard to enumerate.
set -euo pipefail
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-ctl}"
export SDPO_COEF=0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_train_common.sh"
