#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Stage 1, the supervised warm start. See sft/README.md.
#   bash scripts/run_sft.sh /path/to/Qwen3-8B [hydra-style overrides...]
# Needs llamafactory-cli on PATH (pip install -e your LLaMA-Factory checkout).
BASE_MODEL="${1:?usage: run_sft.sh BASE_MODEL_PATH [overrides...]}"
shift                      # remaining args are passed through as overrides
OUT="${OUT:-$AGENT_MODEL_DIR/sft/qwen3-8b}"

llamafactory-cli train "$AGENT_ROOT/sft/qwen3_customized.yaml" \
  model_name_or_path="$BASE_MODEL" \
  output_dir="$OUT" "$@"

python "$AGENT_ROOT/sft/fix_ckpt_tokenizer.py" "$OUT"
echo "SFT actor ready at $OUT -- set ACTOR_MODEL_PATH to it before stage 2"
