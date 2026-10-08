#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

BASE_MODEL="${1:?usage: run_sft.sh BASE_MODEL_PATH}"
LF_DIR="${LF_DIR:?set LF_DIR to your LLaMA-Factory checkout}"
OUT="${OUT:-$AGENT_MODEL_DIR/sft/qwen3-8b}"

llamafactory-cli train "$AGENT_ROOT/sft/qwen3_customized.yaml" \
  model_name_or_path="$BASE_MODEL" \
  output_dir="$OUT" "$@"

python "$AGENT_ROOT/sft/fix_ckpt_tokenizer.py" "$OUT"
echo "SFT actor ready at $OUT -- set ACTOR_MODEL_PATH to it before stage 2"
