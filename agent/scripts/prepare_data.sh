#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
SRC="${USERRL_SRC:-$AGENT_ROOT/third_party/UserRL}"
[ -d "$SRC" ] || { echo "run scripts/install_gyms.sh first" >&2; exit 1; }

mkdir -p "$AGENT_DATA_DIR"
for d in "$SRC"/data/*_multiturn*; do
  [ -d "$d" ] && cp -r "$d" "$AGENT_DATA_DIR/"
done

python "$AGENT_ROOT/examples/data_preprocess/make_rl_val_split.py" \
  --train "$AGENT_DATA_DIR/alltrain_multiturn/train.parquet" \
  --outdir "$AGENT_DATA_DIR/rl_split" "$@"

python - <<'PY'
import os, pandas as pd
d = os.environ["AGENT_DATA_DIR"]
for s in ("train", "val"):
    p = f"{d}/rl_split/{s}.parquet"
    df = pd.read_parquet(p)
    print(f"  {s:5s} {len(df):5d} rows  {df['data_source'].value_counts().to_dict()}")
PY
