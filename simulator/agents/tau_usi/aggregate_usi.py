# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate tau-USI per-run metrics into one leaderboard-style table.

``usi_metric score`` writes a ``<label>_aggregate_metrics.json`` per run (one per
model×seed). This collects them, averages each metric across seeds of the same model,
and prints (and writes) a markdown table sorted by USI. Model = label with the
``-seed<N>`` suffix stripped, so it pairs with ``run_tau_usi_eval.sh``'s labels.

Usage:
  python -m agents.tau_usi.aggregate_usi                     # scans outputs/tau_usi
  python -m agents.tau_usi.aggregate_usi --dir DIR --out FILE.md
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Columns to show; keys match usi_metric's aggregate_metrics "model" dict. Each value
# there is a [mean, std] pair (or null when the term is unavailable, e.g. eval_agree
# with no survey_data). ECE is stored 0-1 and shown ×100.
METRICS = [
    ("D1_conv", "D1 Conv", 1.0),
    ("D2_info", "D2 Info", 1.0),
    ("D3_clarif", "D3 Clarif", 1.0),
    ("D4_react", "D4 React", 1.0),
    ("eval_agree", "Eval", 1.0),
    ("ece", "ECE", 100.0),
    ("usi", "USI", 1.0),
]


def _mean_of_pair(entry, key):
    """Return the mean component of a [mean, std] metric pair, or None."""
    pair = (entry or {}).get(key)
    if isinstance(pair, (list, tuple)) and pair and pair[0] is not None:
        return float(pair[0])
    return None


def _model_of_label(label: str) -> str:
    return re.sub(r"-seed\d+$", "", label)


def collect(metrics_dir: Path):
    """{model: {metric_key: [per-seed means]}} plus the human baseline row."""
    per_model = defaultdict(lambda: defaultdict(list))
    n_seeds = defaultdict(set)
    human = None
    for path in sorted(glob.glob(str(metrics_dir / "*_aggregate_metrics.json"))):
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {path}: {e}")
            continue
        label = data.get("label") or Path(path).name.replace("_aggregate_metrics.json", "")
        model = _model_of_label(label)
        m = re.search(r"-seed(\d+)$", label)
        if m:
            n_seeds[model].add(m.group(1))
        row = data.get("model") or {}
        for key, _, _ in METRICS:
            v = _mean_of_pair(row, key)
            if v is not None:
                per_model[model][key].append(v)
        if human is None and data.get("human_inter_annotator"):
            human = data["human_inter_annotator"]
    return per_model, n_seeds, human


def _fmt(v, scale):
    return f"{v * scale:.2f}" if v is not None else "NA"


def _row_cells(metric_means: dict):
    cells = []
    for key, _, scale in METRICS:
        vals = metric_means.get(key) or []
        cells.append(_fmt(float(np.mean(vals)) if vals else None, scale))
    return cells


def build_table(per_model, n_seeds, human) -> str:
    header = ["Model", "seeds"] + [name for _, name, _ in METRICS]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    if human:
        hcells = []
        for key, _, scale in METRICS:
            hcells.append(_fmt(_mean_of_pair(human, key), scale))
        lines.append("| " + " | ".join(["Human (inter-ann.)", "-"] + hcells) + " |")

    # Sort models by USI descending (NA last).
    def usi_of(model):
        vals = per_model[model].get("usi") or []
        return float(np.mean(vals)) if vals else -1.0

    for model in sorted(per_model, key=usi_of, reverse=True):
        cells = _row_cells(per_model[model])
        seeds = str(len(n_seeds.get(model, set())) or 1)
        lines.append("| " + " | ".join([model, seeds] + cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs/tau_usi", help="dir of *_aggregate_metrics.json (default outputs/tau_usi)")
    ap.add_argument("--out", default=None, help="write the markdown table here (default <dir>/usi_summary.md)")
    args = ap.parse_args(argv)

    metrics_dir = Path(args.dir)
    per_model, n_seeds, human = collect(metrics_dir)
    if not per_model:
        raise SystemExit(f"no *_aggregate_metrics.json under {metrics_dir} — run the eval + usi_metric score first")

    table = build_table(per_model, n_seeds, human)
    print(table)
    out = Path(args.out) if args.out else metrics_dir / "usi_summary.md"
    out.write_text(table + "\n")
    print(f"\n[aggregate_usi] {len(per_model)} models -> {out}")


if __name__ == "__main__":
    main()
