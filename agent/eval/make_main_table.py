#!/usr/bin/env python3
"""Emit the main-results LaTeX table from the finished eval cells."""
import argparse
import json
import os
import subprocess
import sys

SECTIONS = [
    ("GPT 5.6",          "gpt-5.6"),
    ("Opus 5.0",         "claude-opus-5"),
    ("Sonnet 5.0",       "claude-sonnet-5"),
    ("Gemini Flash 3.8", "gemini-3.8-flash"),
    ("Kimi-K3",          "moonshotai-kimi-k3"),
    ("Ditto-8B",         "Ditto-8B"),
    ("Osim-8B",          "osim-8b"),
    ("HumanLM-8B",       "humanlm-opinion"),
    ("Sotopia-7B",       "sotopia-rl-qwen-2.5-7B-grpo"),
]

AGENTS = [
    "Qwen3-8B-stock", "sft-qwen3-8b",
    "qwen3-8b-351852-best120",
    "grpo-ctl-443107-best110", "grpo-ctl-443257-best100",
    "sdpo-a0.01-ABL-466591-best65",
]

LABELS: dict[str, str] = {}

NCOL = 9   # 8 gyms + Avg.


def from_table3(outdir: str) -> dict:
    os.environ["EVAL_OUTPUT_DIR"] = outdir
    here = os.path.dirname(os.path.abspath(__file__))
    data = {}
    for _, sim in SECTIONS:
        r = subprocess.run(
            [sys.executable, os.path.join(here, "table3.py"),
             "--simulator", sim, "--complete-only"],
            capture_output=True, text=True).stdout
        rows = {}
        for line in r.splitlines():
            if not line.startswith("| ") or "Model" in line or line.startswith("|--"):
                continue
            p = [c.strip() for c in line.strip("|").split("|")]
            vals = []
            for c in p[1:]:
                try:
                    vals.append(float(c) * 100)
                except ValueError:
                    vals.append(None)
            if len(vals) == NCOL:
                rows[p[0]] = vals
        data[sim] = rows
    return data


def from_digest(path: str) -> dict:
    """Load the pre-extracted scores. Already x100, already in column order."""
    blob = json.load(open(path))
    return {sim: {a: list(v) for a, v in rows.items()}
            for sim, rows in blob["scores"].items()}


def emit(data: dict) -> None:
    """Print the LaTeX table body for `data` ({simulator: {agent: [9 vals]}})."""
    mean = {}
    for a in AGENTS:
        vs = [rows[a][-1] for rows in data.values()
              if a in rows and rows[a][-1] is not None]
        mean[a] = sum(vs) / len(vs) if vs else float("-inf")
    agents = sorted(AGENTS, key=lambda a: mean[a])

    # global best per column, across every finished section
    gbest = []
    for j in range(NCOL):
        vs = [v[j] for rows in data.values() for v in rows.values()
              if v[j] is not None]
        gbest.append(max(vs) if vs else None)

    def cell(v, j, sbest):
        if v is None:
            return "-"
        t = f"{v:.2f}"
        if gbest[j] is not None and abs(v - gbest[j]) < 1e-9:
            return f"\\textbf{{{t}}}"
        if sbest[j] is not None and abs(v - sbest[j]) < 1e-9:
            return f"\\underline{{{t}}}"
        return t

    def esc(n):
        return LABELS.get(n, n).replace("_", "\\_")

    for label, sim in SECTIONS:
        rows = data.get(sim, {})
        print("\\midrule")
        status = "" if rows else "  % no finished cells for this simulator"
        print(f"\\multicolumn{{10}}{{c}}{{\\textit{{w. {label}}}}} \\\\{status}")
        print("\\midrule")
        sbest = []
        for j in range(NCOL):
            vs = [v[j] for v in rows.values() if v[j] is not None]
            sbest.append(max(vs) if vs else None)
        for a in agents:
            v = rows.get(a)
            cells = (["-"] * NCOL if v is None
                     else [cell(v[j], j, sbest) for j in range(NCOL)])
            print(f"{esc(a)} & " + " & ".join(cells) + " \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--digest", help="read scores from a digest JSON instead of "
                                     "re-aggregating the raw reward caches")
    ap.add_argument("--outputs", default=os.environ.get("EVAL_OUTPUT_DIR", "outputs"),
                    help="directory of *_reward_cache.json (ignored with --digest)")
    args = ap.parse_args()
    emit(from_digest(args.digest) if args.digest else from_table3(args.outputs))


if __name__ == "__main__":
    main()
