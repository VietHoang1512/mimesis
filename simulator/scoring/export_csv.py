#!/usr/bin/env python3
"""Dump SOUL eval results to CSV.

Writes two files, both covering every model in outputs/:

  soul_results.csv           one row per model+config, mean/std/n across seeds
  soul_results_per_seed.csv  one row per model+config+seed, raw scores

The per-seed file is the one to re-analyse from; the summary is derived from it.

Seed selection, degenerate-seed exclusion and the Sotopia-era tag all reuse
aggregate_seeds.py so the CSVs cannot drift from the markdown tables.

Usage: python export_csv.py [--outdir .]
"""
import argparse
import csv
import glob
import os
import re
import statistics

from aggregate_seeds import (
    AXIS_MAP,
    BENCHMARKS,
    SOTOPIA_RESCALE,
    axis_value,
    config_of,
    degenerate_seeds,
    kind,
    load_runs,
    model_of,
    num,
)

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Generation-health probes pulled straight from the eval logs. These are what
# distinguish a genuinely weak model from a broken transport: claude-opus-5 and
# grok-4 both emit zero tokens on lifechoices and never produce a coser metric,
# while a healthy run sits at ~230 chars and ~4.5 turns.
HEALTH_PROBES = {
    "lifechoices_resp_len": r"val-aux/lifechoices/lifechoices/response_length/mean@1[:\s]+([-\d.eE]+)",
    "coser_reward_raw": r"val-core/coser/reward/mean@1[:\s]+([-\d.eE]+)",
    "sim_arena_doc_num_turn": r"val-aux/sim_arena_doc/sim_arena_doc/num_turn/mean@1[:\s]+([-\d.eE]+)",
    "num_turns_mean": r"val-aux/num_turns/mean[:\s]+([-\d.eE]+)",
}


def probe_log(run_key, seed):
    """Read generation-health metrics out of one run's eval log."""
    suffix = f"-seed{seed}" if seed is not None else ""
    path = os.path.join(OUTPUTS, f"{run_key}{suffix}-eval.log")
    out = dict.fromkeys(HEALTH_PROBES, None)
    if not os.path.exists(path):
        return out
    text = open(path, errors="ignore").read()
    for name, pat in HEALTH_PROBES.items():
        m = re.findall(pat, text)
        if m:
            out[name] = float(m[-1])
    return out


def health_flag(probes):
    """OK / BROKEN / UNKNOWN from the probes of a single seed.

    BROKEN means the model produced no output where a working one produces
    plenty -- an infrastructure failure, not a low score.
    """
    lc, doc = probes["lifechoices_resp_len"], probes["sim_arena_doc_num_turn"]
    if lc is None and doc is None:
        return "UNKNOWN"
    if (lc is not None and lc == 0.0) or (doc is not None and doc < 0.5):
        return "BROKEN"
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    axes = list(AXIS_MAP)
    per_seed_rows, summary_rows = [], []

    for run_key, by_seed in sorted(load_runs().items()):
        bad = degenerate_seeds(by_seed)
        ok = {s: v for s, v in by_seed.items() if s not in bad}
        seeded = {s: v for s, v in ok.items() if s is not None}
        # An unseeded run predates the SEEDS loop; use it only as a lone fallback
        # so it is never averaged in alongside properly seeded runs.
        use = seeded if seeded else ok
        if not use:
            continue
        era = "new" if max(t for _, t in use.values()) >= SOTOPIA_RESCALE else "old"

        # Emit every seed we loaded, including the excluded ones, so nothing is
        # silently dropped from the raw dump -- `included` is the filter column.
        for seed, (scores, _) in sorted(by_seed.items(), key=lambda kv: (kv[0] is None, kv[0])):
            probes = probe_log(run_key, seed)
            row = {
                "run_key": run_key,
                "model": model_of(run_key),
                "config": config_of(run_key),
                "src": kind(run_key),
                "seed": "unseeded" if seed is None else seed,
                "included": seed in use,
                "exclude_reason": "degenerate" if seed in bad
                else ("" if seed in use else "superseded_by_seeded_runs"),
                "sotopia_era": era,
                "health": health_flag(probes),
                "Overall": num(scores.get("Overall")),
                "Overall_v1": num(scores.get("Overall_v1")),
            }
            row.update({a: axis_value(scores, k) for a, k in AXIS_MAP.items()})
            row.update({b: num(scores.get(b)) for b in BENCHMARKS})
            row.update(probes)
            per_seed_rows.append(row)

        used = [s for s, _ in use.values()]
        srow = {
            "run_key": run_key,
            "model": model_of(run_key),
            "config": config_of(run_key),
            "src": kind(run_key),
            "n_seeds": len(use),
            # "|"-joined, not ",": a comma here would force CSV quoting on every
            # row and break naive `cut -d,` inspection of the dump.
            "seeds": "|".join(str(s) for s in sorted(x for x in use if x is not None)) or "unseeded",
            "sotopia_era": era,
            "health": "BROKEN" if any(
                health_flag(probe_log(run_key, s)) == "BROKEN" for s in use
            ) else "OK",
        }
        fields = {b: (lambda s, b=b: num(s.get(b))) for b in BENCHMARKS}
        fields.update({a: (lambda s, k=k: axis_value(s, k)) for a, k in AXIS_MAP.items()})
        fields["Overall"] = lambda s: num(s.get("Overall"))
        fields["Overall_v1"] = lambda s: num(s.get("Overall_v1"))
        for name, fn in fields.items():
            vals = [v for v in (fn(s) for s in used) if v is not None]
            srow[f"{name}_mean"] = sum(vals) / len(vals) if vals else None
            srow[f"{name}_std"] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
            srow[f"{name}_n"] = len(vals)
        summary_rows.append(srow)

    summary_rows.sort(key=lambda r: (r["Overall_mean"] is None, -(r["Overall_mean"] or 0)))
    per_seed_rows.sort(key=lambda r: (r["run_key"], str(r["seed"])))

    def write(path, rows, header):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in header})
        print(f"wrote {path}  ({len(rows)} rows, {len(header)} cols)")

    meta = ["run_key", "model", "config", "src", "n_seeds", "seeds", "sotopia_era", "health"]
    metrics = ["Overall", "Overall_v1"] + axes + BENCHMARKS
    write(
        os.path.join(args.outdir, "soul_results.csv"),
        summary_rows,
        meta + [f"{m}_{s}" for m in metrics for s in ("mean", "std", "n")],
    )
    write(
        os.path.join(args.outdir, "soul_results_per_seed.csv"),
        per_seed_rows,
        ["run_key", "model", "config", "src", "seed", "included", "exclude_reason",
         "sotopia_era", "health"] + metrics + list(HEALTH_PROBES),
    )


if __name__ == "__main__":
    main()
