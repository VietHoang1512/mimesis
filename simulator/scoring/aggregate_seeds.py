#!/usr/bin/env python3
"""Aggregate SOUL eval results across seeds.

Both run_eval.sh (local checkpoints) and run_eval_api.sh (remote API models)
write one outputs/<EXPERIMENT_NAME>-eval.scores.json per run, where
EXPERIMENT_NAME ends in "-seed<N>". This groups those by run key (the name with
the seed suffix stripped) and reports mean +/- std across seeds.

Axis averages and Overall are computed per seed and THEN averaged, so the std
reflects run-to-run variation of the reported number itself.

Usage:
  python aggregate_seeds.py                # all runs, markdown
  python aggregate_seeds.py --min-seeds 2  # only runs with >=2 seeds
  python aggregate_seeds.py --detailed     # per-benchmark table too
"""
import argparse
import datetime
import glob
import json
import os
import re
import statistics
from collections import defaultdict

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SUFFIX = "-eval.scores.json"
SEED_RE = re.compile(r"-seed(\d+)$")

AXIS_MAP = {
    "CONV": ["UserLLM", "MirrorBench", "Humanual-Chat", "SimArena-Doc"],
    "SS": ["Sotopia-Hard"],
    "COG": ["Fantom", "Hitom", "Paratomi", "Social-R1"],
    "ROLE": ["Coser", "Lifechoices", "Twinvoice", "BehaviorChain",
             "SimArena-Math", "Mistakes", "Humanual-Email", "Humanual-News",
             "Humanual-Politics", "Humanual-Book", "Humanual-Opinion"],
    "EVAL": ["AlignX-Demo", "AlignX-Pair", "AlignX-UGC", "AlignX-Arbitrary",
             "AlignX-History16", "SocSci210", "HumanLLM"],
}
BENCHMARKS = [b for keys in AXIS_MAP.values() for b in keys]


# agents/sotopia/agent.py switched its reward from `0.1 * eval_result["actor_avg"]`
# to the raw `reward` between the 2026-08-05 and 2026-08-07 eval batches. The old
# form lands in roughly -6..35, the new one in 59..72, so Sotopia-Hard is on two
# incompatible scales and must never be averaged or ranked across the boundary.
# Era is taken from the -eval.log mtime (when the eval actually ran); the
# scores.json mtime is unreliable because aggregate.py gets re-run over old logs.
SOTOPIA_RESCALE = 1786060800.0  # 2026-08-06T00:00:00 local


def num(v):
    return v if isinstance(v, (int, float)) else None


def load_runs():
    """-> {run_key: {seed_or_None: (scores_dict, ran_at)}}"""
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(OUTPUTS, "*" + SUFFIX))):
        base = os.path.basename(path)[: -len(SUFFIX)]
        try:
            scores = json.load(open(path))
        except json.JSONDecodeError:
            continue
        # Drop runs that produced no numeric metric at all (crashed eval).
        if not any(isinstance(v, (int, float)) for v in scores.values()):
            continue
        log = path[: -len(SUFFIX)] + "-eval.log"
        ran_at = os.path.getmtime(log if os.path.exists(log) else path)
        m = SEED_RE.search(base)
        key, seed = (base[: m.start()], int(m.group(1))) if m else (base, None)
        runs[key][seed] = (scores, ran_at)
    return runs


def degenerate_seeds(by_seed):
    """Seeds whose eval visibly broke mid-run, as opposed to a genuinely bad model.

    Signature: a block of benchmarks at exactly 0.0 AND an Overall far below the
    group's median. Both conditions matter -- grok-4 (dead gateway route) and the
    thoughttrace-330442 checkpoint score ~0 on every seed consistently, which is a
    real result and must be kept; gpt-5.5 seed1 scored 22.9 next to 62.7 on the
    sibling seed, which is infrastructure noise and must not enter the mean.
    """
    overalls = [num(s.get("Overall")) for s, _ in by_seed.values()]
    overalls = [o for o in overalls if o is not None]
    if len(overalls) < 2:
        return set()
    med = statistics.median(overalls)
    bad = set()
    for seed, (s, _) in by_seed.items():
        zeros = sum(1 for b in BENCHMARKS if s.get(b) == 0)
        o = num(s.get("Overall"))
        if zeros >= 8 and o is not None and med - o > 15:
            bad.add(seed)
    return bad


def axis_value(scores, keys):
    vals = [num(scores.get(k)) for k in keys]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(seed_scores):
    """seed_scores: list of per-seed score dicts -> {metric: (mean, std, n)}"""
    out = {}
    fields = {b: (lambda s, b=b: num(s.get(b))) for b in BENCHMARKS}
    fields.update({a: (lambda s, k=k: axis_value(s, k)) for a, k in AXIS_MAP.items()})
    fields["Overall"] = lambda s: num(s.get("Overall"))
    # score_v1: the harness's average over ONLY the Ditto-paper datasets, i.e. the
    # number comparable to published Ditto results. Overall (=score_v2) spans all 27.
    fields["Overall_v1"] = lambda s: num(s.get("Overall_v1"))
    for name, fn in fields.items():
        vals = [v for v in (fn(s) for s in seed_scores) if v is not None]
        if not vals:
            out[name] = (None, None, 0)
        else:
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out[name] = (sum(vals) / len(vals), std, len(vals))
    return out


def cell(stat, seeds_expected):
    mean, std, n = stat
    if mean is None:
        return "—"
    if n == 1:
        return f"{mean:.1f}"
    s = f"{mean:.1f}±{std:.1f}"
    if n < seeds_expected:  # some seeds missing this metric
        s += f" ({n})"
    return s


def kind(key):
    return "api" if "-eval-usersim-" in key else "local"


def model_of(key):
    return key.split("-eval-", 1)[0]


def config_of(key):
    """Everything after the model name and judge, e.g. think-modeltmpl."""
    tail = key.split("-eval-", 1)[1] if "-eval-" in key else ""
    tail = tail.replace("usersim-", "").replace("gpt-5-5", "").strip("-")
    return tail or "default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seeds", type=int, default=1)
    ap.add_argument("--detailed", action="store_true")
    ap.add_argument("--filter", default="", help="substring match on run key")
    ap.add_argument("--src", choices=["api", "local"], help="only this transport")
    ap.add_argument("--since", metavar="YYYY-MM-DD",
                    help="only runs evaluated on/after this date; adds a Ran column")
    args = ap.parse_args()

    # Dates come from the -eval.log mtime (when the eval actually ran), the same
    # source the Sotopia-era tag uses -- the scores.json mtime is unreliable because
    # aggregate.py gets re-run over old logs.
    since_ts = None
    if args.since:
        since_ts = datetime.datetime.strptime(args.since, "%Y-%m-%d").timestamp()

    runs = load_runs()
    rows, dropped = [], []
    for key, by_seed in runs.items():
        if args.filter and args.filter not in key:
            continue
        if args.src and kind(key) != args.src:
            continue
        bad = degenerate_seeds(by_seed)
        dropped += [f"{key}-seed{s}" for s in sorted(bad, key=lambda x: (x is None, x))]
        ok = {s: v for s, v in by_seed.items() if s not in bad}
        seeded = {s: v for s, v in ok.items() if s is not None}
        legacy = None in ok
        # An unseeded run predates the SEEDS loop; use it only as a lone
        # fallback so it never gets averaged into a properly seeded set.
        use = seeded if seeded else {s: v for s, v in ok.items()}
        if not use or len(use) < args.min_seeds:
            continue
        scores = [s for s, _ in use.values()]
        ran_at = max(t for _, t in use.values())
        if since_ts is not None and ran_at < since_ts:
            continue
        era = "new" if ran_at >= SOTOPIA_RESCALE else "old"
        rows.append({
            "key": key,
            "model": model_of(key),
            "config": config_of(key),
            "kind": kind(key),
            "era": era,
            "ran": datetime.date.fromtimestamp(ran_at).isoformat(),
            "n": len(use),
            "seeds": sorted(s for s in use if s is not None),
            "legacy_extra": legacy and bool(seeded),
            "stats": summarize(scores),
        })

    rows.sort(key=lambda r: (r["stats"]["Overall"][0] is None,
                             -(r["stats"]["Overall"][0] or 0)))

    axes = list(AXIS_MAP)
    hdr = ["Model", "Config", "Src", "Seeds"]
    if args.since:
        hdr.append("Ran")
    hdr += axes + ["Overall"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        seeds = ",".join(str(s) for s in r["seeds"]) if r["seeds"] else "unseeded"
        cells = [r["model"], r["config"], r["kind"], f"{r['n']} ({seeds})"]
        if args.since:
            cells.append(r["ran"])
        for a in axes:
            c = cell(r["stats"][a], r["n"])
            # Mark the Sotopia scale the row was produced under.
            if a == "SS" and c != "—":
                c += "ᴺ" if r["era"] == "new" else "ᴼ"
            cells.append(c)
        cells.append(cell(r["stats"]["Overall"], r["n"]))
        print("| " + " | ".join(cells) + " |")

    if args.detailed:
        print()
        hdr = ["Model", "Config", "Seeds"] + BENCHMARKS + ["Overall"]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for r in rows:
            cells = [r["model"], r["config"], str(r["n"])]
            cells += [cell(r["stats"][b], r["n"]) for b in BENCHMARKS]
            cells.append(cell(r["stats"]["Overall"], r["n"]))
            print("| " + " | ".join(cells) + " |")

    print("\nSS scale: ᴼ = pre-2026-08-06 (0.1*actor_avg), "
          "ᴺ = post (raw reward). Not comparable across eras.")
    extra = [r["key"] for r in rows if r["legacy_extra"]]
    if extra:
        print(f"Note: {len(extra)} run(s) also have an older unseeded "
              f"scores.json, excluded from the average.")
    if dropped:
        print(f"Excluded {len(dropped)} broken seed run(s): {', '.join(dropped)}")


if __name__ == "__main__":
    main()
