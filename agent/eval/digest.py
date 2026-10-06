#!/usr/bin/env python3
"""Extract a compact score digest from a completed evaluation sweep.

`eval/table3.py` and `eval/make_main_table.py` read `*_reward_cache.json`, which
for the full sweep (6 agents x 9 simulators x 15 gyms) runs to several GB and
takes roughly 40 GPU-hours to regenerate. That is not something a reader can be
asked to reproduce just to see the main table.

This walks those caches once and writes the aggregated scores -- a few hundred
KB -- so `make_main_table.py --digest` reproduces the table exactly.

It deliberately calls `make_main_table.from_table3()` rather than
reimplementing the aggregation, so the digest cannot drift from the live path.

    EVAL_OUTPUT_DIR=/path/to/outputs python eval/digest.py -o results/main_results.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_main_table as M  # noqa: E402

# Digest key (as used in make_main_table.SECTIONS) -> the key that appears in
# the eval filenames, for simulators whose served name differs from the name
# used in the paper. One simulator in this sweep was reached under a
# deployment-specific alias; such an alias is an artifact of routing and carries
# no meaning, so the digest records the plain name. Extend this if your own
# serving names differ.
SOURCE_KEY = {
    k: v for k, v in
    (p.split("=", 1) for p in os.environ.get("DIGEST_SOURCE_KEYS", "").split(",") if "=" in p)
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="results/main_results.json")
    ap.add_argument("--outputs", default=os.environ.get("EVAL_OUTPUT_DIR", "outputs"))
    args = ap.parse_args()

    # Query table3 with the key the files actually use, store under the digest
    # key that make_main_table expects to read back.
    back = {src: key for key, src in SOURCE_KEY.items()}
    M.SECTIONS = [(label, SOURCE_KEY.get(key, key)) for label, key in M.SECTIONS]
    data = M.from_table3(args.outputs)

    scores, n = {}, 0
    keep = set(M.AGENTS)
    for sim, rows in data.items():
        # Restrict to the agents in the paper. The sweep directory also holds
        # intermediate checkpoints and abandoned arms; carrying those into a
        # released artifact invites them to be read as results.
        rows = {a: v for a, v in rows.items() if a in keep}
        if not rows:
            continue
        scores[back.get(sim, sim)] = rows
        n += len(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "note": "Aggregated by eval/table3.py; values are x100. Column order "
                    "is 8 gyms then uid-weighted Avg. Regenerate with eval/digest.py.",
            "columns": ["TravelGym", "FunctionGym", "IntentionGym", "PersuadeGym",
                        "TauGym", "TelepathyGym", "TurtleGym", "SearchGym", "Avg."],
            "scores": scores,
        }, fh, indent=1, sort_keys=True)
    print(f"  wrote {args.out}: {len(scores)} simulators, {n} agent rows "
          f"({os.path.getsize(args.out)} B)", file=sys.stderr)


if __name__ == "__main__":
    main()
