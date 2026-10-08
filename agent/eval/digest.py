#!/usr/bin/env python3
"""Extract a compact score digest from a completed evaluation sweep."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_main_table as M  # noqa: E402

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

    back = {src: key for key, src in SOURCE_KEY.items()}
    M.SECTIONS = [(label, SOURCE_KEY.get(key, key)) for label, key in M.SECTIONS]
    data = M.from_table3(args.outputs)

    scores, n = {}, 0
    keep = set(M.AGENTS)
    for sim, rows in data.items():
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
