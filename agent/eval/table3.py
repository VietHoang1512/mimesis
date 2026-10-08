#!/usr/bin/env python3

import argparse
import json
import os
import pathlib

COLUMNS = [
    ("travel", "TravelGym"), ("turtle", "TurtleGym"), ("function", "FunctionGym"),
    ("tau", "TauGym"), ("persuasion", "PersuadeGym"),
    ("intention", "IntentionGym"), ("telepathy", "TelepathyGym"),
    ("bamboogle", "SearchGym"),
]
HELD_OUT = {"intention", "telepathy", "bamboogle"}


def score_file(path: pathlib.Path) -> dict:
    """Aggregate one reward cache into per-gym scores and per-task uid counts."""
    data = json.loads(path.read_text())
    per_task, travel_avgs = {}, []
    counts = {}
    for task, entries in data.items():
        if task == "interact":
            continue
        if "travel" not in task:
            uid_avgs = []
            for content in entries.values():
                per_rollout = [sum(rollout) for rollout in content["reward"]]
                if per_rollout:
                    uid_avgs.append(sum(per_rollout) / len(per_rollout))
            per_task[task] = sum(uid_avgs) / len(uid_avgs) if uid_avgs else 0.0
            counts[task] = counts.get(task, 0) + len(entries)
        else:
            aspect_num = len(task.split("travel")[-1])
            counts["travel"] = counts.get("travel", 0) + len(entries)
            for content in entries.values():
                per_rollout = []
                for history in content["history"]:
                    best = {}
                    for turn in history:
                        if turn["choice"] == "answer" and turn["content"] != "":
                            k = turn["content"][0]
                            best[k] = max(best.get(k, 0.0), turn["reward"])
                    per_rollout.append(sum(best.values()) / aspect_num)
                if per_rollout:
                    travel_avgs.append(sum(per_rollout) / len(per_rollout))
    per_task["travel"] = sum(travel_avgs) / len(travel_avgs) if travel_avgs else 0.0
    return per_task, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_out = (pathlib.Path(os.environ["EVAL_OUTPUT_DIR"]) / "outputs"
                    if os.environ.get("EVAL_OUTPUT_DIR")
                    else pathlib.Path(__file__).parent / "outputs")
    ap.add_argument("--outputs", type=pathlib.Path, default=_default_out)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--simulator", default="",
                    help="only files whose __<simulator>__ segment matches this")
    ap.add_argument("--complete-only", action="store_true",
                    help="skip agents that have not finished all 15 gyms")
    args = ap.parse_args()

    files = sorted(args.outputs.glob("*_reward_cache.json"))
    if args.simulator:
        files = [f for f in files if f"__{args.simulator}__" in f.name]
    if not files:
        print(f"no matching *_reward_cache.json under {args.outputs}"); return

    ENVS = ("travel22 travel33 travel44 travel233 travel333 travel334 travel444 "
            "travel2222 function intention persuasion tau telepathy turtle "
            "bamboogle_closedbook").split()

    rows = {}
    for f in files:
        if args.complete_only:
            try:
                r = json.loads(f.with_name(
                    f.name.replace("_reward_cache.json", "_results.json")).read_text())
            except Exception:
                continue
            if any("1" not in r.get(e, {}) for e in ENVS):
                continue
        name = f.name.replace("_reward_cache.json", "").replace("results_", "")
        if args.simulator:
            name = name.split("__")[0]
        rows[name] = score_file(f)
    if not rows:
        print(f"no COMPLETE cells for simulator {args.simulator!r} yet"); return
    counts = {}
    for _sc, _c in rows.values():
        for k, v in _c.items():
            counts[k] = max(counts.get(k, 0), v)
    rows = {n: sc for n, (sc, _c) in rows.items()}

    keys = [k for k, _ in COLUMNS]
    heads = [n for _, n in COLUMNS]

    def avg(scores):
        """Weighted by #uids per task."""
        num = sum(counts.get(k, 0) * scores[k] for k in keys if k in scores)
        den = sum(counts.get(k, 0) for k in keys if k in scores)
        return num / den if den else 0.0

    if args.latex:
        print("Model & " + " & ".join(heads) + " & Avg. \\\\\n\\midrule")
        for name, sc in rows.items():
            cells = [f"{sc[k]:.4f}" if k in sc else "--" for k in keys]
            print(f"{name} & " + " & ".join(cells) + f" & {avg(sc):.4f} \\\\")
        return

    w = max(len(n) for n in rows) + 2
    print("| " + "Model".ljust(w) + " | " + " | ".join(h.rjust(11) for h in heads) + " |    Avg. |")
    print("|" + "-" * (w + 2) + "|" + "|".join(["-" * 13] * len(heads)) + "|--------:|")
    for name, sc in rows.items():
        cells = [(f"{sc[k]:.4f}" if k in sc else "n/a").rjust(11) for k in keys]
        print(f"| {name.ljust(w)} | " + " | ".join(cells) + f" | {avg(sc):.4f} |")
    missing = {k for sc in rows.values() for k in keys if k not in sc}
    if missing:
        names = [dict(COLUMNS)[k] for k in missing]
        print(f"\nn/a: {', '.join(sorted(names))} -- not evaluated "
              f"(SearchGym needs SERPER_API_KEY). Held-out gyms: "
              f"{', '.join(dict(COLUMNS)[k] for k in HELD_OUT)}.")


if __name__ == "__main__":
    main()
