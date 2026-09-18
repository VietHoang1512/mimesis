import json
import re
import sys
from pathlib import Path


# Dataset configs: maps paper name -> log key pattern
DATASET_CONFIG = {
    # CONV
    "UserLLM": {"key": "val-core/userllm/reward/mean@1"},
    "MirrorBench": {"key": "val-core/mirrorbench/reward/mean@1"},
    "Humanual-Chat": {"key": "val-core/humanual-chat/reward/mean@1"},
    "SimArena-Doc": {"key": "val-core/sim_arena_doc/reward/mean@1"},
    # SS
    # Raw-based weighted reward (0.1*raw_dim_avg), matching the paper table.
    # Can go negative because 4 of the 7 Sotopia dimensions have negative
    # ranges (relationship, secret, social_rules, financial); a negative value
    # means the eval run genuinely scored poorly / degenerated.
    # See agents/sotopia/agent.py.
    "Sotopia-Hard": {"key": "val-core/sotopia/reward/mean@1"},
    # COG
    "Fantom": {"key": "val-core/fantom/reward/mean@1"},
    "Hitom": {"key": "val-core/hitom/reward/mean@1"},
    "Paratomi": {"key": "val-core/paratomi/reward/mean@1"},
    "Social-R1": {"key": "val-core/social_r1/reward/mean@1"},
    # ROLE
    "Coser": {"key": "val-core/coser/reward/mean@1"},
    "Lifechoices": {"key": "val-core/lifechoices/reward/mean@1"},
    "Twinvoice": {"key": "val-core/twinvoice/reward/mean@1"},
    "BehaviorChain": {"key": "val-core/behavior_chain/reward/mean@1"},
    "SimArena-Math": {"key": "val-core/sim_arena_math/reward/mean@1"},
    "Mistakes": {"key": "val-core/mistakes/reward/mean@1"},
    "Humanual-Email": {"key": "val-core/humanual-email/reward/mean@1"},
    "Humanual-News": {"key": "val-core/humanual-news/reward/mean@1"},
    "Humanual-Politics": {"key": "val-core/humanual-politics/reward/mean@1"},
    # The eval emits 27 datasets; these two were absent from this map, so they were
    # dropped from every per-task and per-category number. Overall was unaffected
    # (it reads val-core/all/score, which the harness computes over ALL data), which
    # is exactly why the omission was invisible -- the categories simply stopped
    # reconciling with Overall.
    "Humanual-Book": {"key": "val-core/humanual-book/reward/mean@1"},
    "Humanual-Opinion": {"key": "val-core/humanual-opinion/reward/mean@1"},
    # EVAL
    "AlignX-Demo": {"key": "val-core/alignx_demo/reward/mean@1"},
    "AlignX-Pair": {"key": "val-core/alignx_pair/reward/mean@1"},
    "AlignX-UGC": {"key": "val-core/alignx_ugc/reward/mean@1"},
    "AlignX-Arbitrary": {"key": "val-core/alignx_arbitrary/reward/mean@1"},
    "AlignX-History16": {"key": "val-core/alignx_history16/reward/mean@1"},
    "SocSci210": {"key": "val-core/socsci210/reward/mean@1"},
    "HumanLLM": {"key": "val-core/humanllm/reward/mean@1"},
    # Aggregate
    "Overall": {"key": "val-core/all/score"},
    # score_v1 averages only the datasets used in the Ditto paper, so it is the
    # number comparable to published Ditto results; score (=score_v2) spans all 27.
    "Overall_v1": {"key": "val-core/all/score_v1"},
}

# AlignX sub-splits for computing the combined score
ALIGNX_SPLITS = [
    "AlignX-Demo",
    "AlignX-Pair",
    "AlignX-UGC",
    "AlignX-Arbitrary",
    "AlignX-History16",
]

# Axis grouping for display
AXIS_MAP = {
    "CONV": ["UserLLM", "MirrorBench", "Humanual-Chat", "SimArena-Doc"],
    "SS": ["Sotopia-Hard"],
    "COG": ["Fantom", "Hitom", "Paratomi", "Social-R1"],
    "ROLE": [
        "Coser",
        "Lifechoices",
        "Twinvoice",
        "BehaviorChain",
        "SimArena-Math",
        "Mistakes",
        "Humanual-Email",
        "Humanual-News",
        "Humanual-Politics",
    ],
    "EVAL": [
        "AlignX-Demo",
        "AlignX-Pair",
        "AlignX-UGC",
        "AlignX-Arbitrary",
        "AlignX-History16",
        "SocSci210",
        "HumanLLM",
    ],
}


def extract_metrics_from_log(log_path: str) -> dict:
    """
    Extract val-core reward metrics from an MIMESIS evaluation log.

    Returns dict mapping dataset name -> score (on 0-100 scale, matching paper Table 2).
    """
    log_text = Path(log_path).read_text()

    results = {}

    for dataset_name, config in DATASET_CONFIG.items():
        key = config["key"]
        # Value may be a bare number or wrapped as np.float64(...) / np.int32(...)
        num = r"(?:np\.\w+\()?\s*([-\d.eE]+)"
        # Pattern: key:value or 'key': value (handles both flat log lines and dict format)
        # Flat format: val-core/fantom/reward/mean@1:np.float64(0.5)
        pattern_flat = re.escape(key) + r"[:\s]+" + num
        # Dict format: 'val-core/fantom/reward/mean@1': np.float64(0.5)
        pattern_dict = r"'" + re.escape(key) + r"':\s*" + num

        match = re.search(pattern_dict, log_text) or re.search(pattern_flat, log_text)

        if match:
            raw_value = float(match.group(1))
            # Every key -- per-task rewards and the all/score composites alike --
            # is a 0-1 fraction, so one uniform x100. (There used to be a branch
            # here special-casing val-core/all/score; both arms did the same thing.)
            results[dataset_name] = raw_value * 100
        else:
            results[dataset_name] = None

    # Compute combined AlignX score (average of 5 sub-splits)
    alignx_scores = [results[s] for s in ALIGNX_SPLITS if results.get(s) is not None]
    if alignx_scores:
        results["AlignX (combined)"] = sum(alignx_scores) / len(alignx_scores)

    return results


def print_results(results: dict):
    """Pretty-print results grouped by axis."""
    print("=" * 70)
    print(f"{'Dataset':<20} {'Score (0-100)':<15} {'Axis':<6}")
    print("=" * 70)

    for axis, datasets in AXIS_MAP.items():
        for dataset in datasets:
            score = results.get(dataset)
            score_str = f"{score:.1f}" if score is not None else "N/A"
            print(f"{dataset:<20} {score_str:<15} {axis:<6}")
        print("-" * 70)

    # Print AlignX combined
    alignx_combined = results.get("AlignX (combined)")
    if alignx_combined is not None:
        print(f"{'AlignX (combined)':<20} {alignx_combined:<15.1f} {'EVAL':<6}")

    # Print overall
    overall = results.get("Overall")
    if overall is not None:
        print(f"{'Overall':<20} {overall:<15.1f} {'ALL':<6}")

    print("=" * 70)

    # Compute per-axis averages (excluding None and SimArena zeros)
    print("\nPer-Axis Averages:")
    for axis, datasets in AXIS_MAP.items():
        scores = [results[d] for d in datasets if results.get(d) is not None]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {axis}: {avg:.1f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_scores.py <log_file_path>")
        print("\nExample: python extract_scores.py eval.log")
        sys.exit(1)

    log_path = sys.argv[1]

    if not Path(log_path).exists():
        print(f"Error: File not found: {log_path}")
        sys.exit(1)

    print(f"Extracting scores from: {log_path}")
    print()

    results = extract_metrics_from_log(log_path)
    print_results(results)

    # Guard against clobbering good data: some logs are incomplete re-runs that
    # contain zero metrics. Don't overwrite an existing scores.json (which may
    # hold valid results from a prior complete run) with an all-null result.
    json_path = Path(log_path).with_suffix(".scores.json")
    has_values = any(isinstance(v, (int, float)) for v in results.values())
    if not has_values and json_path.exists():
        print(f"\nNo metrics found in log; keeping existing {json_path} unchanged.")
        return

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON output saved to: {json_path}")


if __name__ == "__main__":
    main()
