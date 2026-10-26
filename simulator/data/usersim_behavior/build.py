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

"""
Build the usersim-behavior RL parquets from ABCD.

One row = one ABCD scenario plus either a behavior directive or nothing (a control row).
The rollout starts from an empty conversation and the sim plays the customer for
``turn_budget`` turns against a frozen support agent primed with that subflow's policy.

Row shape follows the MIMESIS convention measured off the existing parquets in
``sim_rl_data/``: three columns, a stub ``prompt``, and a dict ``extra_info`` carrying
everything the agent loop reads. Every list/dict payload inside ``extra_info`` is
JSON-encoded because pyarrow needs one uniform struct schema across all rows and
per-behavior kwargs are heterogeneous -- the same reason IFBench stores
``instruction_id_list``/``kwargs`` as strings (``agents/instruct/ifbench_agent.py:100``).

Usage:
    python -m data.usersim_behavior.build --out-dir /genai/.../sim_rl_data
    python -m data.usersim_behavior.build --dry-run           # stats only, writes nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from random import Random
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.usersim_behavior.abcd_data import is_covered  # noqa: E402
from agents.usersim_behavior.behaviors import BEHAVIOR_DICT  # noqa: E402

DATA_SOURCE = "usersim-behavior"
# Held out of train entirely: val and the offline eval set draw only from this flow, so a
# rising val curve means the behavior transferred to an unseen procedure, not just an
# unseen conversation.
VAL_FLOW = "subscription_inquiry"
# Generated sim turns per rollout, on top of the replayed identity check
# (abcd_data.verification_prefix). At 4 the agent was still finishing verification when the
# budget ran out: no eligibility decision was ever reached, so five of eight behaviors
# scored exhibit=0.0 for structural reasons -- identical across a GRPO group, hence zero
# advantage and no gradient. 8 matches sim_arena (agents/sim_arena/agent_doc.py:135), the
# closest structural sibling in the mix; sotopia runs ~8 policy turns and coser ~10.
TURN_BUDGET = 8


def _scenario_pool(convos: list[dict], *, flow_filter, seen_ids: set[int] | None = None) -> list[dict]:
    out = []
    for c in convos:
        sc = c["scenario"]
        if not is_covered(sc["subflow"]) or not flow_filter(sc["flow"]):
            continue
        if seen_ids is not None and c["convo_id"] in seen_ids:
            continue
        out.append(c)
    return out


def _make_row(
    *, index: int, convo: dict, behavior_id: str | None, rng: Random
) -> dict[str, Any] | None:
    """One parquet row. Returns None if the behavior turns out not to apply."""
    sc = convo["scenario"]
    flow, subflow = sc["flow"], sc["subflow"]

    if behavior_id is None:
        behavior_ids, kwargs_list, distractors = [], [], []
    else:
        behavior = BEHAVIOR_DICT[behavior_id]
        if not behavior.applicable(flow, subflow, sc):
            return None
        behavior_ids = [behavior_id]
        kwargs_list = [behavior.derive_kwargs(flow, subflow, sc, rng)]
        distractors = behavior.confusable_with

    return {
        "prompt": [{"role": "user", "content": "x"}],  # stub; rl_dataset.py:297 requires the column
        "data_source": DATA_SOURCE,
        "extra_info": {
            "index": index,
            "behavior_ids": json.dumps(behavior_ids),
            "kwargs": json.dumps(kwargs_list),
            # Read only by run_eval.py's confusion matrix; the training loop ignores it.
            "distractor_ids": json.dumps(distractors),
            "convo_id": int(convo["convo_id"]),
            "flow": flow,
            "subflow": subflow,
            "scenario": json.dumps(sc),
            "turn_budget": TURN_BUDGET,
        },
    }


def build_split(
    convos: list[dict],
    *,
    per_behavior: int,
    control_frac: float,
    seed: int,
    start_index: int = 0,
) -> tuple[list[dict], Counter]:
    rng = Random(seed)
    rows: list[dict] = []
    stats: Counter = Counter()
    index = start_index

    for name, behavior in BEHAVIOR_DICT.items():
        pool = [c for c in convos if behavior.applicable(c["scenario"]["flow"], c["scenario"]["subflow"], c["scenario"])]
        if not pool:
            stats[f"EMPTY POOL {name}"] += 1
            continue
        # Sample without replacement while the pool allows it, so a behavior never trains
        # on the same scenario twice before it has seen every other one.
        picks = rng.sample(pool, per_behavior) if len(pool) >= per_behavior else [rng.choice(pool) for _ in range(per_behavior)]
        if len(pool) < per_behavior:
            stats[f"OVERSAMPLED {name} (pool={len(pool)})"] += 1
        for convo in picks:
            row = _make_row(index=index, convo=convo, behavior_id=name, rng=rng)
            if row is None:
                continue
            rows.append(row)
            stats[name] += 1
            index += 1

    n_control = round(len(rows) * control_frac / (1 - control_frac))
    for convo in (rng.choice(convos) for _ in range(n_control)):
        row = _make_row(index=index, convo=convo, behavior_id=None, rng=rng)
        rows.append(row)
        stats["__control__"] += 1
        index += 1

    rng.shuffle(rows)
    # index must stay distinct per prompt -- verl groups the rollout.n samples of one
    # prompt by it (agent_loop.py:897), so a duplicate silently merges two prompts.
    for i, row in enumerate(rows):
        row["extra_info"]["index"] = start_index + i
    return rows, stats


def _report(name: str, rows: list[dict], stats: Counter) -> None:
    print(f"\n=== {name}: {len(rows)} rows ===")
    warnings = [k for k in stats if k.startswith(("EMPTY POOL", "OVERSAMPLED"))]
    for behavior in list(BEHAVIOR_DICT) + ["__control__"]:
        n = stats.get(behavior, 0)
        print(f"  {behavior:<34} {n:>5}  ({100 * n / max(len(rows), 1):.1f}%)")
    for w in warnings:
        print(f"  WARNING: {w}")
    flows = Counter(r["extra_info"]["flow"] for r in rows)
    print(f"  flows: {dict(flows)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abcd", default="abcd_v1.1.json")
    ap.add_argument("--out-dir", default="${MIMESIS_DATA_DIR}/sim_rl_data")
    # 13 x 63 + 20% control ~= 1024, matching the house standard across the other 22 tasks.
    ap.add_argument("--train-per-behavior", type=int, default=63)
    ap.add_argument("--val-per-behavior", type=int, default=15)
    ap.add_argument("--control-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.abcd) as f:
        abcd = json.load(f)

    # Train: every covered flow except the held-out one. Val: only the held-out flow,
    # pooling ABCD's own train+dev so the smaller flow still fills the split.
    train_convos = _scenario_pool(abcd["train"], flow_filter=lambda f: f != VAL_FLOW)
    val_convos = _scenario_pool(abcd["train"] + abcd["dev"], flow_filter=lambda f: f == VAL_FLOW)
    print(f"scenario pools -- train: {len(train_convos)}  val ({VAL_FLOW}): {len(val_convos)}")

    train_rows, train_stats = build_split(
        train_convos, per_behavior=args.train_per_behavior, control_frac=args.control_frac, seed=args.seed
    )
    val_rows, val_stats = build_split(
        val_convos, per_behavior=args.val_per_behavior, control_frac=args.control_frac, seed=args.seed + 1
    )
    _report("train", train_rows, train_stats)
    _report("val", val_rows, val_stats)

    # No scenario may appear in both splits. The flow holdout should guarantee it; assert
    # rather than trust, because a silent leak here inflates every val curve.
    overlap = {r["extra_info"]["convo_id"] for r in train_rows} & {r["extra_info"]["convo_id"] for r in val_rows}
    assert not overlap, f"train/val convo_id overlap: {sorted(overlap)[:10]}"

    if args.dry_run:
        print("\n--dry-run: nothing written")
        print("\nsample row:")
        print(json.dumps(train_rows[0], indent=2)[:1500])
        return 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Explicit schema, matched field-for-field to the existing sim_rl_data parquets.
    # pd.DataFrame.to_parquet infers `large_string` for object columns on this pyarrow
    # version, and struct field order follows dict insertion order -- both differ from the
    # house convention, and `datasets.concatenate_datasets` refuses to align them:
    #   ValueError: the key data_source ... has unexpected type - Value('large_string')
    # It only bites under TASK=all (a usersim-only run has nothing to concatenate), which
    # is exactly the run you want, so pin the types rather than let them be inferred.
    schema = pa.schema(
        [
            pa.field("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
            pa.field("data_source", pa.string()),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        ("index", pa.int64()),
                        ("behavior_ids", pa.string()),
                        ("kwargs", pa.string()),
                        ("distractor_ids", pa.string()),
                        ("convo_id", pa.int64()),
                        ("flow", pa.string()),
                        ("subflow", pa.string()),
                        ("scenario", pa.string()),
                        ("turn_budget", pa.int64()),
                    ]
                ),
            ),
        ]
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("usersim_behavior_rl_train", train_rows), ("usersim_behavior_val", val_rows)):
        path = os.path.join(args.out_dir, f"{name}.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        written = pq.ParquetFile(path).schema_arrow
        assert written.field("data_source").type == pa.string(), written
        print(f"wrote {len(rows):>5} rows -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
