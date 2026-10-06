#!/usr/bin/env python3
"""Carve the RL validation split out of the TRAINING set, not the test set.

    python examples/data_preprocess/make_rl_val_split.py

Why this exists
---------------
Upstream's train.sh points `data.val_files` at data/alltest_multiturn/test.parquet,
so in-training validation runs against the held-out evaluation set. Combined with
verl picking a best checkpoint on that metric, the model would be selected on the
very data the paper's generalization claim rests on -- IntentionGym, TelepathyGym
and SearchGym are "entirely held-out evaluation environments" (Section 4), and
choosing a checkpoint by their scores quietly contaminates them.

Appendix B says what the paper actually did: "we further reserve 5% of data as
validation set and pick the best saved checkpoint for final result evaluation for
each setting" -- 5% of the RL *training* data. This script reproduces that.

A useful side effect: the training set contains only the five held-in gyms, so a
split of it never touches SearchGym and needs no Serper API key. The test set can
then stay untouched for the final eval/ run.

Stratified per data_source with a fixed seed, so every run in the sweep validates
on exactly the same rows and the curves are comparable.
"""

import argparse
import pathlib

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=REPO / "data/alltrain_multiturn/train.parquet")
    ap.add_argument("--outdir", default=REPO / "data/rl_split")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Stratify so a gym with 200 rows is not dropped from validation entirely by
    # an unlucky uniform sample; ceil keeps at least one row per gym.
    val_parts = []
    for source, group in df.groupby("data_source", sort=True):
        n = max(1, round(len(group) * args.val_frac))
        val_parts.append(group.sample(n=n, random_state=args.seed))
    val = pd.concat(val_parts).sort_index()
    train = df.drop(index=val.index)

    assert len(train) + len(val) == len(df), "split lost or duplicated rows"
    assert not set(train.index) & set(val.index), "train/val overlap"

    train.to_parquet(outdir / "train.parquet", index=False)
    val.to_parquet(outdir / "val.parquet", index=False)

    print(f"source : {args.train}  ({len(df)} rows)")
    print(f"train  : {outdir / 'train.parquet'}  ({len(train)} rows)")
    print(f"val    : {outdir / 'val.parquet'}  ({len(val)} rows, {len(val) / len(df):.1%})")
    print("\nper-gym:")
    print(f"  {'data_source':24s} {'train':>6s} {'val':>5s}")
    for source in sorted(df["data_source"].unique()):
        print(f"  {source:24s} {(train['data_source'] == source).sum():6d}"
              f" {(val['data_source'] == source).sum():5d}")


if __name__ == "__main__":
    main()
