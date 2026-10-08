#!/usr/bin/env python3

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
