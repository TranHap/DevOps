"""Step 2b - Train/test split shared by ALL models (supervised + unsupervised), so results are comparable.

Usage: python src/split.py --dataset HDFS|BGL [--seeds 10]

Like the paper, sessions are shuffled then split (HDFS 70/30, BGL 80/20). Deviation from the paper: we
stratify on the label so train and test keep the same anomaly rate (matters on BGL, ~143 test sessions).
seed 0 = main split (replication); seeds 1..9 = resamples for the Scott-Knott ranking (extension).

Output: data/splits/<DS>_seed<k>.npz with train_idx, test_idx (row positions in <DS>_sessions.pkl)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_RATIO = {"HDFS": 0.7, "BGL": 0.8}


def split(labels, train_ratio, seed, stratify=True):
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(idx, train_size=train_ratio, shuffle=True, random_state=seed,
                                           stratify=labels if stratify else None)
    return np.sort(train_idx), np.sort(test_idx)


def load_split(ds, seed=0):
    d = np.load(os.path.join(ROOT, "data", "splits", f"{ds}_seed{seed}.npz"))
    return d["train_idx"], d["test_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(TRAIN_RATIO))
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    ds = args.dataset

    sessions = pd.read_pickle(os.path.join(ROOT, "data", "sessions", f"{ds}_sessions.pkl"))
    y = sessions["label"].to_numpy()
    out_dir = os.path.join(ROOT, "data", "splits")
    os.makedirs(out_dir, exist_ok=True)
    for seed in range(args.seeds):
        tr, te = split(y, TRAIN_RATIO[ds], seed)
        np.savez(os.path.join(out_dir, f"{ds}_seed{seed}.npz"), train_idx=tr, test_idx=te)
        if seed == 0:
            print(f"{ds} seed 0: train {len(tr):,} ({y[tr].sum():,} anomalous, {y[tr].mean():.2%}) | "
                  f"test {len(te):,} ({y[te].sum():,} anomalous, {y[te].mean():.2%})")
    print(f"  saved {args.seeds} splits to {out_dir}")


if __name__ == "__main__":
    main()
