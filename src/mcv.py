"""Step 2c - Message Count Vector (raw counts; no TF-IDF / scaling here, so any later transform stays possible).

Usage: python src/mcv.py --dataset HDFS|BGL [--seed 0]

The vocabulary (= feature columns) is built from the TRAIN sessions only. Events never seen in train are
counted in one extra "UNK" column, so nothing about the test set leaks into the feature space.

Output: data/features/<DS>_MCV_seed<k>.npz with x_train, y_train, x_test, y_test, feature_names
        (same keys as Wu et al.'s replication package, so their model scripts can load it)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split import ROOT, load_split  # noqa: E402

UNK = "UNK"


def build_vocab(train_events):
    return np.unique(np.concatenate(list(train_events)))


def count_matrix(events, vocab):
    """events: iterable of int arrays -> (n_sessions, len(vocab)+1) int32 counts; last column = UNK."""
    col = {e: i for i, e in enumerate(vocab)}
    unk = len(vocab)
    X = np.zeros((len(events), len(vocab) + 1), dtype=np.int32)
    for r, seq in enumerate(events):
        ids, cnt = np.unique(seq, return_counts=True)
        for e, c in zip(ids, cnt):
            X[r, col.get(e, unk)] += c
    return X


def build_mcv(sessions, train_idx, test_idx):
    ev = sessions["events"].to_numpy()
    vocab = build_vocab(ev[train_idx])
    names = np.array([f"E{e}" for e in vocab] + [UNK])
    return count_matrix(ev[train_idx], vocab), count_matrix(ev[test_idx], vocab), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["HDFS", "BGL"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset

    sessions = pd.read_pickle(os.path.join(ROOT, "data", "sessions", f"{ds}_sessions.pkl"))
    tr, te = load_split(ds, args.seed)
    x_train, x_test, names = build_mcv(sessions, tr, te)
    y = sessions["label"].to_numpy()

    out = os.path.join(ROOT, "data", "features", f"{ds}_MCV_seed{args.seed}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, x_train=x_train, y_train=y[tr], x_test=x_test, y_test=y[te], feature_names=names)

    const = (x_train.min(0) == x_train.max(0)).sum()
    print(f"{ds} MCV seed {args.seed}: x_train {x_train.shape}, x_test {x_test.shape}; "
          f"{len(names) - 1} train templates + UNK; {const} columns constant in train; "
          f"test sessions with UNK events: {(x_test[:, -1] > 0).sum()}")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
