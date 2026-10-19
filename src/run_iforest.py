"""Extension E1 - classical UNSUPERVISED model: Isolation Forest on MCV (labels never used for training).

Usage: python src/run_iforest.py [--dataset HDFS] [--seed 0]

Trained on all train sessions (normal + anomalous, labels ignored), as an unsupervised detector would be.
Anomaly score = -score_samples (higher = more anomalous); predicted anomalous if score is above the
contamination quantile. contamination = 0.03 is loglizer's default for Isolation Forest, fixed a priori
(not tuned on test). It happens to be close to HDFS's true anomaly rate (2.9%), which favours the model:
mention it as a threat.
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import ROOT, evaluate, save_result  # noqa: E402

CONTAMINATION = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HDFS", choices=["HDFS", "BGL"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(os.path.join(ROOT, "data", "features", f"{args.dataset}_MCV_seed{args.seed}.npz"))
    x_train, x_test, y_test = d["x_train"], d["x_test"], d["y_test"]

    t0 = time.time()
    clf = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=args.seed, n_jobs=-1)
    clf.fit(x_train)
    train_time = time.time() - t0
    scores = -clf.score_samples(x_test)
    threshold = -clf.offset_   # sklearn: predict() == -1  <=>  score_samples < offset_

    res = {"model": "IsolationForest", "representation": "MCV", "dataset": args.dataset, "seed": args.seed,
           "contamination": CONTAMINATION, "train_time_s": round(train_time, 1),
           **evaluate(y_test, scores, threshold=threshold)}
    out = save_result(f"IF_MCV_{args.dataset}_seed{args.seed}", res, subdir="ext")
    print(f"IF {args.dataset} seed {args.seed}: P={res['precision']} R={res['recall']} F1={res['f1']} "
          f"AUC={res['auc']} ({train_time:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
