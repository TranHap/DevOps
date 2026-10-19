"""RQ1 replication - classical supervised model: Random Forest on MCV.

Usage: python src/run_rf.py --dataset HDFS|BGL [--seed 0] [--select none|vif]
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import ROOT, evaluate, save_result  # noqa: E402
from feature_selection import select_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["HDFS", "BGL"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", default="none", choices=["none", "vif"])
    args = ap.parse_args()

    d = np.load(os.path.join(ROOT, "data", "features", f"{args.dataset}_MCV_seed{args.seed}.npz"))
    x_train, y_train, x_test, y_test, names = d["x_train"], d["y_train"], d["x_test"], d["y_test"], d["feature_names"]

    report = None
    if args.select == "vif":
        cols, report = select_features(x_train, names)
        x_train, x_test = x_train[:, cols], x_test[:, cols]

    t0 = time.time()
    # sklearn defaults, like the paper's replication package
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    clf.fit(x_train, y_train)
    train_time = time.time() - t0
    scores = clf.predict_proba(x_test)[:, 1]

    res = {"model": "RF", "representation": "MCV", "dataset": args.dataset, "seed": args.seed,
           "select": args.select, "n_features": int(x_train.shape[1]), "train_time_s": round(train_time, 1),
           **evaluate(y_test, scores),
           "feature_importance": dict(sorted(zip(names[cols].tolist() if report else names.tolist(),
                                                 clf.feature_importances_.round(4).tolist()),
                                             key=lambda kv: -kv[1])),
           "selection_report": report}
    out = save_result(f"RF_MCV_{args.dataset}_{args.select}_seed{args.seed}", res)
    print(f"RF {args.dataset} select={args.select}: features={res['n_features']} "
          f"P={res['precision']} R={res['recall']} F1={res['f1']} AUC={res['auc']} ({train_time:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
