"""Summarize RQ1 replication: compare with the paper, and test with vs without correlation/VIF selection.

Usage: python src/summarize_rq1.py
Outputs: results/rq1_all_runs.csv, results/rq1_summary.csv (+ printed tables)

With/without selection is a paired comparison (same split per seed) -> Wilcoxon signed-rank test over seeds,
plus Cliff's delta as effect size.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_F1 = {("RF", "HDFS"): 0.999, ("RF", "BGL"): 0.891, ("LSTM", "HDFS"): 0.955, ("LSTM", "BGL"): 0.871}
METRICS = ["precision", "recall", "f1", "auc"]


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def magnitude(d):
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def main():
    rows = []
    # glob.escape: ROOT contains "[DevOps]", which glob would read as a character class
    for f in glob.glob(os.path.join(glob.escape(ROOT), "results", "rq1", "*.json")):
        r = json.load(open(f, encoding="utf-8"))
        rows.append({k: r[k] for k in ["model", "representation", "dataset", "seed", "select", "n_features",
                                       "train_time_s", *METRICS]})
    df = pd.DataFrame(rows).sort_values(["model", "dataset", "select", "seed"])
    df.to_csv(os.path.join(ROOT, "results", "rq1_all_runs.csv"), index=False)

    print("== Seed 0 (main split) vs paper ==")
    s0 = df[df.seed == 0].copy()
    s0["paper_f1"] = [PAPER_F1[(m, d)] for m, d in zip(s0.model, s0.dataset)]
    print(s0[["model", "dataset", "select", "n_features", *METRICS, "paper_f1"]].to_string(index=False))

    print("\n== All seeds: mean +- std, and paired test none vs vif ==")
    out = []
    for (m, d), g in df.groupby(["model", "dataset"]):
        p = g.pivot(index="seed", columns="select", values=METRICS + ["n_features"]).dropna()
        row = {"model": m, "dataset": d, "n_seeds": len(p), "paper_f1": PAPER_F1[(m, d)],
               "n_features_none": p["n_features"]["none"].mean(), "n_features_vif": p["n_features"]["vif"].mean()}
        for met in METRICS:
            a, b = p[met]["none"].to_numpy(), p[met]["vif"].to_numpy()
            row[f"{met}_none"] = f"{a.mean():.3f} +- {a.std():.3f}"
            row[f"{met}_vif"] = f"{b.mean():.3f} +- {b.std():.3f}"
            if met in ("f1", "auc") and len(p) >= 5 and np.any(a != b):
                row[f"{met}_wilcoxon_p"] = round(float(wilcoxon(a, b).pvalue), 4)
                dlt = cliffs_delta(a, b)
                row[f"{met}_cliffs_delta"] = f"{dlt:+.2f} ({magnitude(dlt)})"
        out.append(row)
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(ROOT, "results", "rq1_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summ.T.to_string(header=False))


if __name__ == "__main__":
    main()
