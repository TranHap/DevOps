"""Session-level evaluation, metrics.csv, comparison with the paper.

Positive class = anomaly, threshold 0.5 on the session score.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

METRIC_COLS = ["dataset", "model", "representation", "feature_selection", "split", "seed", "n_features",
               "precision", "recall", "f1", "auc", "tp", "fp", "tn", "fn", "train_time_s",
               "n_train", "n_test", "n_test_anomalous"]
KEY = ["dataset", "model", "representation", "feature_selection", "split", "seed"]

# Paper Tables 2 and 3 (P, R, F1)
PAPER = {
    ("hdfs", "RF", "MCV"): (0.998, 1.000, 0.999),
    ("hdfs", "LSTM", "BERT"): (0.998, 0.923, 0.959),
    ("bgl", "RF", "MCV"): (0.830, 0.963, 0.891),
    ("bgl", "LSTM", "BERT"): (0.871, 1.000, 0.931),
}


def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")
    return dict(precision=float(p), recall=float(r), f1=float(f1), auc=float(auc),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def upsert_metrics(path, row):
    row = {k: row.get(k) for k in METRIC_COLS}
    if path.exists():
        df = pd.read_csv(path, keep_default_na=False)
        mask = np.ones(len(df), dtype=bool)
        for k in KEY:
            mask &= df[k].astype(str).values == str(row[k])
        df = pd.concat([df[~mask], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    order = {"hdfs": 0, "bgl": 1}
    df = df.sort_values(by=["dataset", "split", "model", "feature_selection", "seed"],
                        key=lambda s: s.map(order).fillna(9) if s.name == "dataset" else s).reset_index(drop=True)
    df.to_csv(path, index=False, columns=METRIC_COLS)
    return df


def comparison_with_paper(metrics_df, seed=42, split="random"):
    rows = []
    m = metrics_df[(metrics_df.seed.astype(int) == seed) & (metrics_df.split == split)]
    for _, r in m.iterrows():
        key = (r.dataset, r.model, r.representation)
        if key not in PAPER:
            continue
        pp, pr, pf = PAPER[key]
        rows.append(dict(dataset=r.dataset, model=r.model, representation=r.representation,
                         feature_selection=r.feature_selection, seed=int(r.seed),
                         paper_precision=pp, ours_precision=round(r.precision, 3), delta_precision=round(r.precision - pp, 3),
                         paper_recall=pr, ours_recall=round(r.recall, 3), delta_recall=round(r.recall - pr, 3),
                         paper_f1=pf, ours_f1=round(r.f1, 3), delta_f1=round(r.f1 - pf, 3), ours_auc=round(r.auc, 3)))
    return pd.DataFrame(rows)
