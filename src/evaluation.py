"""Shared metrics + result saving for all models (precision, recall, F1, AUC on the test sessions)."""
import json
import os

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(y_true, scores, threshold=0.5):
    """scores = anomaly score / probability per session; prediction = score > threshold."""
    y_pred = (np.asarray(scores) > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
            "auc": round(float(roc_auc_score(y_true, scores)), 4)}


def save_result(name, result, subdir="rq1"):
    out = os.path.join(ROOT, "results", subdir, f"{name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out
