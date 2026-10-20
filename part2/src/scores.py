"""Score definitions and session-level metrics (fairness rules F5 and F6).

Every model outputs one continuous anomaly score per test session; its binary prediction is a
threshold on that same score, so F1 (binary) and ROC-AUC (score) always describe the same model.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

# model id -> how the score is defined and turned into a decision
SCORE_DEF = {
    "rf":           dict(label="RF + MCV", score="predict_proba[:, 1]", rule=">=", threshold="0.5"),
    "lstm":         dict(label="LSTM + BERT", score="max window P(anomaly) (Part 1 merging)", rule=">=", threshold="0.5"),
    "pca":          dict(label="PCA", score="SPE (squared prediction error)", rule=">", threshold="Q-statistic"),
    "loggpt":       dict(label="LogGPT", score="max rank of the actual next key", rule=">", threshold="K"),
    # not ranked
    "loggpt_norl":  dict(label="LogGPT w/o RL", score="max rank of the actual next key", rule=">", threshold="K"),
    "pca_normal5k": dict(label="PCA (5,000 normal)", score="SPE (squared prediction error)", rule=">", threshold="Q-statistic"),
}
RANKED = ["rf", "lstm", "pca", "loggpt"]
LABELS_USED = {"rf": "full", "lstm": "full", "pca": "none", "loggpt": "normal-only filter",
               "loggpt_norl": "normal-only filter", "pca_normal5k": "normal-only filter"}


def decide(model, scores, threshold):
    rule = SCORE_DEF[model]["rule"]
    scores = np.asarray(scores)
    return (scores >= threshold if rule == ">=" else scores > threshold).astype(np.int8)


def session_metrics(y_true, scores, pred):
    y_true = np.asarray(y_true).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")
    return dict(precision=float(p), recall=float(r), f1=float(f1), auc=float(auc),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
