"""Extension E3 - explain the best model (Random Forest + MCV on HDFS) with SHAP.

Usage: python src/explain_rf.py [--seed 0] [--select vif|none] [--n 1000]

We explain the RF trained on the corr/VIF-selected MCV (28 features on seed 0) by default: SHAP splits credit
between correlated features, which makes rankings on the full 46 features misleading, and RQ1 showed the
selected model performs the same in practice (F1 0.997 vs 0.998). Features = log templates, so the
explanation reads as "which events make a block look anomalous".

SHAP (TreeExplainer, exact for trees) on a stratified test sample: n anomalous + n normal sessions.
Outputs (results/ext/shap/): global importance table + bar plot, beeswarm plot, one local explanation.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import ROOT, evaluate  # noqa: E402
from feature_selection import select_features  # noqa: E402


def save_current(path, size=(11, 6)):
    """shap plots draw on their own figure; widen it so long template labels and the axis fit."""
    fig = plt.gcf()
    fig.set_size_inches(*size)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def short(t, n=45):
    return t if len(t) <= n else t[:n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", default="vif", choices=["vif", "none"])
    ap.add_argument("--n", type=int, default=1000, help="sessions per class to explain")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", "ext", "shap")
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(os.path.join(ROOT, "data", "features", f"HDFS_MCV_seed{args.seed}.npz"))
    x_train, y_train, x_test, y_test, names = d["x_train"], d["y_train"], d["x_test"], d["y_test"], d["feature_names"]
    if args.select == "vif":
        cols, _ = select_features(x_train, names)
        x_train, x_test, names = x_train[:, cols], x_test[:, cols], names[cols]

    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1).fit(x_train, y_train)
    perf = evaluate(y_test, clf.predict_proba(x_test)[:, 1])
    print(f"RF ({args.select}, {len(names)} features): {perf}")

    tpl = pd.read_csv(os.path.join(ROOT, "data", "parsed", "HDFS", "HDFS_templates.csv")).set_index("EventId")["EventTemplate"]
    label = {n: (f"{n}: {short(tpl[int(n[1:])])}" if n.startswith("E") else n) for n in names}

    rng = np.random.default_rng(args.seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y_test == c), args.n, replace=False) for c in (1, 0)])
    X = pd.DataFrame(x_test[idx], columns=[label[n] for n in names])
    explainer = shap.TreeExplainer(clf)
    sv = explainer(X)[:, :, 1]          # SHAP values for the "anomaly" class
    S = sv.values
    ys = y_test[idx]

    imp = pd.DataFrame({
        "feature": names, "template": [tpl[int(n[1:])] if n.startswith("E") else n for n in names],
        "mean_abs_shap": np.abs(S).mean(0),
        "mean_shap_anomalous": S[ys == 1].mean(0), "mean_shap_normal": S[ys == 0].mean(0),
        "mean_count_anomalous": x_test[idx][ys == 1].mean(0), "mean_count_normal": x_test[idx][ys == 0].mean(0),
        "rf_impurity_importance": clf.feature_importances_,
    }).sort_values("mean_abs_shap", ascending=False)
    imp.round(4).to_csv(os.path.join(out_dir, f"shap_importance_{args.select}_seed{args.seed}.csv"), index=False)
    print(imp.head(10).round(3).to_string(index=False))

    shap.plots.bar(sv, max_display=12, show=False)
    plt.title(f"HDFS RF+MCV ({args.select}): mean |SHAP| (anomaly class)")
    save_current(os.path.join(out_dir, f"shap_bar_{args.select}_seed{args.seed}.png"))

    shap.plots.beeswarm(sv, max_display=12, show=False)
    save_current(os.path.join(out_dir, f"shap_beeswarm_{args.select}_seed{args.seed}.png"))

    # Local explanation: the correctly detected anomaly with the highest predicted probability
    p = clf.predict_proba(X.values)[:, 1]
    i = int(np.argmax(np.where(ys == 1, p, -1)))
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    save_current(os.path.join(out_dir, f"shap_waterfall_{args.select}_seed{args.seed}.png"))

    with open(os.path.join(out_dir, f"shap_meta_{args.select}_seed{args.seed}.json"), "w") as f:
        json.dump({"seed": args.seed, "select": args.select, "n_features": len(names),
                   "explained_sessions": {"anomalous": args.n, "normal": args.n}, "performance": perf,
                   "base_value": float(np.ravel(sv.base_values)[0])}, f, indent=2)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
