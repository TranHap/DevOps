"""Build report-ready tables and figures from the result files (no number is typed by hand).

Inputs : results/metrics.csv, results/parsing_eval.csv, results/bert_rank.json,
         results/feature_selection_*_info.json, data/predictions/*/*.json, data/sessions/*/sessions_meta.json,
         data/parsed/*/parse_meta.json, data/splits/*.json, results/discrepancy_notes.md (hand-written analysis)
Outputs: results/comparison_with_paper.csv, results/seed_summary.csv, results/figures/*.png, results/RESULTS.md

Usage: python -m src.report
"""
import json

import numpy as np
import pandas as pd

from .common import PART1_ROOT, load_config, read_json
from .evaluate import PAPER, comparison_with_paper

RES = PART1_ROOT / "results"
FIG = RES / "figures"
DATA = PART1_ROOT / "data"
RUN_ORDER = [("RF", "without"), ("RF", "with"), ("LSTM", "n/a")]
LABEL = {("RF", "without"): "RF + MCV", ("RF", "with"): "RF + MCV + step 3a", ("LSTM", "n/a"): "LSTM + BERT"}


def _md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)) and not pd.isna(v):
                fmt = 1 if ("time" in c.lower() or "mean" in c.lower()) else (2 if "%" in c else floatfmt)
                cells.append(f"{v:.{fmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_metrics():
    m = pd.read_csv(RES / "metrics.csv", keep_default_na=False)
    for c in ["precision", "recall", "f1", "auc", "train_time_s"]:
        m[c] = m[c].astype(float)
    m["seed"] = m["seed"].astype(int)
    return m


def main_table(m):
    rows = []
    for ds in ("hdfs", "bgl"):
        for model, fs in RUN_ORDER:
            r = m[(m.dataset == ds) & (m.model == model) & (m.feature_selection == fs) & (m.seed == 42) & (m.split == "random")]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"Dataset": ds.upper(), "Combination": LABEL[(model, fs)], "#features": int(r.n_features),
                         "P": r.precision, "R": r.recall, "F1": r.f1, "AUC": r.auc,
                         "TP": int(r.tp), "FP": int(r.fp), "TN": int(r.tn), "FN": int(r.fn),
                         "Train time (s)": round(r.train_time_s, 1)})
    return pd.DataFrame(rows)


def seed_summary(m):
    rows = []
    mm = m[m.split == "random"]
    for ds in ("hdfs", "bgl"):
        for model, fs in RUN_ORDER:
            r = mm[(mm.dataset == ds) & (mm.model == model) & (mm.feature_selection == fs)]
            if len(r) < 2:
                continue
            row = {"Dataset": ds.upper(), "Combination": LABEL[(model, fs)], "n_seeds": len(r),
                   "seeds": " ".join(map(str, sorted(r.seed)))}
            for k in ("precision", "recall", "f1", "auc"):
                row[k] = f"{r[k].mean():.3f} ± {r[k].std(ddof=1):.4f}"
            row["min F1"], row["max F1"] = round(r.f1.min(), 3), round(r.f1.max(), 3)
            row["mean #features"] = round(r.n_features.astype(float).mean(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def fig_f1_vs_paper(comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = comp[comp.feature_selection.isin(["without", "n/a"])]
    labels = [f"{r.dataset.upper()}\n{r.model}+{r.representation}" for _, r in c.iterrows()]
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, c.paper_f1, 0.4, label="Paper (Wu et al. 2023)", color="#9aa5b1")
    ax.bar(x + 0.2, c.ours_f1, 0.4, label="This replication (seed 42)", color="#2f6db5")
    for i, (p, o) in enumerate(zip(c.paper_f1, c.ours_f1)):
        ax.text(i - 0.2, p + 0.005, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, o + 0.005, f"{o:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("F1 (test, session level)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "f1_vs_paper.png", dpi=200)
    plt.close(fig)


def fig_lstm_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    for ds in ("hdfs", "bgl"):
        p = DATA / "predictions" / ds / "lstm_random_seed42.json"
        if not p.exists():
            continue
        h = pd.DataFrame(read_json(p)["history"])
        best = read_json(p)["best_epoch"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(h.epoch, h.val_f1, "-o", ms=3, label="validation F1")
        ax.plot(h.epoch, h.val_precision, ":", label="validation precision")
        ax.plot(h.epoch, h.loss, "--", label="training loss")
        curve = RES / f"lstm_test_curve_{ds}_random_seed42.json"
        if curve.exists():
            ht = pd.DataFrame(read_json(curve)["history"])
            ax.plot(ht.epoch, ht.test_f1, "-", color="grey", alpha=0.6, label="test F1 (diagnostic only)")
        ax.axvline(best, color="red", lw=0.8, ls="--", label=f"selected epoch ({best})")
        ax.set_xlabel("epoch")
        ax.set_title(f"LSTM + BERT on {ds.upper()}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f"lstm_training_{ds}.png", dpi=200)
        plt.close(fig)
        made.append(f"lstm_training_{ds}.png")
    return made


def data_table():
    rows = []
    for ds in ("hdfs", "bgl"):
        pm = read_json(DATA / "parsed" / ds / "parse_meta.json")
        sm = read_json(DATA / "sessions" / ds / "sessions_meta.json")
        sp = read_json(DATA / "splits" / f"{ds}_seed42.json")
        rows.append({"Dataset": ds.upper(), "Log lines": pm["n_parsed_lines"], "Templates": pm["n_templates"],
                     "Sessions": sm["n_sessions"], "Anomalous sessions": sm["n_anomalous"],
                     "Mean events/session": round(sm["mean_len"], 1), "Train": sp["n_train"], "Test": sp["n_test"],
                     "Train anomaly %": round(100 * sp["anomaly_ratio_train"], 2),
                     "Test anomaly %": round(100 * sp["anomaly_ratio_test"], 2)})
    return pd.DataFrame(rows)


def fs_table():
    rows = []
    for ds in ("hdfs", "bgl"):
        p = RES / f"feature_selection_{ds}_info.json"
        if not p.exists():
            continue
        d = read_json(p)
        rows.append({"Dataset": ds.upper(), "MCV features (train vocab)": d["n_in"], "Constant": d["n_constant"],
                     "Correlation clusters": d["n_corr_clusters"], "Removed (correlation)": d["n_correlation"],
                     "Removed (VIF > 5)": d["n_vif"], "Kept": d["n_kept"],
                     "Max VIF kept": round(max(d["final_vif"].values()), 2)})
    return pd.DataFrame(rows)


def main():
    m = load_metrics()
    comp = comparison_with_paper(m)
    rank = {("RF", "without"): 0, ("RF", "with"): 1, ("LSTM", "n/a"): 2}
    comp = comp.assign(_d=comp.dataset.map({"hdfs": 0, "bgl": 1}),
                       _m=[rank[(a, b)] for a, b in zip(comp.model, comp.feature_selection)])
    comp = comp.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"])
    comp.to_csv(RES / "comparison_with_paper.csv", index=False)
    ss = seed_summary(m)
    if len(ss):
        ss.to_csv(RES / "seed_summary.csv", index=False)
    fig_f1_vs_paper(comp)
    figs = ["f1_vs_paper.png"] + fig_lstm_curves()
    figs += [p.name for p in sorted(FIG.glob("dendrogram_*.png")) if "seed" not in p.name]

    parsing = pd.read_csv(RES / "parsing_eval.csv")
    parsing = parsing[["dataset", "config", "n_templates_ground_truth", "n_templates_parsed", "GA", "PA", "PA_relaxed"]]
    ranks = read_json(RES / "bert_rank.json")
    rank_df = pd.DataFrame([{"Dataset": k.upper(), **{kk: v[kk] for kk in ("n_templates", "embedding_dim", "rank",
                                                                            "rank_centered", "n_dims_linearly_dependent")}}
                            for k, v in ranks.items() if k in ("hdfs", "bgl")])
    sens = m[m.split != "random"]
    sens_df = pd.DataFrame([{"Dataset": r.dataset.upper(), "Split": r.split, "Combination": LABEL[(r.model, r.feature_selection)],
                             "P": r.precision, "R": r.recall, "F1": r.f1, "AUC": r.auc} for _, r in sens.iterrows()])
    win_rows = []
    for ds in ("hdfs", "bgl"):
        p = DATA / "predictions" / ds / "lstm_random_seed42_window_level.json"
        if p.exists():
            w = read_json(p)
            win_rows.append({"Dataset": ds.upper(), "P (window)": w["precision"], "R (window)": w["recall"],
                             "F1 (window)": w["f1"], "windows": w["tp"] + w["fp"] + w["tn"] + w["fn"]})
    abl_rows = []
    for ds in ("hdfs", "bgl"):
        for kind, label in (("test_curve", "this replication (packed sequences, whole short sessions, mini-batches)"),
                            ("package_windows", "package windowing + right padding / last step"),
                            ("package_faithful", "package windowing + padding + one optimiser step per session")):
            p = RES / f"lstm_{kind}_{ds}_random_seed42.json"
            if not p.exists():
                continue
            d = read_json(p)
            h = d["history"][d["best_val_epoch"] - 1]
            abl_rows.append({"Dataset": ds.upper(), "LSTM variant": label, "val-selected epoch": d["best_val_epoch"],
                             "P": h["test_precision"], "R": h["test_recall"], "F1": h["test_f1"],
                             "best test F1 over 30 epochs": d["best_test_f1"],
                             "best window-level test F1": d["best_test_window_f1"]})
    comp_view = comp.drop(columns=["seed"]).rename(columns=lambda c: c.replace("_", " "))

    notes_p = RES / "discrepancy_notes.md"
    notes = notes_p.read_text() if notes_p.exists() else "_(discrepancy_notes.md not written yet)_\n"
    md = ["# Part 1 results: replication of RQ1 (Wu, Li & Khomh, EMSE 2023)", "",
          "> Generated by `python -m src.report` from the files in `results/` and `data/`. "
          "Do not edit by hand; edit `results/discrepancy_notes.md` instead.", "",
          "All runs: seed 42, random session-level split (HDFS 70/30, BGL 80/20), session-level metrics on the test set, "
          "positive class = anomaly, threshold 0.5.", "",
          "## 1. Data after parsing and partitioning", "", _md(data_table()), "",
          "## 2. Parsing quality on the Loghub-2.0 2k samples", "", _md(parsing, 3), "",
          "## 3. Main results (6 required runs)", "", _md(main_table(m)), "",
          "## 4. Comparison with the paper (Tables 2 and 3; the paper reports no AUC)", "", _md(comp_view), "",
          "## 5. Step 3a: correlation + redundancy analysis (RF + MCV, fitted on train only)", "", _md(fs_table()), "",
          "Per-feature removals and reasons: `results/feature_selection_hdfs.csv`, `results/feature_selection_bgl.csv`.", "",
          "## 6. Why step 3a is not applied to LSTM + BERT (rank of the template-embedding matrix)", "", _md(rank_df), ""]
    if len(ss):
        md += ["## 7. Seed sensitivity (random split re-drawn per seed; mean ± sample std)", "", _md(ss), ""]
    if len(sens_df):
        md += ["## 8. Split sensitivity: the replication package's loglizer 'uniform' split (seed-independent)", "",
               _md(sens_df), ""]
    if win_rows:
        md += ["## 9. LSTM: window-level metrics (how the package's `evaluation()` computes them), seed 42", "",
               _md(pd.DataFrame(win_rows)), ""]
    if abl_rows:
        md += ["## 10. LSTM ablation towards the replication package (seed 42; diagnostic, not in metrics.csv)", "",
               "Session-level test metrics at the epoch selected on validation; the last two columns peek at the test "
               "set over all epochs (what the package's best-epoch-on-test selection would report).", "",
               _md(pd.DataFrame(abl_rows)), ""]
    md += ["## Figures", ""] + [f"- `results/figures/{f}`" for f in figs] + ["", "## Discrepancy analysis", "", notes]
    (RES / "RESULTS.md").write_text("\n".join(md))
    print("Wrote", RES / "RESULTS.md")


if __name__ == "__main__":
    main()
