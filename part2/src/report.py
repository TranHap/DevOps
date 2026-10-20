"""Build results/RESULTS_PART2.md, descriptive tables and figures from the result files.

  python -m src.report [--config configs/part2_hdfs.yaml]

No number is typed by hand: everything comes from results/runs.csv, results/scores/*.json|parquet,
results/sk_*.csv, results/explain/* and Part 1's results/metrics.csv. Paper reference values (LogGPT,
Table II/III) are constants below, quoted from arXiv 2309.14482.
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import p1  # noqa: F401
from .p1 import PART1_ROOT
from .run_resamples import load_config, p2path
from .scores import LABELS_USED, RANKED, SCORE_DEF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots (validated: dataviz reference palette, light mode), fixed per model
COLOR = {"rf": "#2a78d6", "lstm": "#eb6834", "pca": "#1baf7a", "loggpt": "#eda100", "loggpt_norl": "#6250d6",
         "pca_normal5k": "#1baf7a"}
LINESTYLE = {"rf": "-", "lstm": "--", "pca": "-.", "loggpt": ":", "loggpt_norl": (0, (5, 1, 1, 1))}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PAPER_LOGGPT = {  # arXiv 2309.14482, HDFS, mean ± std over 10 runs
    "loggpt": dict(precision="0.884 ± 0.030", recall="0.921 ± 0.066", f1="0.901 ± 0.036"),
    "loggpt_norl": dict(precision="0.932 ± 0.015", recall="0.790 ± 0.101", f1="0.853 ± 0.065"),
    "pca_normal5k": dict(precision="0.166 ± 0.008", recall="0.059 ± 0.003", f1="0.087 ± 0.002"),
}
METRICS = ["precision", "recall", "f1", "auc"]


def lab(m):
    return SCORE_DEF[m]["label"]


def md_table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i in range(len(df)):                    # column-wise access keeps each column's dtype (iterrows upcasts)
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else format(v, fmt.get(c, ".4f")))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def read_json(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------- tables
def descriptive(runs, models):
    rows = []
    for m in models:
        r = runs[runs.model == m]
        if r.empty:
            continue
        for k in METRICS:
            v = r[k].astype(float)
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            rows.append(dict(model=lab(m), metric=k, n_resamples=len(v), median=v.median(), q1=q1, q3=q3, iqr=q3 - q1,
                             mean=v.mean(), std=v.std(ddof=1) if len(v) > 1 else np.nan, min=v.min(), max=v.max()))
    return pd.DataFrame(rows)


def cost(runs, models):
    rows = []
    for m in models:
        r = runs[runs.model == m]
        if r.empty:
            continue
        rows.append({"model": lab(m), "n_resamples": len(r), "training sessions": int(r.n_train_sessions.median()),
                     "train time median (s)": r.train_time_s.median(), "train time mean ± std (s)":
                     f"{r.train_time_s.mean():.1f} ± {r.train_time_s.std(ddof=1):.1f}" if len(r) > 1 else f"{r.train_time_s.mean():.1f}",
                     "inference median (s)": r.infer_time_s.median(),
                     "seed 0 train (s), machine to itself": float(r[r.seed == 0].train_time_s.iloc[0]) if (r.seed == 0).any() else np.nan,
                     "concurrent processes": " / ".join(map(str, sorted(set(r.get("concurrent_processes", pd.Series([1])).astype(int))))),
                     "device": "; ".join(sorted(set(r.device)))})
    return pd.DataFrame(rows)


def loggpt_k_sensitivity(sdir, seeds, Ks=(6, 7, 8, 9)):
    """Re-threshold the stored LogGPT scores at fixed K (decision rule only; sensitivity, never used for selection)."""
    from .scores import session_metrics
    rows = []
    for s in seeds:
        for m in ("loggpt", "loggpt_norl"):
            p = sdir / f"{m}_seed{s}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p, columns=["label", "score"])
            k_rule = int(read_json(sdir / f"{m}_seed{s}.json")["threshold"])
            for k in Ks:
                met = session_metrics(d.label.values, d.score.values, (d.score.values > k).astype(int))
                rows.append(dict(seed=s, model=m, K=k, K_rule=k_rule, **met))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def boxplots(runs, fig_dir, models, suffix=""):
    rng = np.random.default_rng(0)
    for k in ("f1", "auc"):
        data = [runs[runs.model == m][k].astype(float).values for m in models]
        if not all(len(d) for d in data):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(COLOR[m] + "55")
            patch.set_edgecolor(COLOR[m])
        for i, (d, m) in enumerate(zip(data, models), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=22, color=COLOR[m], edgecolor="white", lw=0.8, zorder=3)
        ax.set_xticks(range(1, len(models) + 1), [lab(m) for m in models])
        ax.set_ylabel(k.upper() if k == "auc" else "F1")
        n = min(len(d) for d in data)
        ax.set_title(f"{'ROC-AUC' if k == 'auc' else 'F1'} over {n} resamples (HDFS, session level; dots = seeds)",
                     fontsize=10, color=INK)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"boxplot_{k}{suffix}.png", dpi=170)
        plt.close(fig)


def roc_seed0(scores_dir, fig_dir, models):
    from sklearn.metrics import roc_curve
    curves = {}
    for m in models:
        p = scores_dir / f"{m}_seed0.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["label", "score"])
            fpr, tpr, _ = roc_curve(d.label, d.score)
            curves[m] = (fpr, tpr)
    if not curves:
        return
    meta = {m: read_json(scores_dir / f"{m}_seed0.json") for m in curves}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, logx in zip(axes, (False, True)):
        for m, (fpr, tpr) in curves.items():
            x = np.clip(fpr, 1e-6, 1) if logx else fpr
            ax.plot(x, tpr, color=COLOR[m], ls=LINESTYLE.get(m, "-"), lw=2,
                    label=f"{lab(m)} (AUC {meta[m]['metrics']['auc']:.4f})")
            # operating point = the model's own threshold
            mt = meta[m]["metrics"]
            op_fpr = mt["fp"] / (mt["fp"] + mt["tn"])
            ax.scatter(max(op_fpr, 1e-6) if logx else op_fpr, mt["recall"], s=55, color=COLOR[m],
                       edgecolor="white", lw=1.2, zorder=4)
        ax.set_xlabel("false positive rate" + (" (log scale)" if logx else ""))
        ax.set_ylabel("true positive rate (recall)")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 1)
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_title("ROC, seed 0 (dot = operating point of the model's own threshold)", fontsize=9, color=INK)
    axes[1].set_title("same curves, FPR on a log scale", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_seed0.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2_hdfs.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = p2path(cfg, "results_dir")
    fig_dir = p2path(cfg, "results_dir", "figures")
    sdir = p2path(cfg, "results_dir", "scores")
    runs = pd.read_csv(res / "runs.csv")
    seeds = sorted(runs.seed.unique())
    present = [m for m in RANKED if m in set(runs.model)]
    extra = [m for m in ("loggpt_norl", "pca_normal5k") if m in set(runs.model)]

    desc = descriptive(runs, present + extra)
    desc.to_csv(res / "descriptive_stats.csv", index=False)
    cst = cost(runs, present + extra)
    cst.to_csv(res / "cost.csv", index=False)
    boxplots(runs, fig_dir, present)
    if "loggpt_norl" in extra and "loggpt" in present:
        boxplots(runs, fig_dir, ["loggpt_norl", "loggpt"], suffix="_loggpt_ablation")
    roc_seed0(sdir, fig_dir, present + [m for m in extra if m == "loggpt_norl"])

    L = ["# Part 2 results: unsupervised / semi-supervised models, statistical ranking, explanation", "",
         "> Generated by `python -m src.report` from the files in `results/`. Do not edit by hand.", "",
         f"Dataset: HDFS (Part 1 parsing and sessions: 575,061 sessions, 16,838 anomalous). Resamples: seeds "
         f"{', '.join(map(str, seeds))} ({len(seeds)}) of Part 1's random 70/30 session-level split. "
         "All models of a resample are scored on the same full 30% test split (asserted on session ids).", ""]

    # 1. setup
    n_lg = int(cfg["loggpt"]["train_samples"])
    L += ["## 1. Setup and fairness protocol", ""]
    rows = []
    for m in present + extra:
        r = runs[runs.model == m].iloc[0]
        rows.append({"model": lab(m), "ranked": "yes" if m in RANKED else "no (ablation)",
                     "learning regime": {"rf": "supervised", "lstm": "supervised", "pca": "unsupervised",
                                         "loggpt": "semi-supervised", "loggpt_norl": "semi-supervised",
                                         "pca_normal5k": "semi-supervised (normal-only)"}[m],
                     "labels used": LABELS_USED[m], "training sessions": int(r.n_train_sessions),
                     "score": SCORE_DEF[m]["score"], "decision": f"score {SCORE_DEF[m]['rule']} {SCORE_DEF[m]['threshold']}"})
    L += [md_table(pd.DataFrame(rows)), "",
          "Training data per resample: RF and LSTM use the whole train split with labels (the LSTM holds 10% of it out "
          "for epoch selection). PCA uses the whole train split with its labels replaced by −1. LogGPT uses "
          f"{n_lg:,} normal sessions sampled from the train split ({int(0.9 * n_lg):,} for training, "
          f"{n_lg - int(0.9 * n_lg):,} for validation); labels are used only for this "
          "filter. No model uses test data or anomaly labels for any training-time decision.", ""]

    # 2. performance
    L += ["## 2. Performance over the resamples (session level, positive class = anomaly)", ""]
    if len(desc):
        piv = desc.copy()
        piv["median [IQR]"] = piv.apply(lambda r: f"{r['median']:.4f} [{r.q1:.4f}–{r.q3:.4f}]", axis=1)
        piv["mean ± std"] = piv.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if not pd.isna(r["std"])
                                      else f"{r['mean']:.4f}", axis=1)
        for k in METRICS:
            sub = piv[piv.metric == k][["model", "n_resamples", "median [IQR]", "mean ± std", "min", "max"]]
            L += [f"**{k.upper() if k == 'auc' else k.capitalize()}**", "", md_table(sub), ""]
    per_seed = runs.pivot_table(index="seed", columns="model", values="f1")
    per_seed = per_seed[[m for m in present + extra if m in per_seed.columns]]
    per_seed.columns = [lab(m) for m in per_seed.columns]
    L += ["F1 per seed:", "", md_table(per_seed.reset_index().astype({"seed": int})), "",
          "Confusion matrices per run: `results/runs.csv` (tp, fp, tn, fn). Figures: `figures/boxplot_f1.png`, "
          "`figures/boxplot_auc.png`, `figures/roc_seed0.png`.", ""]

    # 3. Scott-Knott
    L += ["## 3. Scott-Knott ESD ranking (ScottKnottESD 3.0.0, `version = \"np\"`, α = 0.05)", ""]
    for k in ("f1", "auc"):
        p = res / f"sk_{k}.csv"
        if not p.exists():
            L += [f"_{k.upper()}: not run yet (`Rscript src/scott_knott.R results`)._", ""]
            continue
        sk = pd.read_csv(p)
        eff = pd.read_csv(res / f"sk_{k}_effect_sizes.csv")
        L += [f"**{k.upper()}** (group 1 = best; {int(sk.n_resamples.iloc[0])} resamples)", "",
              md_table(sk[["model", "group", "median", "mean"]]), "",
              "Pairwise effect sizes (`checkDifference`: |Cohen's d|; plus Cliff's δ, the effect size the np version "
              "uses to merge groups):", "",
              md_table(eff[["i", "j", "ranki", "rankj", "cohen_d_abs", "cohen_d_magnitude", "cliff_delta",
                            "cliff_magnitude"]], {"cohen_d_abs": ".3f", "cliff_delta": ".3f"}), "",
              f"Plot: `figures/sk_{k}.png`; input data: `sk_input_{k}.csv`.", ""]

    # 4. cost
    L += ["## 4. Cost (wall-clock seconds per resample)", "", md_table(cst, {"train time median (s)": ".1f",
                                                                               "inference median (s)": ".2f",
                                                                               "seed 0 train (s), machine to itself": ".1f"}), "",
          "`concurrent processes` = number of run_resamples processes sharing the machine during the run; every "
          "reported run had the machine to itself (DECISIONS P2-17: a parallel attempt slowed each LogGPT run ≈ 2.4× "
          "and was abandoned).",
          "Training time includes everything inside `fit` (LSTM: windowing and 30 epochs with validation; LogGPT: 100 "
          "pre-training epochs + RL episodes). Inference = scoring the full test split (172,519 sessions).", ""]

    # LogGPT / PCA details
    lg = []
    for s in seeds:
        p = sdir / f"loggpt_seed{s}.json"
        if not p.exists():
            continue
        j = read_json(p)
        i, v = j["info"], j["info"]["verify"]
        jn = read_json(sdir / f"loggpt_norl_seed{s}.json")["info"]["verify"]
        lg.append(dict(seed=s, unique_train_keys=i["n_unique_train_keys"], K=i["K"], K_code_default=i["K_original_code_default"],
                       pretrain_best_val_epoch=i["pretrain_best_val_epoch"], rl_episodes=i["rl_episodes_run"],
                       rl_selected_episode=i["rl_selected_episode"], early_stopped=i["rl_early_stopped"],
                       pretrain_s=round(i["times"]["pretrain_s"], 1), rl_s=round(i["times"]["rl_s"], 1),
                       test_with_unk=v["n_test_with_unk"],
                       unk_rule_changes=f"{v['n_test_decisions_changed_by_unk_rule']} / {jn['n_test_decisions_changed_by_unk_rule']}",
                       verify_mismatch=f"{v['mismatch_batched_vs_original']}+{v['mismatch_without_unk']} / "
                                       f"{jn['mismatch_batched_vs_original']}+{jn['mismatch_without_unk']}",
                       distinct_test_seqs=i["score_ft"]["n_distinct_sequences"],
                       max_test_len=i["score_ft"]["max_len"], n_positions=j["info"].get("n_positions", 512)))
    if lg:
        L += ["### LogGPT run details", "", md_table(pd.DataFrame(lg), {"pretrain_s": ".1f", "rl_s": ".1f"}), "",
              "`unk_rule_changes`: test decisions that differ from the original code's `<unk>` handling (LogGPT / w/o RL). "
              "`verify_mismatch`: mismatches between our score and the original `_predict_topk` on ~300 test sessions, "
              "(batched implementation) + (sessions without `<unk>`), LogGPT / w/o RL; 0+0 means exact reproduction "
              "(DECISIONS P2-13). `max_test_len` = longest test session in events; with `<bos>`/`<eos>` it must fit in "
              "`n_positions` (GPT-2 context), so no session is truncated.", ""]
    pc = []
    for s in seeds:
        for m in ("pca", "pca_normal5k"):
            p = sdir / f"{m}_seed{s}.json"
            if p.exists():
                i = read_json(p)["info"]
                pc.append(dict(seed=s, model=lab(m), n_features=i["n_features"], n_components=i["n_components"],
                               explained_variance=i["explained_variance"], spe_threshold=i["threshold"]))
    if pc:
        L += ["### PCA run details", "", md_table(pd.DataFrame(pc)), ""]

    # 5. ablations
    L += ["## 5. Ablations (not ranked)", ""]
    if "loggpt_norl" in extra:
        a_ = runs[runs.model.isin(["loggpt", "loggpt_norl"])].pivot_table(index="seed", columns="model", values=METRICS)
        rows = []
        for k in METRICS:
            d = a_[k]["loggpt"] - a_[k]["loggpt_norl"]
            rows.append(dict(metric=k, **{"LogGPT w/o RL (mean)": a_[k]["loggpt_norl"].mean(),
                                          "LogGPT (mean)": a_[k]["loggpt"].mean(),
                                          "mean paired difference": d.mean(),
                                          "seeds RL better / equal / worse": f"{int((d > 0).sum())} / {int((d == 0).sum())} / {int((d < 0).sum())}"}))
        L += ["**LogGPT vs LogGPT w/o RL** (same pre-trained model, before/after RL fine-tuning; paired by seed)", "",
              md_table(pd.DataFrame(rows)), "", "Figure: `figures/boxplot_f1_loggpt_ablation.png`.", ""]
    if "loggpt" in present:
        ks = loggpt_k_sensitivity(sdir, seeds)
        ks.to_csv(res / "loggpt_k_sensitivity.csv", index=False)
        lsm = runs[runs.model == "lstm"].f1
        pcm = runs[runs.model == "pca"].f1
        piv = ks.pivot_table(index="model", columns="K", values=["precision", "recall", "f1"]).round(4)
        rows = []
        for m in ("loggpt", "loggpt_norl"):
            for k in sorted(ks.K.unique()):
                sub = ks[(ks.model == m) & (ks.K == k)]
                rows.append({"model": lab(m), "K (post hoc)": int(k), "P mean": sub.precision.mean(), "R mean": sub.recall.mean(),
                             "F1 mean": sub.f1.mean(), "F1 min": sub.f1.min(), "F1 max": sub.f1.max(),
                             "seeds where this K is the 50% rule": int((sub.K_rule == k).sum())})
        kk = ks[ks.model == "loggpt"].drop_duplicates("seed")
        L += ["**LogGPT: sensitivity to K (post hoc, not used for the ranking)**", "",
              f"K = int(0.5 × unique training keys) gives K = 7 in {int((kk.K_rule == 7).sum())} seeds and K = 8 in "
              f"{int((kk.K_rule == 8).sum())} seeds (15–17 keys among the 5,000 sampled sessions; the code's default is "
              "always 7). The table re-thresholds the stored scores (score > K) of every seed at a fixed K. Caveat: the RL "
              "reward of each seed used that seed's own K, so this changes only the decision rule, not the fine-tuning.", "",
              md_table(pd.DataFrame(rows)), "",
              f"For reference: LSTM + BERT F1 ranges {lsm.min():.4f}–{lsm.max():.4f} and PCA {pcm.min():.4f}–{pcm.max():.4f} "
              "over the 10 resamples. Per-seed values: `loggpt_k_sensitivity.csv`.", ""]
    if "pca_normal5k" in extra:
        r = runs[runs.model.isin(["pca", "pca_normal5k"]) & runs.seed.isin(runs[runs.model == "pca_normal5k"].seed)]
        t = r[["seed", "model", "n_train_sessions", "labels_used"] + METRICS + ["tp", "fp", "fn"]].copy()
        t["model"] = t.model.map(lab)
        L += ["**PCA: full unlabeled train split vs the 5,000 normal sessions LogGPT uses**", "", md_table(t), ""]

    # 6. reference values
    L += ["## 6. Comparison with reference values", ""]
    m1 = pd.read_csv(PART1_ROOT / "results" / "metrics.csv")
    m1 = m1[(m1.dataset == "hdfs") & (m1.split == "random") & m1.seed.isin([0, 1, 2, 3, 4])]
    rows = []
    for m, (mod, fs) in (("rf", ("RF", "without")), ("lstm", ("LSTM", None))):
        p1r = m1[(m1.model == mod) & ((m1.feature_selection == fs) if fs else m1.feature_selection.isna())]
        p2r = runs[(runs.model == m) & runs.seed.isin(p1r.seed)]
        if p2r.empty:
            continue
        common = sorted(set(p1r.seed) & set(p2r.seed))
        f1a, f1b = p1r[p1r.seed.isin(common)].f1.mean(), p2r[p2r.seed.isin(common)].f1.mean()
        rows.append({"model": lab(m), "seeds": " ".join(map(str, common)), "Part 1 mean F1": f1a, "Part 2 mean F1": f1b,
                     "|Δ|": abs(f1b - f1a), "check (|Δ| ≤ 0.01)": "PASS" if abs(f1b - f1a) <= 0.01 else "FAIL"})
    L += ["**Regression check against Part 1** (brief §6: same seeds, mean F1 within 0.01)", "",
          md_table(pd.DataFrame(rows), {"|Δ|": ".6f"}), ""]
    rows = []
    for m in ("loggpt", "loggpt_norl", "pca_normal5k", "pca"):
        r = runs[runs.model == m]
        if r.empty:
            continue
        pv = PAPER_LOGGPT.get(m if m != "pca" else "pca_normal5k")
        rows.append({"model": lab(m), "n runs": len(r), "ours P": f"{r.precision.mean():.3f} ± {r.precision.std(ddof=1):.3f}" if len(r) > 1 else f"{r.precision.mean():.3f}",
                     "ours R": f"{r.recall.mean():.3f} ± {r.recall.std(ddof=1):.3f}" if len(r) > 1 else f"{r.recall.mean():.3f}",
                     "ours F1": f"{r.f1.mean():.3f} ± {r.f1.std(ddof=1):.3f}" if len(r) > 1 else f"{r.f1.mean():.3f}",
                     "paper P": pv["precision"], "paper R": pv["recall"], "paper F1": pv["f1"],
                     "paper row": {"loggpt": "LogGPT", "loggpt_norl": "LogGPT w/o RL"}.get(m, "PCA (normal-only training)")})
    L += ["**LogGPT paper (HDFS, Table II / III, 10 runs)**", "", md_table(pd.DataFrame(rows)), "",
          "Protocol differences: the paper's test set is every session not used for training (553,223 normal + 16,838 "
          "anomalous) and its parsing is its own Drain run; ours is the 30% split (≈ 167.5k normal + ≈ 5.0k anomalous) "
          "with Part 1's parsing. The paper's PCA is trained on normal sessions only (loglizer), which our "
          "`PCA (5,000 normal)` row mirrors; our ranked PCA is trained on the full unlabeled train split.", ""]

    # 7. explanation
    ex = res / "explain"
    L += ["## 7. Explanation of RF + MCV (without step 3a), SHAP", ""]
    if (ex / "shap_grouped.csv").exists():
        info = read_json(ex / "sample_seed0_info.json")
        g = pd.read_csv(ex / "shap_global.csv").head(10)
        gr = pd.read_csv(ex / "shap_grouped.csv").head(10)
        L += [f"Sample (seed 0): {info['n_anomalous']} anomalous + {info['n_normal']} normal test sessions "
              f"({info['sampling']}); base value {info['base_value']:.4f}. Figures: `explain/shap_bar.png`, "
              "`explain/shap_beeswarm.png`, `explain/shap_grouped_bar.png`.", "",
              "**Top templates (mean |SHAP|)**", "",
              md_table(g[["rank", "template_id", "template", "mean_abs_shap", "mean_shap_anomalous", "mean_shap_normal",
                          "mean_count_anomalous", "mean_count_normal"]]), "",
              "**Top correlation clusters (mean |grouped SHAP|)**", "",
              md_table(gr[["rank", "group", "n_members", "members", "member_templates", "mean_abs_grouped_shap",
                           "sum_member_mean_abs_shap"]]), ""]
    if (ex / "shap_crosscheck.csv").exists():
        cc = pd.read_csv(ex / "shap_crosscheck.csv")
        L += ["**Agreement**", "", md_table(cc, {"kendall_tau": ".3f", "p_value": ".2g"}), ""]
    if (ex / "stability.csv").exists():
        st = pd.read_csv(ex / "stability.csv").head(10)
        si = read_json(ex / "stability_info.json")
        L += [f"**Stability over {len(si['seeds'])} resamples** (grouping: {si['grouping']}; own clusters identical "
              f"to seed 0's in {sum(si['own_clusters_identical_to_seed0'].values())} of "
              f"{len(si['own_clusters_identical_to_seed0'])} seeds)", "",
              md_table(st[["group", "members", "mean_importance", "std_importance", "median_rank", "best_rank",
                           "worst_rank", "n_top5"]], {"median_rank": ".1f", "best_rank": ".0f", "worst_rank": ".0f"}), ""]
        if (ex / "stability_own_clusters.csv").exists():
            own = pd.read_csv(ex / "stability_own_clusters.csv").head(10)
            ari = [v for k, v in si.get("adjusted_rand_index_vs_seed0", {}).items() if k != "0"]
            L += ["Robustness to the clustering itself: the same importance computed with **each seed's own** step-3a "
                  "clusters (recomputed on that seed's train split). Adjusted Rand index of seeds 1–9's clusters vs seed 0's: "
                  f"min {min(ari):.2f}, median {np.median(ari):.2f}, max {max(ari):.2f}." if ari else "", "",
                  md_table(own[["members", "member_templates", "n_seeds_present", "mean_importance", "median_rank",
                                "n_top5"]], {"median_rank": ".1f"}), ""]
        if (ex / "sk_feature_importance.csv").exists():
            skf = pd.read_csv(ex / "sk_feature_importance.csv").head(10)
            L += ["Scott-Knott ESD (np) on the per-resample cluster importance (top 10):", "", md_table(skf), ""]
    if (ex / "local_cases.md").exists():
        L += ["Local explanations (2 TP, every FN, up to 5 FP of seed 0): `explain/local_cases.md`, figures in "
              "`explain/local/`.", ""]

    # 8. threats
    n_rf = int(runs[runs.model == "rf"].n_train_sessions.median()) if "rf" in present else None
    L += ["## 8. Threats to validity", "",
          "- **Different label usage.** RF and LSTM learn from all labels, LogGPT only from the knowledge that its training "
          "sessions are normal, PCA from no label at all. The ranking compares learning regimes, not only algorithms.",
          f"- **Different training-set sizes.** LogGPT trains on 5,000 sessions (paper protocol), the supervised models on "
          f"{n_rf:,} sessions." if n_rf else "- **Different training-set sizes.** LogGPT trains on 5,000 sessions.",
          "- **Temporal leakage.** The random session-level split places later blocks in the training set (Part 1, D-10).",
          "- **Fixed thresholds.** K (50% of the training keys) and PCA's Q-statistic threshold come from the original "
          "papers/code and were not tuned; the AUC column shows the threshold-free view. K itself moves with the sample: "
          "the 5,000 training sessions contain 15–17 unique keys, so K is 7 or 8 depending on the seed (§5, post-hoc "
          "sensitivity).",
          "- **One dataset.** HDFS only (BGL windows are ≈ 50% anomalous, which breaks the rare-anomaly assumption of the "
          "unsupervised detectors).",
          "- **Implementation.** LogGPT was ported to current libraries and fed Part 1's sessions; its scoring was verified "
          "against the original code (§4), but GPU kernels are not bit-reproducible.", ""]
    (res / "RESULTS_PART2.md").write_text("\n".join(L))
    print(f"wrote {res / 'RESULTS_PART2.md'}")


if __name__ == "__main__":
    main()
