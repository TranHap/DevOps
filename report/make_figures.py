"""Figures for the report (vector PDF, sized for IEEE columns). Reads only result files.

  python report/make_figures.py        (from the repository root)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "part2" / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "lstm", "loggpt", "pca"]
LABEL = {"rf": "RF+MCV", "lstm": "LSTM+BERT", "pca": "PCA", "loggpt": "LogGPT", "loggpt_norl": "LogGPT w/o RL"}
COLOR = {"rf": "#2a78d6", "lstm": "#eb6834", "pca": "#1baf7a", "loggpt": "#eda100", "loggpt_norl": "#6250d6"}
LS = {"rf": "-", "lstm": "--", "pca": "-.", "loggpt": ":", "loggpt_norl": (0, (5, 1, 1, 1))}
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42})


def style(ax, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def boxplots():
    runs = pd.read_csv(RES / "runs.csv")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ax, k, name in zip(axes, ("f1", "auc"), ("F1", "ROC-AUC")):
        data = [runs[runs.model == m][k].values for m in MODELS]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLOR[m] + "55")
            patch.set_edgecolor(COLOR[m])
        for i, (d, m) in enumerate(zip(data, MODELS), start=1):
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(d)), d, s=10, color=COLOR[m], edgecolor="white", lw=0.5, zorder=3)
        ax.set_xticks(range(1, 5), [LABEL[m] for m in MODELS])
        ax.set_ylabel(name)
        style(ax)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "boxplots.pdf")
    plt.close(fig)


def roc():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in MODELS + ["loggpt_norl"]:
        d = pd.read_parquet(RES / "scores" / f"{m}_seed0.parquet", columns=["label", "score", "prediction"])
        fpr, tpr, _ = roc_curve(d.label, d.score)
        ax.plot(np.clip(fpr, 1e-6, 1), tpr, color=COLOR[m], ls=LS[m], lw=1.4, label=LABEL[m])
        tp = ((d.prediction == 1) & (d.label == 1)).sum()
        fp = ((d.prediction == 1) & (d.label == 0)).sum()
        ax.scatter(max(fp / (d.label == 0).sum(), 1e-6), tp / (d.label == 1).sum(), s=22, color=COLOR[m],
                   edgecolor="white", lw=0.8, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_xlabel("false positive rate (log scale)")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right")
    style(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(OUT / "roc_seed0.pdf")
    plt.close(fig)


GROUP_NAME = {
    "G33": "T33 delete error (BlockInfo not found)",
    "G1": "T1, T5 receiving / blockMap updated",
    "G3": "T3, T4, T16 PacketResponder / received",
    "G6": "T6, T7, T9, T12 replication",
    "G17": "T17, T32 deleting block",
    "G21": "T21 redundant addStoredBlock",
    "G34": "T34 block not in any file",
    "G18": "T18 empty packet",
    "G10": "T10, T35 serving block",
    "G13": "T13 verification succeeded",
}


def shap_groups(top=10):
    per = pd.read_csv(RES / "explain" / "stability_per_seed.csv")
    order = per.groupby("group").mean_abs_grouped_shap.mean().sort_values(ascending=False).index[:top]
    s0 = per[per.seed == 0].set_index("group").mean_abs_grouped_shap
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = np.arange(len(order))[::-1]
    ax.barh(y, [s0[g] for g in order], color="#2a78d6", alpha=0.85, height=0.62, label="seed 0")
    for yi, g in zip(y, order):
        v = per[per.group == g].mean_abs_grouped_shap.values
        ax.scatter(v, np.full(len(v), yi), s=7, color="black", zorder=3, label="seeds 0–9" if g == order[0] else None)
    ax.set_yticks(y, [GROUP_NAME.get(g, g) for g in order])
    ax.set_xlabel("mean |grouped SHAP value|")
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "shap_groups.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    boxplots()
    roc()
    shap_groups()
    print("figures written to", OUT)
