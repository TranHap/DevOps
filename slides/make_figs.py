"""Slide-sized version of the Scott-Knott boxplot (big fonts, short labels).
Usage: .venv/Scripts/python slides/make_figs.py  -> slides/fig_ranking_{f1,auc}.png
Reads the same inputs as src/rank_models.py, so numbers are identical."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "results", "ext")
SHORT = {"RF (sup.)": "RF", "LSTM (sup.)": "LSTM", "IsolationForest (unsup.)": "Isolation\nForest",
         "DeepLog+EOS (semi-sup.)": "DeepLog\n+EOS"}
COLOR = {"RF (sup.)": "#3D5A80", "LSTM (sup.)": "#3D5A80", "IsolationForest (unsup.)": "#7B5EA7",
         "DeepLog+EOS (semi-sup.)": "#7B5EA7"}

for metric in ["f1", "auc"]:
    wide = pd.read_csv(os.path.join(EXT, f"ranking_{metric}_input.csv"))
    ranks = pd.read_csv(os.path.join(EXT, f"ranking_{metric}.csv")).set_index("model")["rank"]
    order = ranks.sort_values().index
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    bp = ax.boxplot([wide[m] for m in order], patch_artist=True, widths=0.55,
                    medianprops={"color": "#1B2A35", "linewidth": 1.5})
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(COLOR[m]); patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(order) + 1), [f"{SHORT[m]}\nrank {ranks[m]}" for m in order], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = os.path.join(ROOT, "slides", f"fig_ranking_{metric}.png")
    fig.savefig(out, dpi=220)
    print(out)
