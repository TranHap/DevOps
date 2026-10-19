"""Extension E2 - Scott-Knott ESD ranking of the 4 models on HDFS over the 10 resamples (seeds 0-9).

Usage: python src/rank_models.py [--dataset HDFS]

Models: RF + MCV and LSTM + TF-IDF(Text) (supervised, without corr/VIF: their best variants from RQ1),
Isolation Forest + MCV (unsupervised), DeepLog (semi-supervised). Every model uses the same 10 splits.
Outputs: results/ext/ranking_<metric>.csv, results/ext/ranking_summary.csv, results/ext/ranking_<metric>.png
"""
import argparse
import glob
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RSCRIPT = r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe"
METRICS = ["f1", "auc", "precision", "recall"]


def load(ds):
    pats = {"RF (sup.)": f"rq1/RF_MCV_{ds}_none_seed*.json",
            "LSTM (sup.)": f"rq1/LSTM_TFIDF_{ds}_none_seed*.json",
            "IsolationForest (unsup.)": f"ext/IF_MCV_{ds}_seed*.json",
            "DeepLog (semi-sup.)": f"ext/DeepLog_{ds}_seed*.json",
            "DeepLog+EOS (semi-sup.)": f"ext/DeepLog_eos_{ds}_seed*.json"}
    rows = []
    for name, pat in pats.items():
        for f in glob.glob(os.path.join(glob.escape(ROOT), "results", pat)):
            r = json.load(open(f, encoding="utf-8"))
            rows.append({"model": name, "seed": r["seed"], **{m: r[m] for m in METRICS}})
    return pd.DataFrame(rows)


MAIN = ["RF (sup.)", "LSTM (sup.)", "IsolationForest (unsup.)", "DeepLog+EOS (semi-sup.)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HDFS")
    ds = ap.parse_args().dataset
    df = load(ds)
    print("runs per model:", df.groupby("model").size().to_dict())
    # main ranking = the 4 models of the assignment; ablation adds the original DeepLog (no EOS)
    for prefix, models in [("ranking", MAIN), ("ranking_ablation", sorted(df["model"].unique()))]:
        print(f"\n== {prefix} ==")
        rank(df[df["model"].isin(models)], ds, prefix)


def rank(df, ds, prefix):
    out_dir = os.path.join(ROOT, "results", "ext")

    summary = []
    for metric in ["f1", "auc"]:
        wide = df.pivot(index="seed", columns="model", values=metric).dropna()
        wide_path = os.path.join(out_dir, f"{prefix}_{metric}_input.csv")
        wide.to_csv(wide_path, index=False)
        rank_path = os.path.join(out_dir, f"{prefix}_{metric}.csv")
        subprocess.run([RSCRIPT, os.path.join(ROOT, "src", "scott_knott.R"), wide_path, rank_path],
                       check=True, capture_output=True, text=True)
        ranks = pd.read_csv(rank_path).set_index("model")["rank"]
        for m in wide.columns:
            summary.append({"metric": metric, "model": m, "sk_rank": int(ranks[m]),
                            "mean": round(wide[m].mean(), 4), "std": round(wide[m].std(), 4),
                            "median": round(wide[m].median(), 4), "n": len(wide)})

        order = ranks.sort_values().index
        fig, ax = plt.subplots(figsize=(1.9 * len(order) + 1, 3.6))
        ax.boxplot([wide[m] for m in order], showmeans=True)
        ax.set_xticks(range(1, len(order) + 1), [f"{m}\nrank {ranks[m]}" for m in order], fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{ds}: {metric.upper()} over {len(wide)} resamples (Scott-Knott ESD ranks)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_{metric}.png"), dpi=200)
        plt.close(fig)

    s = pd.DataFrame(summary).sort_values(["metric", "sk_rank", "mean"], ascending=[True, True, False])
    s.to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
