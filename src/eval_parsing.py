"""Evaluate parsing accuracy on Loghub-2k ground truth (logparser's corrected version).

Usage: python src/eval_parsing.py
Metrics:
    GA  (Group Accuracy, Zhu et al. ICSE-SEIP'19): fraction of lines whose predicted group contains
        exactly the same set of lines as their ground-truth group.
    #templates predicted vs ground truth.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import ROOT, make_content_parser  # noqa: E402


def group_accuracy(pred, truth):
    df = pd.DataFrame({"pred": pred, "truth": truth})
    truth_sizes = df.groupby("truth").size()
    correct = 0
    for _, g in df.groupby("pred"):
        truths = g["truth"].unique()
        # predicted group is correct iff it maps to exactly one truth group and covers all of it
        if len(truths) == 1 and truth_sizes[truths[0]] == len(g):
            correct += len(g)
    return correct / len(df)


def main():
    rows = []
    for ds in ["HDFS", "BGL"]:
        gt = pd.read_csv(os.path.join(ROOT, "data", "raw", "loghub_2k", f"{ds}_2k_structured.csv"),
                         keep_default_na=False)
        for preprocess in [False, True]:
            parse, templates = make_content_parser(ds, preprocess=preprocess)
            pred = [parse(c or "<EMPTY>") for c in gt["Content"]]
            rows.append({"dataset": ds, "preprocessing": preprocess,
                         "GA": round(group_accuracy(pred, gt["EventId"]), 3),
                         "#templates": len(set(pred)), "#truth": gt["EventId"].nunique()})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    out = os.path.join(ROOT, "results", "parsing_accuracy.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
