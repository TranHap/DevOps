"""Parsing quality on the Loghub-2.0 2k samples (corrected ground truth): GA and PA.

Configurations (Drain depth/st from the config):
  none          Drain without any masking
  loghub        Drain + Loghub benchmark regexes ("default configuration")
  paper         Drain + the replication package's regexes (HDFS: none listed -> benchmark defaults)
  paper_refined Drain + the package's refined BGL regexes (BGL only)
  qin           Drain + Qin et al. (SANER 2025) preprocessing framework

GA (grouping accuracy): logpai evaluator -- a message is correct if its predicted group
   contains exactly the messages of its ground-truth group.
PA (parsing accuracy): Loghub-2.0 definition -- share of messages whose predicted template
   string equals the ground-truth template (whitespace-normalised), no template correction.
PA_relaxed: same, after collapsing placeholders separated only by punctuation ('<*>:<*>' -> '<*>',
   '/<*>' -> '<*>') in both strings, so that masking conventions (IP:port as one or two
   placeholders) do not count as errors. Reported as a supplementary number only.

Usage: python -m src.parsing_eval   -> results/parsing_eval.csv
"""
import re

import pandas as pd

from .common import get_logger, load_config, resolve, results_dir
from .parse import parse_contents
from .preprocess import Masker


def grouping_accuracy(gt, pred):
    gt, pred = pd.Series(list(gt)), pd.Series(list(pred))
    gt_counts = gt.value_counts()
    correct = 0
    for p, idx in pred.groupby(pred).groups.items():
        g = gt[idx].value_counts()
        if g.size == 1 and gt_counts[g.index[0]] == len(idx):
            correct += len(idx)
    return correct / len(gt)


def _norm(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def _relax(t):
    """Placeholder-convention-insensitive form: '<*>:<*>', '/<*>', '<*>,<*>' ... -> '<*>'."""
    t = _norm(t)
    t = re.sub(r"[/]?<\*>(?:[:/.,\-]?<\*>)+", "<*>", t)
    return re.sub(r"/<\*>", "<*>", t)


def parsing_accuracy(gt_templates, pred_templates, relaxed=False):
    f = _relax if relaxed else _norm
    return sum(f(a) == f(b) for a, b in zip(gt_templates, pred_templates)) / len(gt_templates)


def evaluate(cfg, mode_name, log):
    gt = pd.read_csv(resolve(cfg["sample_2k"]), keep_default_na=False)
    contents = gt.Content.astype(str).tolist()
    if mode_name == "none":
        masker = Masker("none")
    elif mode_name == "qin":
        masker = Masker("qin")
    else:
        masker = Masker("regex", cfg["regex"][mode_name])
    pred, templates = parse_contents(contents, masker, cfg["drain"]["depth"], cfg["drain"]["st"],
                                     cfg["drain"].get("max_child", 100))
    ga = grouping_accuracy(gt.EventTemplate, pred)
    pa = parsing_accuracy(gt.EventTemplate, pred)
    pa_r = parsing_accuracy(gt.EventTemplate, pred, relaxed=True)
    row = dict(dataset=cfg["name"], config=mode_name, n_lines=len(gt),
               n_templates_ground_truth=gt.EventTemplate.nunique(), n_templates_parsed=len(templates),
               GA=round(ga, 4), PA=round(pa, 4), PA_relaxed=round(pa_r, 4), masking=str(masker.describe()))
    log.info("%-5s %-14s GA %.3f PA %.3f PA_relaxed %.3f templates %d (gt %d)", cfg["name"], mode_name, ga, pa, pa_r,
             len(templates), row["n_templates_ground_truth"])
    return row


def main():
    log = get_logger("parsing_eval", "logs/parsing_eval.log")
    rows = []
    for name in ("hdfs", "bgl"):
        cfg = load_config(name)
        for mode in ["none", "loghub", "paper"] + (["paper_refined"] if "paper_refined" in cfg["regex"] else []) + ["qin"]:
            rows.append(evaluate(cfg, mode, log))
    out = results_dir(load_config("hdfs")) / "parsing_eval.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
