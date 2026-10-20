"""Part 1 driver.

  python -m src.run_part1 --dataset {hdfs,bgl,<config path>} --seed 42 \
      --stage {parse,partition,split,features,train,eval,all} [--model {rf,rf_fs,lstm,all}] \
      [--split-method {random,loglizer_uniform}] [--force]

Every stage caches its output and is skipped when the output exists (use --force to redo).
"""
import argparse
import time

import numpy as np
import pandas as pd

from .common import (data_dir, get_logger, load_config, read_json, results_dir, set_seed,
                     write_json)
from .evaluate import comparison_with_paper, compute_metrics, upsert_metrics
from .parse import parse_file
from .partition import load_sessions, partition
from .represent_bert import compute_embeddings, embedding_rank
from .represent_mcv import MCVVectorizer
from .split import get_split

STAGES = ["parse", "partition", "split", "features", "train", "eval"]
MODELS = ["rf", "rf_fs", "lstm"]
MODEL_ID = {"rf": ("RF", "MCV", "without"), "rf_fs": ("RF", "MCV", "with"), "lstm": ("LSTM", "BERT", "n/a")}


def load_templates(cfg):
    return pd.read_csv(data_dir(cfg, "parsed") / "templates.csv", keep_default_na=False)


def pred_path(cfg, model, split, seed):
    return data_dir(cfg, "predictions") / f"{model}_{split}_seed{seed}.npz"


def stage_features(cfg, seed, split, log, force):
    templates = load_templates(cfg)
    E = compute_embeddings(cfg, templates, log, force=force)
    res = results_dir(cfg)
    rank_file = res / "bert_rank.json"
    ranks = read_json(rank_file) if rank_file.exists() else {}
    ranks[cfg["name"]] = embedding_rank(E)
    write_json(rank_file, ranks)
    log.info("BERT rank: %s", {k: ranks[cfg['name']][k] for k in ('n_templates', 'embedding_dim', 'rank', 'rank_centered')})
    s = load_sessions(cfg)
    tr, te = get_split(cfg, seed, log, split)
    vec = MCVVectorizer(len(templates)).fit(s.subset(tr))
    stats = {}
    vec.transform(s.subset(te), stats)
    info = dict(n_templates=len(templates), n_features_train_vocab=int(len(vec.vocab)),
                n_templates_unseen_in_train=int(len(templates) - len(vec.vocab)), test=stats)
    write_json(data_dir(cfg, "features") / f"mcv_{split}_seed{seed}.json", info)
    log.info("MCV: %d templates, %d in train vocabulary; test events unseen in train: %d of %d (%d sessions)",
             len(templates), len(vec.vocab), stats["n_events_unseen_in_train"], stats["n_events"],
             stats["n_sessions_with_unseen"])


def lstm_test_curve(cfg, seed, split, log, package=False, package_windows=False):
    """DIAGNOSTIC: retrain the LSTM exactly as in stage_train but also log test metrics per epoch.
    package=True: ablation with the replication package's windowing / padding / per-session steps.
    Nothing here feeds metrics.csv; output: results/lstm_{test_curve|package_faithful}_<name>_<split>_seed<seed>.json"""
    from .models.lstm import LSTMBert
    s = load_sessions(cfg)
    tr, te = get_split(cfg, seed, log, split)
    set_seed(seed)
    E = np.load(data_dir(cfg, "features") / "bert_embeddings.npy")
    params = dict(cfg["lstm"], package_faithful=package, package_windows=package_windows)
    model = LSTMBert(E, cfg["window"], params, seed, log).fit(s.subset(tr), diag_test_sessions=s.subset(te))
    h = model.history
    best_test = max(h, key=lambda r: r["test_f1"])
    best_test_w = max(h, key=lambda r: r["test_window_f1"])
    kind = "package_faithful" if package else ("package_windows" if package_windows else "test_curve")
    out = results_dir(cfg) / f"lstm_{kind}_{cfg['name']}_{split}_seed{seed}.json"
    write_json(out, dict(note="diagnostic only; model selection uses validation F1", package_faithful=package,
                         package_windows=package_windows,
                         best_val_epoch=model.best_epoch,
                         test_window_f1_at_best_val_epoch=h[model.best_epoch - 1]["test_window_f1"],
                         best_test_window_epoch=best_test_w["epoch"], best_test_window_f1=best_test_w["test_window_f1"],
                         best_test_window_precision=best_test_w["test_window_precision"],
                         best_test_window_recall=best_test_w["test_window_recall"],
                         test_f1_at_best_val_epoch=h[model.best_epoch - 1]["test_f1"],
                         best_test_epoch=best_test["epoch"], best_test_f1=best_test["test_f1"],
                         best_test_precision=best_test["test_precision"], best_test_recall=best_test["test_recall"],
                         history=h))
    log.info("[lstm %s] session-level test F1 at best-val epoch %d = %.3f; best test F1 over epochs = %.3f (epoch %d); "
             "best window-level test F1 = %.3f (epoch %d)", kind, model.best_epoch, h[model.best_epoch - 1]["test_f1"],
             best_test["test_f1"], best_test["epoch"], best_test_w["test_window_f1"], best_test_w["epoch"])


def stage_train(cfg, seed, split, models, log, force):
    from .models.lstm import LSTMBert
    from .models.rf import RFMCV
    templates = load_templates(cfg)
    s = load_sessions(cfg)
    tr, te = get_split(cfg, seed, log, split)
    train_s, test_s = s.subset(tr), s.subset(te)
    res = results_dir(cfg)
    mdir = data_dir(cfg, "models")
    main_run = seed == 42 and split == "random"
    for m in models:
        out = pred_path(cfg, m, split, seed)
        if out.exists() and not force:
            log.info("[%s] prediction cache hit: %s", m, out.name)
            continue
        set_seed(seed)
        log.info("[%s] training on %d sessions (seed %d, split %s)", m, len(train_s), seed, split)
        extra = {}
        if m in ("rf", "rf_fs"):
            suffix = "" if main_run else f"_{split}_seed{seed}"
            fig = res / "figures" / f"dendrogram_{cfg['name']}{suffix}.png" if m == "rf_fs" else None
            model = RFMCV(templates, cfg["rf"], seed, cfg["feature_selection"] if m == "rf_fs" else None, fig, log)
            t0 = time.time()
            model.fit(train_s)
            train_time = time.time() - t0
            stats = {}
            scores = model.predict_scores(test_s, stats)
            extra["mcv_test_stats"] = stats
            if m == "rf_fs":
                model.fs_report.to_csv(res / f"feature_selection_{cfg['name']}{suffix}.csv", index=False)
                write_json(res / f"feature_selection_{cfg['name']}{suffix}_info.json", model.fs_info)
            model.save(mdir / f"{m}_{split}_seed{seed}.joblib")
            wp = ws = None
        else:
            E = np.load(data_dir(cfg, "features") / "bert_embeddings.npy")
            model = LSTMBert(E, cfg["window"], cfg["lstm"], seed, log)
            t0 = time.time()
            model.fit(train_s)
            train_time = time.time() - t0
            scores, wp, ws = model.predict_scores(test_s, return_windows=True)
            extra.update(best_epoch=model.best_epoch, history=model.history)
            model.save(mdir / f"{m}_{split}_seed{seed}.pt")
        np.savez(out, scores=scores, labels=test_s.labels, test_idx=te,
                 window_probs=wp if wp is not None else np.array([]),
                 window_session=ws if ws is not None else np.array([]))
        write_json(out.with_suffix(".json"), dict(model=m, seed=seed, split=split, train_time_s=train_time,
                                                  n_features=int(model.n_features), n_train=len(train_s),
                                                  n_test=len(test_s), **extra))
        log.info("[%s] done in %.1fs, %d features", m, train_time, model.n_features)


def stage_eval(cfg, seed, split, models, log):
    res = results_dir(cfg)
    df = None
    for m in models:
        p = pred_path(cfg, m, split, seed)
        if not p.exists():
            log.warning("[%s] no predictions for seed %d (%s); skipped", m, seed, split)
            continue
        z = np.load(p)
        meta = read_json(p.with_suffix(".json"))
        met = compute_metrics(z["labels"], z["scores"])
        model, rep, fs = MODEL_ID[m]
        row = dict(dataset=cfg["name"], model=model, representation=rep, feature_selection=fs, split=split,
                   seed=seed, n_features=meta["n_features"], train_time_s=round(meta["train_time_s"], 2),
                   n_train=meta["n_train"], n_test=meta["n_test"], n_test_anomalous=int(z["labels"].sum()), **met)
        df = upsert_metrics(res / "metrics.csv", row)
        log.info("[%s] P %.3f R %.3f F1 %.3f AUC %.3f (tp %d fp %d tn %d fn %d)", m, met["precision"], met["recall"],
                 met["f1"], met["auc"], met["tp"], met["fp"], met["tn"], met["fn"])
        if m == "lstm" and len(z["window_probs"]):
            # supplementary: window-level metrics as computed in the replication package's evaluation()
            wl = z["labels"][z["window_session"]]
            wm = compute_metrics(wl, z["window_probs"])
            write_json(p.with_name(p.stem + "_window_level.json"), wm)
            log.info("[lstm] window-level (package-style) P %.3f R %.3f F1 %.3f", wm["precision"], wm["recall"], wm["f1"])
    if df is not None:
        comparison_with_paper(df).to_csv(res / "comparison_with_paper.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="hdfs | bgl | path to a config yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--split-method", default="random", choices=["random", "loglizer_uniform"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lstm-test-curve", action="store_true",
                    help="diagnostic: log LSTM test metrics per epoch (does not touch metrics.csv)")
    ap.add_argument("--lstm-package-faithful", action="store_true",
                    help="ablation: LSTM with the package's windowing/padding/per-session steps, test curve logged")
    ap.add_argument("--lstm-package-windows", action="store_true",
                    help="ablation: package windowing + padding only (mini-batch training), test curve logged")
    a = ap.parse_args()
    cfg = load_config(a.dataset)
    log = get_logger("part1", f"logs/{cfg['name']}_seed{a.seed}.log")
    log.info("=== %s | seed %d | stage %s | model %s | split %s", cfg["name"], a.seed, a.stage, a.model, a.split_method)
    set_seed(a.seed)
    if a.lstm_test_curve or a.lstm_package_faithful or a.lstm_package_windows:
        lstm_test_curve(cfg, a.seed, a.split_method, log, package=a.lstm_package_faithful,
                        package_windows=a.lstm_package_windows)
        return
    models = MODELS if a.model == "all" else [a.model]
    stages = STAGES if a.stage == "all" else [a.stage]
    if "parse" in stages:
        parse_file(cfg, log, force=a.force, n_workers=a.workers)
    if "partition" in stages:
        partition(cfg, log, force=a.force)
    if "split" in stages:
        get_split(cfg, a.seed, log, a.split_method, force=a.force)
    if "features" in stages:
        stage_features(cfg, a.seed, a.split_method, log, force=a.force)
    if "train" in stages:
        stage_train(cfg, a.seed, a.split_method, models, log, force=a.force)
    if "eval" in stages:
        stage_eval(cfg, a.seed, a.split_method, models, log)


if __name__ == "__main__":
    main()
