"""Part 2 driver: retrain and score every model on each resample (seed) of HDFS.

  python -m src.run_resamples --seeds 0-9 --models rf,lstm,pca,loggpt [--config configs/part2_hdfs.yaml] [--force]

Models: rf, lstm (Part 1 code and configuration), pca (loglizer port), loggpt (LogGPT port; also writes
the unranked ablation loggpt_norl), pca_normal5k (sensitivity: PCA trained on LogGPT's 5,000 normal
sessions; only for `pca_sensitivity_seeds`).

Per (model, seed) the driver writes results/scores/{model}_seed{seed}.parquet (one row per test session:
session_idx, session_id, label, score, prediction) and a sidecar .json (times, device, model info), then
rebuilds results/runs.csv from all sidecars. A (model, seed) whose files exist is skipped (--force redoes it).
Every model of a seed is scored on exactly the same test sessions: the full 30% split (asserted).
"""
import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import p1  # noqa: F401  (registers Part 1 as `p1`)
from .p1 import PART1_ROOT, PART2_ROOT
from .scores import LABELS_USED, RANKED, SCORE_DEF, decide, session_metrics
from p1.common import data_dir as p1_data_dir
from p1.common import get_logger, read_json, set_seed, write_json
from p1.common import load_config as load_p1_config
from p1.partition import Sessions, load_sessions
from p1.split import get_split

RUN_COLS = ["seed", "model", "n_train_sessions", "labels_used", "precision", "recall", "f1", "auc",
            "tp", "fp", "tn", "fn", "train_time_s", "infer_time_s", "device",
            "n_test", "n_test_anomalous", "threshold", "ranked", "concurrent_processes"]
ALL_MODELS = ["rf", "lstm", "pca", "loggpt", "pca_normal5k"]


# ---------------------------------------------------------------------------- config / paths
def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    p = Path(path)
    if not p.is_absolute():
        p = PART2_ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        cfg = _merge(load_config(cfg.pop("base")), cfg)
    cfg["_config_path"] = str(p)
    return cfg


def p2path(cfg, key, *parts):
    d = PART2_ROOT / cfg[key]
    d = d.joinpath(*parts) if parts else d
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return f"cpu:{line.split(':', 1)[1].strip()} ({os.cpu_count()} threads)"
    except OSError:
        pass
    return f"cpu:{platform.processor() or 'unknown'} ({os.cpu_count()} threads)"


def gpu_name():
    import torch
    return f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else cpu_name()


def unlabeled(sessions):
    """Copy whose labels are all -1: proves that an unsupervised model never reads them."""
    return Sessions(sessions.offsets, sessions.events, np.full(len(sessions), -1, dtype=np.int8), sessions.keys)


# ---------------------------------------------------------------------------- context per seed
class Ctx:
    def __init__(self, cfg, seed, log):
        self.cfg, self.seed, self.log = cfg, seed, log
        self.cfg1 = load_p1_config(cfg["part1_config"])
        self.templates = pd.read_csv(p1_data_dir(self.cfg1, "parsed") / "templates.csv", keep_default_na=False)
        self.s = load_sessions(self.cfg1)
        tr, te = get_split(self.cfg1, seed, log)          # Part 1's split function (cached in part1/data/splits)
        if cfg.get("subset"):
            rng = np.random.default_rng(seed)
            tr = np.sort(rng.choice(tr, int(len(tr) * cfg["subset"]), replace=False))
            te = np.sort(rng.choice(te, int(len(te) * cfg["subset"]), replace=False))
        self.tr, self.te = tr, te
        self.train_s, self.test_s = self.s.subset(tr), self.s.subset(te)
        self.test_ids = self.s.keys[te]
        self.scores_dir = p2path(cfg, "results_dir", "scores")
        self.models_dir = p2path(cfg, "data_root", "models")
        log.info("seed %d: train %d (%d anomalous) / test %d (%d anomalous)", seed, len(tr),
                 int(self.train_s.labels.sum()), len(te), int(self.test_s.labels.sum()))

    def normal_sample(self):
        """LogGPT's training sessions: `train_samples` normal sessions of the train split, sampled with the
        resample seed (labels are used only for this filter). Cached so PCA's sensitivity run reuses them."""
        path = p2path(self.cfg, "data_root", "loggpt", f"seed{self.seed}") / "train_sessions.json"
        n = self.cfg["loggpt"]["train_samples"]
        if path.exists():
            d = read_json(path)
            assert d["n"] == n and d["n_train_split"] == len(self.tr)
            return np.array(d["session_idx"])
        normal_tr = self.tr[self.s.labels[self.tr] == 0]
        pick = np.random.default_rng(self.seed).choice(normal_tr, size=n, replace=False)   # sampled order kept
        write_json(path, dict(seed=self.seed, n=n, n_train_split=len(self.tr), n_normal_in_train_split=len(normal_tr),
                              session_idx=pick.tolist(), session_id=[str(k) for k in self.s.keys[pick]]))
        return pick


# ---------------------------------------------------------------------------- models
def run_rf(ctx):
    from p1.models.rf import RFMCV
    model = RFMCV(ctx.templates, ctx.cfg1["rf"], ctx.seed)
    t0 = time.time()
    model.fit(ctx.train_s)
    t_train = time.time() - t0
    t0 = time.time()
    scores = model.predict_scores(ctx.test_s)
    t_inf = time.time() - t0
    model.save(ctx.models_dir / f"rf_seed{ctx.seed}.joblib")
    return [dict(model="rf", scores=scores, threshold=0.5, train_time_s=t_train, infer_time_s=t_inf,
                 device=cpu_name(), n_train_sessions=len(ctx.train_s), info=dict(n_features=int(model.n_features)))]


def run_lstm(ctx):
    from p1.models.lstm import LSTMBert
    E = np.load(p1_data_dir(ctx.cfg1, "features") / "bert_embeddings.npy")
    params = _merge(ctx.cfg1["lstm"], ctx.cfg.get("lstm_overrides", {}))
    model = LSTMBert(E, ctx.cfg1["window"], params, ctx.seed, ctx.log)
    t0 = time.time()
    model.fit(ctx.train_s)
    t_train = time.time() - t0
    t0 = time.time()
    scores = model.predict_scores(ctx.test_s)
    t_inf = time.time() - t0
    model.save(ctx.models_dir / f"lstm_seed{ctx.seed}.pt")
    return [dict(model="lstm", scores=scores, threshold=0.5, train_time_s=t_train, infer_time_s=t_inf,
                 device=gpu_name(), n_train_sessions=len(ctx.train_s),
                 info=dict(best_epoch=model.best_epoch, history=model.history))]


def _run_pca(ctx, name, train_sessions):
    import joblib
    from .models.pca import PCAMCV
    model = PCAMCV(len(ctx.templates), ctx.cfg["pca"], ctx.seed, ctx.log)
    t0 = time.time()
    model.fit(unlabeled(train_sessions))
    t_train = time.time() - t0
    t0 = time.time()
    scores = model.predict_scores(ctx.test_s)
    t_inf = time.time() - t0
    joblib.dump(model, ctx.models_dir / f"{name}_seed{ctx.seed}.joblib", compress=3)
    return [dict(model=name, scores=scores, threshold=model.threshold, train_time_s=t_train, infer_time_s=t_inf,
                 device=cpu_name(), n_train_sessions=len(train_sessions), info=model.info())]


def run_pca(ctx):
    return _run_pca(ctx, "pca", ctx.train_s)


def run_pca_normal5k(ctx):
    return _run_pca(ctx, "pca_normal5k", ctx.s.subset(ctx.normal_sample()))


def run_loggpt(ctx):
    import torch
    from .models.loggpt import LogGPT
    pick = ctx.normal_sample()
    normal = ctx.s.subset(pick)
    assert int(normal.labels.sum()) == 0 and np.isin(pick, ctx.tr).all() and not np.isin(pick, ctx.te).any()
    ckpt = p2path(ctx.cfg, "data_root", "loggpt", f"seed{ctx.seed}")
    p = ctx.cfg["loggpt"]
    model = LogGPT(p, ctx.seed, "cuda" if torch.cuda.is_available() else "cpu", ctx.log, ckpt)
    model.fit(normal)
    out = []
    for which, name, t_train in (("pre", "loggpt_norl", model.times["pretrain_s"]),
                                 ("ft", "loggpt", model.times["pretrain_s"] + model.times["rl_s"])):
        t0 = time.time()
        scores = model.max_rank_scores(ctx.test_s, which, p["score_batch_size"])
        t_inf = time.time() - t0
        verify = verify_loggpt(ctx, model, which, scores)
        out.append(dict(model=name, scores=scores, threshold=model.K, train_time_s=t_train, infer_time_s=t_inf,
                        device=gpu_name(), n_train_sessions=len(normal),
                        info=dict(verify=verify, times=model.times, pre_history=model.pre_history,
                                  rl_history=model.rl_history,
                                  **{k: v for k, v in model.meta.items() if k != "itos"}, itos=model.vocab.itos)))
    model.save(ctx.models_dir / f"loggpt_seed{ctx.seed}.pt")
    return out


def verify_loggpt(ctx, model, which, scores, n_each=150):
    """Brief 5.4: on a few hundred test sessions, `score > K` must equal the original code's binary output."""
    rng = np.random.default_rng(ctx.seed)
    y = ctx.test_s.labels
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.concatenate([rng.choice(pos, min(n_each, len(pos)), replace=False),
                          rng.choice(neg, min(n_each, len(neg)), replace=False)])
    # add up to 50 sessions that contain a template unseen in LogGPT's training sessions (<unk>)
    known = np.zeros(len(ctx.templates) + 1, dtype=bool)
    known[[int(t) for t in model.vocab.itos[5:]]] = True
    n_unknown = np.add.reduceat((~known[ctx.test_s.events]).astype(np.int64), ctx.test_s.offsets[:-1])
    unk = np.flatnonzero(n_unknown > 0)
    idx = np.unique(np.concatenate([idx, unk[:50]]))
    sub = ctx.test_s.subset(idx)
    repo = model.repo_predict_topk(sub, which)
    K = model.K
    # (a) the batched implementation with the original's <unk> handling must reproduce it exactly
    asrepo = model.max_rank_scores(sub, which, unk_not_in_topk=False)
    mis_impl = int(((asrepo > K).astype(int) != repo).sum())
    # (b) the official score (<unk> = rank V + 1, brief rule) must agree on every session without <unk>
    ours = (scores[idx] > K).astype(int)
    has_unk = np.isin(idx, unk)
    mis_no_unk = int((ours[~has_unk] != repo[~has_unk]).sum())
    mis_unk = int((ours[has_unk] != repo[has_unk]).sum())
    # (c) on the whole test set: sessions whose decision changes because of the <unk> rule
    full_asrepo = model.max_rank_scores(ctx.test_s, which, unk_not_in_topk=False)
    n_rule = int(((full_asrepo > K) != (scores > K)).sum())
    res = dict(n_checked=len(idx), n_checked_with_unk=int(has_unk.sum()), n_test_with_unk=int(len(unk)),
               mismatch_batched_vs_original=mis_impl, mismatch_without_unk=mis_no_unk,
               mismatch_with_unk_due_to_rule=mis_unk, n_test_decisions_changed_by_unk_rule=n_rule,
               repo_positive=int(repo.sum()), ours_positive=int(ours.sum()))
    tag = "loggpt" if which == "ft" else "loggpt_norl"
    ctx.log.info("[%s] verification on %d test sessions (%d with <unk>): batched vs original %d mismatches; "
                 "official score vs original: %d mismatches without <unk>, %d with <unk> (rule); "
                 "<unk> rule changes %d of %d test decisions", tag, len(idx), int(has_unk.sum()), mis_impl,
                 mis_no_unk, mis_unk, n_rule, len(scores))
    if mis_impl or mis_no_unk:
        ctx.log.warning("[%s] LogGPT score does NOT reproduce the original predictions", tag)
    return res


RUNNERS = {"rf": run_rf, "lstm": run_lstm, "pca": run_pca, "loggpt": run_loggpt, "pca_normal5k": run_pca_normal5k}
OUTPUTS = {"loggpt": ["loggpt", "loggpt_norl"]}


# ---------------------------------------------------------------------------- outputs
def write_result(ctx, r):
    name = r["model"]
    scores = np.asarray(r["scores"], dtype=np.float64)
    pred = decide(name, scores, r["threshold"])
    assert len(scores) == len(ctx.te)
    df = pd.DataFrame(dict(session_idx=ctx.te.astype(np.int64), session_id=ctx.test_ids.astype(str),
                           label=ctx.test_s.labels.astype(np.int8), score=scores, prediction=pred))
    df.to_parquet(ctx.scores_dir / f"{name}_seed{ctx.seed}.parquet", index=False)
    met = session_metrics(df.label.values, scores, pred)
    meta = dict(seed=ctx.seed, model=name, n_train_sessions=r["n_train_sessions"], labels_used=LABELS_USED[name],
                train_time_s=r["train_time_s"], infer_time_s=r["infer_time_s"], device=r["device"],
                n_test=len(df), n_test_anomalous=int(df.label.sum()), threshold=float(r["threshold"]),
                ranked=name in RANKED, score_definition=SCORE_DEF[name], subset=ctx.cfg.get("subset"),
                concurrent_processes=ctx.cfg.get("_concurrent", 1),
                metrics=met, info=r["info"])
    write_json(ctx.scores_dir / f"{name}_seed{ctx.seed}.json", meta)
    ctx.log.info("[%s] seed %d: P %.4f R %.4f F1 %.4f AUC %.4f (tp %d fp %d tn %d fn %d) | train %.1fs infer %.1fs",
                 name, ctx.seed, met["precision"], met["recall"], met["f1"], met["auc"], met["tp"], met["fp"],
                 met["tn"], met["fn"], r["train_time_s"], r["infer_time_s"])


def check_same_test_sets(cfg, seed, test_ids, log):
    """F1: every model of this seed was scored on exactly the same test sessions."""
    d = p2path(cfg, "results_dir", "scores")
    files = sorted(d.glob(f"*_seed{seed}.parquet"))
    for f in files:
        ids = pd.read_parquet(f, columns=["session_id"]).session_id.values
        assert len(ids) == len(test_ids) and (ids == test_ids.astype(str)).all(), f"test set mismatch: {f.name}"
    log.info("seed %d: identical test sessions (%d) in %d score files", seed, len(test_ids), len(files))
    return len(files)


def rebuild_runs(cfg):
    d = p2path(cfg, "results_dir", "scores")
    rows = []
    for f in sorted(d.glob("*_seed*.json")):
        if f.stem.endswith("_verify"):
            continue
        m = read_json(f)
        rows.append({"concurrent_processes": 1, **{k: m[k] for k in RUN_COLS if k in m}, **m["metrics"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)[RUN_COLS]
    order = {k: i for i, k in enumerate(RANKED + ["loggpt_norl", "pca_normal5k"])}
    df = df.sort_values(["seed", "model"], key=lambda s: s.map(order) if s.name == "model" else s).reset_index(drop=True)
    df.to_csv(p2path(cfg, "results_dir") / "runs.csv", index=False)
    return df


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2_hdfs.yaml")
    ap.add_argument("--seeds", default=None, help="e.g. 0-9 or 0,3,5 (default: config seeds)")
    ap.add_argument("--models", default="rf,lstm,pca,loggpt,pca_normal5k")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concurrent", type=int, default=1,
                    help="number of run_resamples processes sharing the machine (recorded with the timings, F7)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["_concurrent"] = a.concurrent
    seeds = parse_seeds(a.seeds) if a.seeds else cfg["seeds"]
    models = [m.strip() for m in a.models.split(",")]
    assert all(m in ALL_MODELS for m in models), models
    logs = p2path(cfg, "logs_dir")
    for seed in seeds:
        log = get_logger(f"part2.seed{seed}", logs / f"seed{seed}.log")
        log.info("=== config %s | seed %d | models %s", Path(cfg["_config_path"]).name, seed, models)
        ctx = None
        for m in models:
            if m == "pca_normal5k" and seed not in cfg.get("pca_sensitivity_seeds", []):
                continue
            outs = OUTPUTS.get(m, [m])
            done = all((p2path(cfg, "results_dir", "scores") / f"{o}_seed{seed}.parquet").exists() and
                       (p2path(cfg, "results_dir", "scores") / f"{o}_seed{seed}.json").exists() for o in outs)
            if done and not a.force:
                log.info("[%s] seed %d: cached, skipped", m, seed)
                continue
            if ctx is None:
                ctx = Ctx(cfg, seed, log)
            set_seed(seed)
            log.info("[%s] seed %d: training", m, seed)
            for r in RUNNERS[m](ctx):
                write_result(ctx, r)
            rebuild_runs(cfg)
        if ctx is None:
            ctx = Ctx(cfg, seed, log)
        check_same_test_sets(cfg, seed, ctx.test_ids, log)
    rebuild_runs(cfg)


if __name__ == "__main__":
    main()
