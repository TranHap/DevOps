"""Stage 3: session-level train/test split (reusable for any seed; Part 2 resamples with it).

method=random (default, paper §4.3.2): shuffle sessions with numpy's default_rng(seed),
    first int(n * train_ratio) are train. Not stratified (the paper does not stratify).
method=loglizer_uniform (sensitivity only): what the replication package's sample split
    actually does -- loglizer `_split_data(split_type='uniform')`: per class, the first
    int(n_c * train_ratio) sessions in order of first appearance go to train. This is
    chronological within each class and does not depend on the seed.

Output: data/splits/<name>_seed<seed>.json (or <name>_<method>_seed<seed>.json)
"""
import argparse

import numpy as np

from .common import get_logger, load_config, read_json, splits_dir, write_json
from .partition import load_sessions


def make_split(labels, train_ratio, seed, method="random"):
    n = len(labels)
    if method == "random":
        perm = np.random.default_rng(seed).permutation(n)
        n_train = int(n * train_ratio)
        train, test = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    elif method == "loglizer_uniform":
        train, test = [], []
        for c in (1, 0):
            idx = np.flatnonzero(labels == c)
            k = int(len(idx) * train_ratio)
            train.append(idx[:k])
            test.append(idx[k:])
        train, test = np.sort(np.concatenate(train)), np.sort(np.concatenate(test))
    else:
        raise ValueError(method)
    return train, test


def split_path(cfg, seed, method="random"):
    tag = "" if method == "random" else f"_{method}"
    return splits_dir(cfg) / f"{cfg['name']}{tag}_seed{seed}.json"


def get_split(cfg, seed, log=None, method="random", force=False):
    path = split_path(cfg, seed, method)
    if path.exists() and not force:
        d = read_json(path)
        return np.array(d["train"]), np.array(d["test"])
    s = load_sessions(cfg)
    train, test = make_split(s.labels, cfg["split"]["train_ratio"], seed, method)
    d = dict(dataset=cfg["name"], seed=seed, method=method, train_ratio=cfg["split"]["train_ratio"],
             n_sessions=len(s), n_train=len(train), n_test=len(test),
             n_anomalous_train=int(s.labels[train].sum()), n_anomalous_test=int(s.labels[test].sum()),
             anomaly_ratio_train=float(s.labels[train].mean()), anomaly_ratio_test=float(s.labels[test].mean()),
             train=train.tolist(), test=test.tolist())
    write_json(path, d)
    if log is not None:
        log.info("Split %s seed=%d: train %d (%.2f%% anomalous) / test %d (%.2f%% anomalous) -> %s",
                 method, seed, len(train), 100 * d["anomaly_ratio_train"], len(test),
                 100 * d["anomaly_ratio_test"], path.name)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default="random", choices=["random", "loglizer_uniform"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    get_split(cfg, a.seed, get_logger("split", f"logs/{cfg['name']}_split.log"), a.method, a.force)


if __name__ == "__main__":
    main()
