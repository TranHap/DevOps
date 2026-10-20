"""Shared helpers: paths, config loading, seeding, device selection, logging."""
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

PART1_ROOT = Path(__file__).resolve().parents[1]

# Keep HuggingFace downloads off the (nearly full) home partition.
os.environ.setdefault("HF_HOME", str(PART1_ROOT / ".hf_cache"))


def load_config(path_or_name):
    """Load a YAML config. Accepts a path or a short name such as 'hdfs'."""
    p = Path(path_or_name)
    if not p.exists():
        p = PART1_ROOT / "configs" / f"{path_or_name}.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    return cfg


def resolve(path):
    """Resolve a path relative to part1/ unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else PART1_ROOT / p


def data_dir(cfg, kind):
    """data/<kind>/<name>/ for kind in parsed, sessions, features, models, predictions."""
    d = resolve(cfg.get("data_root", "data")) / kind / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(cfg):
    d = resolve(cfg.get("results_dir", "results"))
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def splits_dir(cfg):
    d = resolve(cfg.get("data_root", "data")) / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(log=None):
    """CUDA -> MPS -> CPU."""
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, name = torch.device("mps"), "Apple MPS"
    else:
        dev, name = torch.device("cpu"), "CPU"
    if log:
        log.info("Device: %s (%s)", dev, name)
    return dev


def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        log_file = resolve(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(getattr(h, "baseFilename", None) == str(log_file) for h in logger.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(fh)
    return logger


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
