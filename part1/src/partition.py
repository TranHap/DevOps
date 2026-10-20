"""Stage 2: group parsed lines into labelled sessions.

HDFS : one session per block id; events in log (time) order; labels from anomaly_label.csv.
       A line that references several blocks is assigned to each of them (loglizer).
BGL  : fixed, non-overlapping windows of `window_hours` on Timestamp, aligned to the first
       timestamp; empty windows dropped; anomalous if any line has Label != '-'.
       Inside a window, events are ordered by (Timestamp, LineId).

Output data/sessions/<name>/sessions.npz with a CSR-like layout:
  offsets [n+1], events [sum len] (template ids, int32), labels [n] (int8), keys [n]
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import data_dir, get_logger, load_config, resolve, write_json


@dataclass
class Sessions:
    offsets: np.ndarray   # int64 [n+1]
    events: np.ndarray    # int32 [total]
    labels: np.ndarray    # int8 [n]
    keys: np.ndarray      # [n] block id / window start

    def __len__(self):
        return len(self.labels)

    @property
    def lengths(self):
        return np.diff(self.offsets)

    def seq(self, i):
        return self.events[self.offsets[i]:self.offsets[i + 1]]

    def subset(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        lens = self.lengths[idx]
        new_off = np.zeros(len(idx) + 1, dtype=np.int64)
        np.cumsum(lens, out=new_off[1:])
        # gather events of the selected sessions without a python loop
        starts = self.offsets[idx]
        pos = np.repeat(starts - new_off[:-1], lens) + np.arange(new_off[-1])
        return Sessions(new_off, self.events[pos], self.labels[idx], self.keys[idx])

    def save(self, path):
        np.savez(path, offsets=self.offsets, events=self.events, labels=self.labels, keys=self.keys)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return Sessions(z["offsets"], z["events"], z["labels"], z["keys"])


def _csr_from_groups(group, order_keys, tids):
    """group: session index per event; sort by (group, order_keys...) -> CSR."""
    order = np.lexsort(tuple(order_keys[::-1]) + (group,)) if order_keys else np.argsort(group, kind="stable")
    g = group[order]
    n = int(g.max()) + 1 if len(g) else 0
    counts = np.bincount(g, minlength=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, tids[order].astype(np.int32)


def partition_hdfs(cfg, z, log):
    tid = z["template_id"]
    pair_line, pair_block, names = z["pair_line"], z["pair_block"], z["block_names"]
    # stable sort by block keeps file (time) order inside each block
    offsets, events = _csr_from_groups(pair_block.astype(np.int64), [], tid[pair_line])
    labels_df = pd.read_csv(resolve(cfg["label_file"]))
    lab = dict(zip(labels_df.BlockId, (labels_df.Label == "Anomaly").astype(np.int8)))
    missing = [b for b in names if b not in lab]
    if missing:
        log.warning("%d blocks have no label; they are dropped", len(missing))
    keep = np.array([b in lab for b in names])
    labels = np.array([lab.get(b, -1) for b in names], dtype=np.int8)
    s = Sessions(offsets, events, labels, names)
    if not keep.all():
        s = s.subset(np.flatnonzero(keep))
    extra = dict(n_label_rows=len(labels_df), n_blocks_in_log=len(names), n_blocks_without_label=len(missing),
                 n_label_rows_not_in_log=int(len(set(labels_df.BlockId) - set(names))))
    return s, extra


def partition_bgl(cfg, z, log):
    tid, ts = z["template_id"], z["timestamp"].astype(np.int64)
    label_names = z["label_names"]
    normal_code = int(np.flatnonzero(label_names == "-")[0])
    is_anom = (z["label"] != normal_code).astype(np.int8)
    width = int(round(cfg["partition"]["window_hours"] * 3600))
    t0 = int(ts.min())
    win = (ts - t0) // width                 # window index of every line
    uniq, group = np.unique(win, return_inverse=True)   # drops empty windows
    line_idx = np.arange(len(ts))
    offsets, events = _csr_from_groups(group.astype(np.int64), [ts, line_idx], tid)
    labels = np.zeros(len(uniq), dtype=np.int8)
    np.maximum.at(labels, group, is_anom)
    keys = (t0 + uniq * width).astype(np.int64)
    n_windows_total = int((ts.max() - t0) // width) + 1
    extra = dict(first_timestamp=t0, last_timestamp=int(ts.max()), window_seconds=width,
                 n_windows_spanned=n_windows_total, n_empty_windows_dropped=n_windows_total - len(uniq),
                 n_anomalous_lines=int(is_anom.sum()),
                 n_non_monotonic_timestamps=int((np.diff(ts) < 0).sum()))
    return Sessions(offsets, events, labels, keys), extra


def partition(cfg, log, force=False):
    out = data_dir(cfg, "sessions")
    path = out / "sessions.npz"
    if path.exists() and not force:
        log.info("Sessions cache hit: %s", path)
        return Sessions.load(path)
    z = np.load(data_dir(cfg, "parsed") / "lines.npz", allow_pickle=False)
    if cfg["kind"] == "hdfs":
        s, extra = partition_hdfs(cfg, z, log)
    else:
        s, extra = partition_bgl(cfg, z, log)
    s.save(path)
    lens = s.lengths
    meta = dict(n_sessions=len(s), n_anomalous=int(s.labels.sum()),
                anomaly_ratio=float(s.labels.mean()), n_events=int(len(s.events)),
                mean_len=float(lens.mean()), median_len=float(np.median(lens)),
                max_len=int(lens.max()), min_len=int(lens.min()), **extra)
    write_json(out / "sessions_meta.json", meta)
    log.info("Sessions: %d (%d anomalous, %.2f%%), mean length %.1f",
             len(s), meta["n_anomalous"], 100 * meta["anomaly_ratio"], meta["mean_len"])
    return s


def load_sessions(cfg):
    return Sessions.load(data_dir(cfg, "sessions") / "sessions.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    partition(cfg, get_logger("partition", f"logs/{cfg['name']}_partition.log"), force=a.force)


if __name__ == "__main__":
    main()
