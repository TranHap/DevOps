"""Stage 1: parse a raw log file with Drain (streaming, bounded memory).

Outputs (in data/parsed/<name>/):
  lines.npz            per-line arrays: line_id, template_id (+ HDFS block pairs / BGL label+timestamp)
  templates.csv        TemplateId (1..n, 0 = padding), EventId (md5[:8] as logpai), EventTemplate, Occurrences
  structured.parquet   LineId, timestamp fields, Label (BGL) / BlockIds (HDFS), EventId, EventTemplate, TemplateId
  parse_meta.json      line counts, masking configuration, timing

Usage: python -m src.parse --config hdfs [--force]
"""
import argparse
import hashlib
import time
from array import array
from itertools import islice
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .common import data_dir, get_logger, load_config, resolve, write_json
from .drain import Drain
from .preprocess import QIN_ESTIMATION_LINES, Masker, hdfs_block_ids, logformat_regex

CHUNK_LINES = 20000

# ---- worker side ---------------------------------------------------------
_W = {}


def _init_worker(kind, log_format, masker):
    headers, regex = logformat_regex(log_format)
    _W.update(kind=kind, headers=headers, regex=regex, masker=masker)


def _split_line(line, kind, headers, regex):
    """Return dict of header fields (incl. Content) or None if the line is unusable."""
    line = line.strip()
    m = regex.search(line)
    if m is not None:
        return {h: m.group(h) for h in headers}
    if kind == "bgl":
        # BGL lines whose Content is empty do not match the format regex (it needs a
        # separator before <Content>). Keep them with an empty Content so that their
        # Label/Timestamp still count when grouping into windows.
        parts = line.split(None, len(headers) - 1)
        if len(parts) == len(headers) - 1:
            parts.append("")
        if len(parts) == len(headers):
            return dict(zip(headers, parts))
    return None


def _meta(kind, f):
    if kind == "hdfs":
        return (int(f["Date"]), int(f["Time"]), hdfs_block_ids(f["Content"]))
    if kind == "bgl":
        return (f["Label"], int(f["Timestamp"]))
    return None


def _process_chunk(args):
    start_lineno, lines = args
    kind, headers, regex, masker = _W["kind"], _W["headers"], _W["regex"], _W["masker"]
    out = []
    for i, line in enumerate(lines):
        f = _split_line(line, kind, headers, regex)
        if f is None:
            continue
        out.append((start_lineno + i, masker.mask(f["Content"]), _meta(kind, f)))
    return out


# ---- main side -----------------------------------------------------------
class _Collector:
    """Accumulates per-line outputs in compact arrays."""

    def __init__(self, kind):
        self.kind = kind
        self.line_id = array("i")
        self.cluster = array("i")
        if kind == "hdfs":
            self.date, self.time = array("i"), array("i")
            self.pair_line, self.pair_block = array("i"), array("i")
            self.block_code = {}
        elif kind == "bgl":
            self.ts = array("q")
            self.label = array("h")
            self.label_code = {}

    def add(self, lineno, cluster_idx, meta):
        row = len(self.line_id)
        self.line_id.append(lineno)
        self.cluster.append(cluster_idx)
        if self.kind == "hdfs":
            d, t, blks = meta
            self.date.append(d)
            self.time.append(t)
            for b in blks:
                code = self.block_code.setdefault(b, len(self.block_code))
                self.pair_line.append(row)
                self.pair_block.append(code)
        elif self.kind == "bgl":
            lab, ts = meta
            self.label.append(self.label_code.setdefault(lab, len(self.label_code)))
            self.ts.append(ts)


def finalize_templates(drain, cluster_per_line):
    """logpai semantics: EventId = md5(final template)[:8]; identical templates merge.
    Integer TemplateId 1..n in order of first appearance; 0 is reserved for padding."""
    tmpl = drain.templates()
    eid = [hashlib.md5(t.encode("utf-8")).hexdigest()[0:8] for t in tmpl]
    first_seen = {}
    cluster_first = np.full(len(tmpl), np.iinfo(np.int64).max, dtype=np.int64)
    cl = np.asarray(cluster_per_line, dtype=np.int64)
    # first line index of each cluster
    np.minimum.at(cluster_first, cl, np.arange(len(cl)))
    order = np.argsort(cluster_first, kind="stable")
    for c in order:
        if cluster_first[c] == np.iinfo(np.int64).max:
            continue
        first_seen.setdefault(eid[c], (len(first_seen) + 1, tmpl[c]))
    cluster_to_tid = np.zeros(len(tmpl), dtype=np.int32)
    for c in range(len(tmpl)):
        cluster_to_tid[c] = first_seen[eid[c]][0]
    tid = cluster_to_tid[cl]
    n = len(first_seen)
    occ = np.bincount(tid, minlength=n + 1)
    rows = sorted(((v[0], k, v[1]) for k, v in first_seen.items()))
    templates = pd.DataFrame(rows, columns=["TemplateId", "EventId", "EventTemplate"])
    templates["Occurrences"] = occ[templates["TemplateId"].values]
    return tid.astype(np.int32), templates


def parse_file(cfg, log, force=False, n_workers=None):
    out = data_dir(cfg, "parsed")
    if (out / "lines.npz").exists() and (out / "templates.csv").exists() and not force:
        log.info("Parse cache hit: %s", out)
        return out
    kind = cfg["kind"]
    pp = cfg["preprocessing"]
    if pp in ("qin", "qin_full"):
        masker = Masker("qin", narrow=(pp == "qin"))
    elif pp == "none":
        masker = Masker("none")
    else:
        masker = Masker("regex", cfg["regex"][pp])
    headers, regex = logformat_regex(cfg["log_format"])
    drain = Drain(depth=cfg["drain"]["depth"], st=cfg["drain"]["st"],
                  maxChild=cfg["drain"].get("max_child", 100))
    col = _Collector(kind)
    raw_path = resolve(cfg["raw_log"])
    log.info("Parsing %s (kind=%s, preprocessing=%s)", raw_path, kind, pp)
    t0 = time.time()
    n_raw = 0
    with open(raw_path, "r", encoding="utf-8", errors="replace") as fin:
        # Qin estimation stage on the first 2000 well-formed lines, in the main process.
        first_fields, first_lineno = [], []
        while len(first_fields) < QIN_ESTIMATION_LINES:
            line = fin.readline()
            if not line:
                break
            n_raw += 1
            f = _split_line(line, kind, headers, regex)
            if f is not None:
                first_fields.append(f)
                first_lineno.append(n_raw)
        masked_first = masker.estimate([f["Content"] for f in first_fields])
        for lineno, f, m in zip(first_lineno, first_fields, masked_first):
            col.add(lineno, drain.add(m.strip().split()), _meta(kind, f))
        log.info("Masking config: %s", masker.describe())

        n_workers = n_workers or max(1, min(16, (__import__("os").cpu_count() or 2) - 2))
        with Pool(n_workers, initializer=_init_worker, initargs=(kind, cfg["log_format"], masker)) as pool:
            while True:
                batch = []
                for _ in range(n_workers * 2):
                    lines = list(islice(fin, CHUNK_LINES))
                    if not lines:
                        break
                    batch.append((n_raw + 1, lines))
                    n_raw += len(lines)
                if not batch:
                    break
                for res in pool.map(_process_chunk, batch):
                    for lineno, masked, meta in res:
                        col.add(lineno, drain.add(masked.strip().split()), meta)
                log.info("  %d lines read, %d clusters, %.0fs", n_raw, len(drain.clusters), time.time() - t0)

    n_lines = len(col.line_id)
    tid, templates = finalize_templates(drain, col.cluster)
    templates.to_csv(out / "templates.csv", index=False)
    arrays = dict(line_id=np.frombuffer(col.line_id, dtype=np.int32), template_id=tid)
    if kind == "hdfs":
        names = np.array(sorted(col.block_code, key=col.block_code.get))
        arrays.update(date=np.frombuffer(col.date, dtype=np.int32), time=np.frombuffer(col.time, dtype=np.int32),
                      pair_line=np.frombuffer(col.pair_line, dtype=np.int32),
                      pair_block=np.frombuffer(col.pair_block, dtype=np.int32), block_names=names)
    elif kind == "bgl":
        lnames = np.array(sorted(col.label_code, key=col.label_code.get))
        arrays.update(timestamp=np.frombuffer(col.ts, dtype=np.int64),
                      label=np.frombuffer(col.label, dtype=np.int16), label_names=lnames)
    np.savez(out / "lines.npz", **arrays)
    elapsed = time.time() - t0
    meta = dict(raw_log=str(raw_path), n_raw_lines=n_raw, n_parsed_lines=n_lines,
                n_dropped_lines=n_raw - n_lines, n_drain_clusters=len(drain.clusters),
                n_templates=len(templates), masking=masker.describe(), drain=cfg["drain"],
                log_format=cfg["log_format"], parse_time_s=round(elapsed, 1))
    write_json(out / "parse_meta.json", meta)
    log.info("Parsed %d/%d lines -> %d templates in %.0fs", n_lines, n_raw, len(templates), elapsed)
    write_structured(cfg, out, arrays, templates, log)
    return out


def write_structured(cfg, out, arrays, templates, log, rows_per_group=1_000_000):
    """Structured output as parquet (dictionary-encoded templates, written in row groups)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    kind = cfg["kind"]
    tid = arrays["template_id"]
    n = len(tid)
    tmpl_text = np.empty(len(templates) + 1, dtype=object)
    tmpl_text[0] = ""
    tmpl_text[templates.TemplateId.values] = templates.EventTemplate.values
    eid_text = np.empty(len(templates) + 1, dtype=object)
    eid_text[0] = ""
    eid_text[templates.TemplateId.values] = templates.EventId.values
    tmpl_dict, eid_dict = pa.array(list(tmpl_text)), pa.array(list(eid_text))
    writer = None
    for a in range(0, n, rows_per_group):
        b = min(n, a + rows_per_group)
        cols = {"LineId": pa.array(arrays["line_id"][a:b])}
        if kind == "hdfs":
            cols["Date"] = pa.array(arrays["date"][a:b])
            cols["Time"] = pa.array(arrays["time"][a:b])
            pl, pb, names = arrays["pair_line"], arrays["pair_block"], arrays["block_names"]
            lo, hi = np.searchsorted(pl, [a, b])
            per_line = [[] for _ in range(b - a)]
            for r, c in zip(pl[lo:hi], pb[lo:hi]):
                per_line[r - a].append(names[c])
            cols["BlockIds"] = pa.array([",".join(x) for x in per_line])
        elif kind == "bgl":
            cols["Timestamp"] = pa.array(arrays["timestamp"][a:b])
            cols["Label"] = pa.DictionaryArray.from_arrays(pa.array(arrays["label"][a:b].astype(np.int32)),
                                                           pa.array(list(arrays["label_names"])))
        t = pa.array(tid[a:b].astype(np.int32))
        cols["TemplateId"] = t
        cols["EventId"] = pa.DictionaryArray.from_arrays(t, eid_dict)
        cols["EventTemplate"] = pa.DictionaryArray.from_arrays(t, tmpl_dict)
        table = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(out / "structured.parquet", table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    log.info("Wrote %s", out / "structured.parquet")


def parse_contents(contents, masker, depth, st, max_child=100):
    """In-memory helper used by the 2k parsing evaluation: returns one template per line."""
    drain = Drain(depth=depth, st=st, maxChild=max_child)
    masked = masker.estimate(contents[:QIN_ESTIMATION_LINES]) + [masker.mask(c) for c in contents[QIN_ESTIMATION_LINES:]]
    cl = [drain.add(m.strip().split()) for m in masked]
    tid, templates = finalize_templates(drain, cl)
    text = dict(zip(templates.TemplateId, templates.EventTemplate))
    return [text[t] for t in tid], templates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = get_logger("parse", f"logs/{cfg['name']}_parse.log")
    parse_file(cfg, log, force=a.force, n_workers=a.workers)


if __name__ == "__main__":
    main()
