"""Step 1 - Log parsing: raw log -> event templates (Drain + regex preprocessing).

Usage:
    python src/parse.py --dataset HDFS
    python src/parse.py --dataset BGL [--limit 200000]

Outputs (in data/parsed/<dataset>/):
    <dataset>_events.csv     one row per log line, only the columns needed downstream
                             HDFS: LineId, BlockIds, EventId
                             BGL : LineId, Timestamp, Label, EventId
    <dataset>_templates.csv  EventId, EventTemplate, Occurrences
    <dataset>_sample.csv     first 2000 lines with Content + template, for manual inspection
"""
import argparse
import csv
import os
import re
import time

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Header formats from Loghub / logparser benchmark
LOG_FORMATS = {
    "HDFS": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "BGL": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
}

# Preprocessing: mask dynamic variables before Drain sees them (in the spirit of Qin et al., SANER 2025).
# Order matters: specific patterns first, generic numbers last.
COMMON_REGEX = [
    r"blk_-?\d+",                                         # HDFS block id
    r"(/|)(\d{1,3}\.){3}\d{1,3}(:\d+)?",                  # IP (+port)
    r"0[xX][0-9a-fA-F]+",                                 # hex with 0x
    r"(?<![\w/])(/[\w.\-]+){2,}/?",                       # file paths
]
DATASET_REGEX = {
    "HDFS": [],
    "BGL": [
        # Added by us after inspecting rare templates
        r"([0-9A-Fa-f]{2}:){5,}[0-9A-Fa-f]{2}",           # iDo chip / LP ids  FF:F2:9F:16:E1:...
        r"R\d{2}-M\d[\w:-]*",                            # node location     R02-M1-N0-C:J12-U11
        # Refined regexes reported in Wu et al.'s replication package
        r"core\.\d+",
        r"(?<=r)\d{1,2}",
        r"(?<=fpr)\d{1,2}",
        r"(0x)?[0-9a-fA-F]{8}",
        r"(?<=\.\.)\d+(?!x)",
        r"\d+(?=:)",
        r"(?<=\=)\d+(?!x)",
    ],
}
NUMBER_REGEX = r"(?<![A-Za-z_<])-?\d+(\.\d+)?(?![A-Za-z_>])"  # standalone numbers

DRAIN_PARAMS = {"HDFS": dict(depth=4, sim_th=0.5), "BGL": dict(depth=4, sim_th=0.5)}
BLOCK_RE = re.compile(r"blk_-?\d+")


def header_regex(log_format):
    parts = re.split(r"(<[^<>]+>)", log_format)
    headers, regex = [], ""
    for p in parts:
        if p.startswith("<") and p.endswith(">"):
            name = p.strip("<>")
            headers.append(name)
            regex += f"(?P<{name}>.*?)" if name != "Content" else f"(?P<{name}>.*)"
        else:
            regex += r"\s+".join(re.escape(s) for s in p.split(" "))
    return headers, re.compile("^" + regex + "$")


def build_masker(dataset):
    patterns = [re.compile(p) for p in COMMON_REGEX + DATASET_REGEX[dataset] + [NUMBER_REGEX]]

    def mask(text):
        for p in patterns:
            text = p.sub("<*>", text)
        # Merge adjacent wildcards: "<*><*>" -> "<*>", and runs of whole wildcard tokens (variable-length
        # lists, e.g. "delete blk_1 blk_2 ...") -> one "<*>". Never insert whitespace next to punctuation
        # ("<*>:" must stay one token): that changes Drain's token count (GA on BGL-2k fell 0.970 -> 0.918).
        text = re.sub(r"(<\*>)+", "<*>", text)
        text = re.sub(r"(?<!\S)<\*>(\s+<\*>)+(?!\S)", "<*>", text)
        return text.strip()
    return mask


WILDCARD_ONLY = re.compile(r"(<\*>\s*)*")
WILDCARD_EVENT_ID = 0   # drain3 gives similarity 0 to all-wildcard messages, so each would get its own
                        # cluster; we route them to one fixed template instead


def make_miner(ds):
    cfg = TemplateMinerConfig()
    cfg.drain_depth = DRAIN_PARAMS[ds]["depth"]
    cfg.drain_sim_th = DRAIN_PARAMS[ds]["sim_th"]
    cfg.drain_max_children = 100
    cfg.masking_instructions = []          # masking is done by our own preprocessing
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def make_content_parser(ds, preprocess=True):
    """Returns parse(content) -> EventId, and a function giving the final {EventId: (template, size)}."""
    miner = make_miner(ds)
    mask = build_masker(ds) if preprocess else (lambda s: s)
    n_wild = [0]

    def parse(content):
        masked = mask(content)
        if WILDCARD_ONLY.fullmatch(masked):
            n_wild[0] += 1
            return WILDCARD_EVENT_ID
        return miner.add_log_message(masked)["cluster_id"]

    def templates():
        t = {c.cluster_id: (c.get_template(), c.size) for c in miner.drain.clusters}
        if n_wild[0]:
            t[WILDCARD_EVENT_ID] = ("<*>", n_wild[0])
        return t
    return parse, templates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOG_FORMATS))
    ap.add_argument("--limit", type=int, default=None, help="parse only the first N lines (for testing)")
    args = ap.parse_args()
    ds = args.dataset

    in_path = os.path.join(ROOT, "data", "raw", ds, f"{ds}.log")
    out_dir = os.path.join(ROOT, "data", "parsed", ds)
    os.makedirs(out_dir, exist_ok=True)

    parse, get_templates = make_content_parser(ds)
    _, line_re = header_regex(LOG_FORMATS[ds])

    events_path = os.path.join(out_dir, f"{ds}_events.csv")
    sample_rows = []
    n, skipped, t0 = 0, 0, time.time()
    with open(in_path, encoding="utf-8", errors="replace") as fin, \
         open(events_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["LineId", "BlockIds", "EventId"] if ds == "HDFS" else ["LineId", "Timestamp", "Label", "EventId"])
        for line in fin:
            n += 1
            if args.limit and n > args.limit:
                n -= 1
                break
            line = line.rstrip("\r\n")
            # Some BGL lines have an empty Content field; pad so the header still matches
            m = line_re.match(line) or line_re.match(line + " ")
            if not m:
                skipped += 1
                continue
            content = m.group("Content") or "<EMPTY>"
            eid = parse(content)
            if ds == "HDFS":
                w.writerow([n, ";".join(sorted(set(BLOCK_RE.findall(content)))), eid])
            else:
                # Loghub BGL: Label '-' = normal, anything else = alert (anomaly)
                w.writerow([n, m.group("Timestamp"), 0 if m.group("Label") == "-" else 1, eid])
            if len(sample_rows) < 2000:
                sample_rows.append((n, content, eid))
            if n % 500_000 == 0:
                print(f"  {n:,} lines, {n / (time.time() - t0):,.0f} lines/s", flush=True)

    # Templates can still change while Drain learns, so write the FINAL template of each cluster
    templates = get_templates()
    with open(os.path.join(out_dir, f"{ds}_templates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EventId", "EventTemplate", "Occurrences"])
        for eid, (tpl, size) in sorted(templates.items(), key=lambda x: -x[1][1]):
            w.writerow([eid, tpl, size])
    with open(os.path.join(out_dir, f"{ds}_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LineId", "Content", "EventId", "EventTemplate"])
        for lid, content, eid in sample_rows:
            w.writerow([lid, content, eid, templates[eid][0]])

    print(f"Done {ds}: {n:,} lines ({skipped} unmatched), {len(templates)} templates, {time.time() - t0:.0f}s")
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()
