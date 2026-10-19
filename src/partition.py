"""Step 2a - Log partition: parsed events -> sessions (event sequences + label).

Usage: python src/partition.py --dataset HDFS|BGL

HDFS: one session per block_id; label from Loghub anomaly_label.csv.
BGL : fixed 6-hour windows on the epoch Timestamp column (no timezone issues); a window is anomalous
      if it contains at least one alert line. Empty windows do not exist (no lines -> no session).
Events inside a session keep the original log order (LineId).

Output: data/sessions/<DS>_sessions.pkl  DataFrame[session_id, events (int32 array), label, n_events]
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BGL_WINDOW_SECONDS = 6 * 3600


def group_sequences(keys, values):
    """Group values by key, keeping the input order inside each group.
    Returns (unique keys in first-appearance order, list of int32 arrays)."""
    codes, uniques = pd.factorize(keys, sort=False)
    order = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.diff(codes[order], prepend=-1))
    return uniques, np.split(values[order].astype(np.int32), starts[1:])


def sessions_hdfs():
    e = pd.read_csv(os.path.join(ROOT, "data/parsed/HDFS/HDFS_events.csv"), dtype={"BlockIds": str})
    e = e.dropna(subset=["BlockIds"])
    # A line mentioning several blocks belongs to all of them (none do in HDFS_v1, but stay safe)
    e = e.assign(session_id=e["BlockIds"].str.split(";")).explode("session_id")
    e = e.sort_values("LineId", kind="stable")
    ids, seqs = group_sequences(e["session_id"].to_numpy(), e["EventId"].to_numpy())
    lab = pd.read_csv(os.path.join(ROOT, "data/raw/HDFS/preprocessed/anomaly_label.csv"))
    lab = lab.set_index("BlockId")["Label"].eq("Anomaly").astype(np.int8)
    df = pd.DataFrame({"session_id": ids, "events": seqs})
    df["label"] = df["session_id"].map(lab)
    assert df["label"].notna().all(), "some blocks have no label"
    return df


def sessions_bgl():
    e = pd.read_csv(os.path.join(ROOT, "data/parsed/BGL/BGL_events.csv"))
    e = e.sort_values("LineId", kind="stable")
    t0 = e["Timestamp"].min()
    win = ((e["Timestamp"] - t0) // BGL_WINDOW_SECONDS).to_numpy(np.int64)
    ids, seqs = group_sequences(win, e["EventId"].to_numpy())
    labels = e.groupby(win)["Label"].max().astype(np.int8)
    df = pd.DataFrame({"session_id": ids, "events": seqs})
    df["label"] = df["session_id"].map(labels)
    df["start_time"] = t0 + df["session_id"] * BGL_WINDOW_SECONDS
    return df.sort_values("session_id", ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["HDFS", "BGL"])
    ds = ap.parse_args().dataset

    df = sessions_hdfs() if ds == "HDFS" else sessions_bgl()
    df["n_events"] = df["events"].map(len)
    out = os.path.join(ROOT, "data", "sessions", f"{ds}_sessions.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_pickle(out)

    print(f"{ds}: {len(df):,} sessions, {df['label'].sum():,} anomalous ({df['label'].mean():.2%})")
    print(f"  events/session: mean {df['n_events'].mean():.1f}, median {df['n_events'].median():.0f}, "
          f"max {df['n_events'].max():,}")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
