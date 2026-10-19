"""RQ1 replication - deep learning supervised model: LSTM on TF-IDF (Text) event vectors.

Usage: python src/run_lstm.py --dataset HDFS|BGL [--seed 0] [--select none|vif] [--epochs N]

Following Wu et al.: sessions are cut into fixed-length sliding windows (HDFS 30/1, BGL 50/50; shorter
sessions are left-padded), each window takes its session's label, and a session's score is the max anomaly
probability over its windows. Network: LSTM(hidden=8) -> FC(2) -> softmax.
Class imbalance: class-weighted cross-entropy (the paper used an imbalanced sampler).
--select vif: correlation + VIF analysis on the token features (fitted on the train sessions' mean vectors),
the LSTM then only sees the kept tokens.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import ROOT, evaluate, save_result  # noqa: E402
from feature_selection import select_features  # noqa: E402
from split import load_split  # noqa: E402
from tfidf_text import event_vectors  # noqa: E402

WINDOW = {"HDFS": (30, 1), "BGL": (50, 50)}
EPOCHS = {"HDFS": 5, "BGL": 30}
BATCH = 1024


def make_windows(events, size, stride, pad_id):
    """-> windows (n, size) of event ids, session index of each window."""
    wins, owner = [], []
    for s, seq in enumerate(events):
        if len(seq) < size:
            seq = np.concatenate([np.full(size - len(seq), pad_id, np.int32), seq])
        starts = range(0, len(seq) - size + 1, stride)
        last = len(seq) - size
        if starts[-1] != last:          # make sure the tail of the session is covered
            starts = list(starts) + [last]
        for st in starts:
            wins.append(seq[st:st + size])
        owner += [s] * len(starts)
    return np.stack(wins).astype(np.int64), np.array(owner)


class LSTMClassifier(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(E), freeze=True)
        self.lstm = nn.LSTM(E.shape[1], 8, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out[:, -1])


def session_scores(model, X, owner, n_sessions):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            probs.append(torch.softmax(model(torch.from_numpy(X[i:i + 8192])), 1)[:, 1].numpy())
    scores = np.zeros(n_sessions)
    np.maximum.at(scores, owner, np.concatenate(probs))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(WINDOW))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", default="none", choices=["none", "vif"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    ds, epochs = args.dataset, args.epochs or EPOCHS[args.dataset]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sessions = pd.read_pickle(os.path.join(ROOT, "data", "sessions", f"{ds}_sessions.pkl"))
    tr, te = load_split(ds, args.seed)
    E, names = event_vectors(ds, sessions, tr)
    pad_id = len(E) - 1

    report = None
    if args.select == "vif":
        # rows = sequence-level (mean) vectors of the train sessions, columns = tokens
        agg = np.stack([E[seq].mean(0) for seq in sessions["events"].to_numpy()[tr]])
        cols, report = select_features(agg, names)
        E = np.ascontiguousarray(E[:, cols])

    size, stride = WINDOW[ds]
    ev, y = sessions["events"].to_numpy(), sessions["label"].to_numpy()
    Xtr, own_tr = make_windows(ev[tr], size, stride, pad_id)
    Xte, own_te = make_windows(ev[te], size, stride, pad_id)
    ytr = y[tr][own_tr].astype(np.int64)
    print(f"{ds}: {E.shape[1]} token features, {len(Xtr):,} train windows ({ytr.mean():.1%} anomalous), "
          f"{len(Xte):,} test windows", flush=True)

    model = LSTMClassifier(E)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2).clip(min=1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm, total = np.random.permutation(len(Xtr)), 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(Xtr[b])), torch.from_numpy(ytr[b]))
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"  epoch {ep + 1}/{epochs} loss {total / len(perm):.4f} ({time.time() - t0:.0f}s)", flush=True)
    train_time = time.time() - t0

    scores = session_scores(model, Xte, own_te, len(te))
    res = {"model": "LSTM", "representation": "TF-IDF (Text)", "dataset": ds, "seed": args.seed,
           "select": args.select, "n_features": int(E.shape[1]), "epochs": epochs,
           "train_time_s": round(train_time, 1), **evaluate(y[te], scores), "selection_report": report}
    out = save_result(f"LSTM_TFIDF_{ds}_{args.select}_seed{args.seed}", res)
    print(f"LSTM {ds} select={args.select}: features={res['n_features']} P={res['precision']} "
          f"R={res['recall']} F1={res['f1']} AUC={res['auc']} ({train_time:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
