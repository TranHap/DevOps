"""Extension E1 - deep-learning SEMI-SUPERVISED model: DeepLog (Du et al., CCS'17).

Usage: python src/run_deeplog.py [--dataset HDFS] [--seed 0]

- Trained on NORMAL train sessions only (labels used only to select them).
- Input: the last h=10 event ids (left-padded), target: the next event id. 2-layer LSTM(64) + softmax over
  the train vocabulary (event ids never seen in train map to UNK).
- A position is abnormal if the true next event is not in the top-g (g=9) predictions; a session is
  anomalous if any position is abnormal (the paper's rule).
- Continuous session score for AUC: max over positions of the rank of the true next event
  (0 = most likely). Prediction "rank >= g"  <=>  score >= g, so F1 and AUC use the same score.
- Hyper-parameters (h=10, g=9, 2x64, 5 epochs, Adam 1e-3) follow DeepLog / deep-loglizer, fixed a priori.
- For speed, train on a random sample of 100k normal train sessions (DeepLog's own paper trained on <1%
  of HDFS), and score each distinct (history, next) pair only once.
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
from split import load_split  # noqa: E402

H, G, HIDDEN, LAYERS, EPOCHS, BATCH, LR = 10, 9, 64, 2, 5, 2048, 1e-3
N_TRAIN_SESSIONS = 100_000


def windows(seqs, vocab_map, unk):
    """-> X (n, H) histories, y (n,) next events, owner (n,) session index. Ids: 0 = pad, unk = unseen."""
    X, Y, owner = [], [], []
    for s, seq in enumerate(seqs):
        ids = np.array([vocab_map.get(e, unk) for e in seq], dtype=np.int64)
        padded = np.concatenate([np.zeros(H, np.int64), ids])
        for i in range(len(ids)):
            X.append(padded[i:i + H])
            Y.append(ids[i])
            owner.append(s)
    return np.array(X), np.array(Y), np.array(owner)


class DeepLog(nn.Module):
    def __init__(self, n_keys):
        super().__init__()
        self.emb = nn.Embedding(n_keys, 32)
        self.lstm = nn.LSTM(32, HIDDEN, LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN, n_keys)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out[:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HDFS", choices=["HDFS", "BGL"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sessions = pd.read_pickle(os.path.join(ROOT, "data", "sessions", f"{args.dataset}_sessions.pkl"))
    tr, te = load_split(args.dataset, args.seed)
    ev, y = sessions["events"].to_numpy(), sessions["label"].to_numpy()
    normal_tr = tr[y[tr] == 0]
    sample = rng.choice(normal_tr, min(N_TRAIN_SESSIONS, len(normal_tr)), replace=False)

    vocab = np.unique(np.concatenate(list(ev[normal_tr])))   # events seen in normal train sessions
    vocab_map = {e: i + 1 for i, e in enumerate(vocab)}      # 0 = padding
    unk = len(vocab) + 1
    n_keys = unk + 1

    Xtr, ytr, _ = windows(ev[sample], vocab_map, unk)
    print(f"{args.dataset} seed {args.seed}: {len(vocab)} keys, {len(sample):,} normal sessions -> "
          f"{len(Xtr):,} training windows", flush=True)

    model = DeepLog(n_keys)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm, total = rng.permutation(len(Xtr)), 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(Xtr[b])), torch.from_numpy(ytr[b]))
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"  epoch {ep + 1}/{EPOCHS} loss {total / len(perm):.4f} ({time.time() - t0:.0f}s)", flush=True)
    train_time = time.time() - t0

    # Score test positions; identical (history, next) pairs are scored once
    Xte, yte, own = windows(ev[te], vocab_map, unk)
    pairs = np.concatenate([Xte, yte[:, None]], axis=1)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    model.eval()
    ranks = np.empty(len(uniq), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(uniq), 16384):
            logits = model(torch.from_numpy(uniq[i:i + 16384, :H]))
            true = torch.from_numpy(uniq[i:i + 16384, H])
            true_logit = logits.gather(1, true[:, None])
            ranks[i:i + 16384] = (logits > true_logit).sum(1).numpy()   # 0 = top-1
    ranks[uniq[:, H] == unk] = n_keys          # unseen event: always abnormal
    pos_rank = ranks[inverse.ravel()]
    scores = np.zeros(len(te))
    np.maximum.at(scores, own, pos_rank)

    res = {"model": "DeepLog", "representation": "event id sequence", "dataset": args.dataset,
           "seed": args.seed, "h": H, "g": G, "train_sessions": int(len(sample)),
           "train_time_s": round(train_time, 1), "n_test_positions": int(len(pairs)),
           "n_unique_test_positions": int(len(uniq)),
           **evaluate(y[te], scores, threshold=G - 0.5)}
    out = save_result(f"DeepLog_{args.dataset}_seed{args.seed}", res, subdir="ext")
    os.makedirs(os.path.join(ROOT, "results", "ext", "scores"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "results", "ext", "scores", f"DeepLog_{args.dataset}_seed{args.seed}.npz"),
                        test_idx=te, y=y[te], scores=scores)
    print(f"DeepLog {args.dataset} seed {args.seed}: P={res['precision']} R={res['recall']} F1={res['f1']} "
          f"AUC={res['auc']} ({train_time:.0f}s train) -> {out}")


if __name__ == "__main__":
    main()
