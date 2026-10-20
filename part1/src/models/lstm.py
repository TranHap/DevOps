"""LSTM + BERT template embeddings (event-level, sequential input).

Architecture from the replication package (models/LSTM.py): LSTM(768 -> hidden 8, 1 layer)
-> Linear(8 -> 2), cross-entropy, Adam lr 1e-3, 30 epochs, ImbalancedDatasetSampler
(sessions drawn with replacement, inverse class-frequency weights).

Memory: windows are int32 arrays of template ids; the 768-d vectors are looked up inside the
model from a frozen nn.Embedding (row 0 = padding). Sequences are right-padded and packed with
their true lengths, so padding never reaches the final hidden state.

Windows (per session, size w, stride s; package collate_fn semantics, with two fixes):
  - L <= w : one window with the whole session (package keeps only fea[L-w+1:], which drops
             events when L < w/2 -- see DECISIONS.md)
  - L >  w : starts 0, s, 2s, ... <= L-w; if events remain after the last full window, one
             more (shorter) window with the remainder, as the package does
Every window inherits its session's label. Session score = max window P(anomaly).
"""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..common import get_device


def make_windows(sessions, win, stride, package=False):
    """Returns W [N, win] int32, lengths [N] int64, session index [N] int64, win_offsets [n+1].

    package=True reproduces the replication package's collate_fn exactly (ablation only):
    L < w keeps fea[L-w+1:] (only the last w-L-1 events when w-L-1 < L), and for L >= w the
    remainder fea[last+stride:] is always appended when non-empty (redundant for stride 1).
    """
    lens = sessions.lengths
    sess, starts, wlen = [], [], []
    short = np.flatnonzero(lens < win) if package else np.flatnonzero(lens <= win)
    st0, ln0 = np.zeros(len(short), dtype=np.int64), lens[short].astype(np.int64)
    if package:
        keep = win - ln0 - 1                       # |L - w + 1|
        cut = (keep > 0) & (keep < ln0)            # python slice with negative start inside the array
        st0 = np.where(cut, ln0 - keep, 0)
        ln0 = np.where(cut, keep, ln0)
    sess.append(short)
    starts.append(st0)
    wlen.append(ln0)
    for i in np.flatnonzero(lens >= win) if package else np.flatnonzero(lens > win):
        L = int(lens[i])
        st = np.arange(0, L - win + 1, stride, dtype=np.int64)
        ln = np.full(len(st), win, dtype=np.int64)
        rem = L - (st[-1] + stride)
        if (package and rem > 0) or (not package and st[-1] + win < L):
            st = np.append(st, st[-1] + stride)
            ln = np.append(ln, L - st[-1])
        sess.append(np.full(len(st), i, dtype=np.int64))
        starts.append(st)
        wlen.append(ln)
    sess, starts, wlen = np.concatenate(sess), np.concatenate(starts), np.concatenate(wlen)
    order = np.lexsort((starts, sess))
    sess, starts, wlen = sess[order], starts[order], wlen[order]
    N = len(sess)
    W = np.zeros((N, win), dtype=np.int32)
    ar = np.arange(win)
    for a in range(0, N, 200_000):
        b = min(N, a + 200_000)
        idx = sessions.offsets[sess[a:b]][:, None] + starts[a:b][:, None] + ar[None, :]
        mask = ar[None, :] < wlen[a:b][:, None]
        W[a:b] = np.where(mask, sessions.events[np.minimum(idx, len(sessions.events) - 1)], 0)
    win_offsets = np.zeros(len(sessions) + 1, dtype=np.int64)
    np.cumsum(np.bincount(sess, minlength=len(sessions)), out=win_offsets[1:])
    return W, wlen, sess, win_offsets


class LSTMNet(nn.Module):
    def __init__(self, embeddings, hidden, layers, dropout, packed=True):
        super().__init__()
        self.packed = packed
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        e = self.emb(x.long())
        if not self.packed:                         # package behaviour: last time step, padding included
            out, _ = self.lstm(e)
            return self.fc(out[:, -1, :])
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


class LSTMBert:
    name = "LSTM"
    representation = "BERT"

    def __init__(self, embeddings, window, params, seed=42, log=None):
        self.E = embeddings.astype(np.float32)
        self.win, self.stride = window["size"], window["stride"]
        self.p = dict(params)
        self.seed = seed
        self.log = log
        self.device = get_device(log)
        self.net = None
        self.history = []
        self.best_epoch = None
        self.n_features = self.E.shape[1]
        # ablation only: reproduce the package's windowing + padding ("package") and, in addition,
        # its one-optimiser-step-per-session training ("package_steps")
        self.package = bool(self.p.get("package_faithful", False) or self.p.get("package_windows", False))
        self.package_steps = bool(self.p.get("package_faithful", False))

    def _seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _window_probs(self, W, L, bs=2048):
        self.net.eval()
        out = np.empty(len(W), dtype=np.float32)
        with torch.no_grad():
            for a in range(0, len(W), bs):
                x = torch.from_numpy(W[a:a + bs]).to(self.device)
                logits = self.net(x, torch.from_numpy(L[a:a + bs]))
                out[a:a + bs] = torch.softmax(logits, 1)[:, 1].float().cpu().numpy()
        return out

    @staticmethod
    def _session_max(win_probs, sess, n):
        s = np.zeros(n, dtype=np.float32)
        np.maximum.at(s, sess, win_probs)
        return s

    def fit(self, train_sessions, diag_test_sessions=None):
        """diag_test_sessions: optional, DIAGNOSTIC ONLY -- test metrics are logged per epoch to quantify
        the optimism of the package's best-epoch-on-test selection; they never influence model selection."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        self._seed()
        rng = np.random.default_rng(self.seed)
        n = len(train_sessions)
        perm = rng.permutation(n)
        n_val = int(round(n * self.p["val_ratio"]))
        val_idx, fit_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        fit_s, val_s = train_sessions.subset(fit_idx), train_sessions.subset(val_idx)
        Wf, Lf, Sf, Of = make_windows(fit_s, self.win, self.stride, self.package)
        Wv, Lv, Sv, _ = make_windows(val_s, self.win, self.stride, self.package)
        if diag_test_sessions is not None:
            Wt, Lt, St, _ = make_windows(diag_test_sessions, self.win, self.stride, self.package)
        yf_sess = fit_s.labels.astype(np.int64)
        if self.log:
            self.log.info("LSTM: %d fit sessions (%d anomalous) -> %d windows; %d val sessions (%d anomalous) -> %d windows",
                          len(fit_s), int(yf_sess.sum()), len(Wf), len(val_s), int(val_s.labels.sum()), len(Wv))
        self.net = LSTMNet(self.E, self.p["hidden"], self.p["layers"], self.p.get("dropout", 0.0),
                           packed=not self.package).to(self.device)
        opt = torch.optim.Adam([q for q in self.net.parameters() if q.requires_grad], lr=self.p["lr"])
        crit = nn.CrossEntropyLoss()
        Wf_t = torch.from_numpy(Wf).to(self.device)
        Lf_t = torch.from_numpy(Lf)
        yw_t = torch.from_numpy(yf_sess[Sf]).to(self.device)
        n_win_per_sess = np.diff(Of)
        if self.p.get("balanced_sampler", True):
            counts = np.bincount(yf_sess, minlength=2).astype(np.float64)
            w = 1.0 / counts[yf_sess]
            w /= w.sum()
        best = (-1.0, None)
        bs = self.p["batch_size"]
        for ep in range(1, self.p["epochs"] + 1):
            t0 = time.time()
            self.net.train()
            if self.p.get("balanced_sampler", True):
                drawn = rng.choice(len(fit_s), size=len(fit_s), replace=True, p=w)
            else:
                drawn = rng.permutation(len(fit_s))
            tot, nb = 0.0, 0
            if self.package_steps:
                # package: one optimiser step per drawn session, on all of its windows
                widx = np.arange(0)
                for d in drawn:
                    n_d = int(Of[d + 1] - Of[d])
                    opt.zero_grad()
                    sess_loss = 0.0
                    # gradient accumulation over chunks == one step on the mean loss of all windows (fits in GPU memory)
                    for a in range(int(Of[d]), int(Of[d + 1]), 1024):
                        ib_t = torch.arange(a, min(a + 1024, int(Of[d + 1])), device=self.device)
                        logits = self.net(Wf_t[ib_t], Lf_t[ib_t.cpu()])
                        loss = crit(logits, yw_t[ib_t]) * (len(ib_t) / n_d)
                        loss.backward()
                        sess_loss += loss.item()
                    opt.step()
                    tot += sess_loss
                    nb += 1
            else:
                # expand drawn sessions to their windows
                reps = n_win_per_sess[drawn]
                widx = np.repeat(Of[drawn], reps) + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))
                widx = widx[rng.permutation(len(widx))]
                for a in range(0, len(widx), bs):
                    ib = widx[a:a + bs]
                    ib_t = torch.from_numpy(ib).to(self.device)
                    logits = self.net(Wf_t[ib_t], Lf_t[torch.from_numpy(ib)])
                    loss = crit(logits, yw_t[ib_t])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                    nb += 1
            pv = self._session_max(self._window_probs(Wv, Lv), Sv, len(val_s))
            yv = val_s.labels
            pred = (pv >= 0.5).astype(int)
            f1 = f1_score(yv, pred, zero_division=0)
            rec = dict(epoch=ep, loss=tot / max(nb, 1), n_windows=len(widx),
                       val_precision=precision_score(yv, pred, zero_division=0),
                       val_recall=recall_score(yv, pred, zero_division=0), val_f1=f1, time_s=time.time() - t0)
            if diag_test_sessions is not None:
                wt = self._window_probs(Wt, Lt)
                pt = (self._session_max(wt, St, len(diag_test_sessions)) >= 0.5).astype(int)
                yt = diag_test_sessions.labels
                pw, yw = (wt >= 0.5).astype(int), yt[St]
                rec.update(test_precision=precision_score(yt, pt, zero_division=0),
                           test_recall=recall_score(yt, pt, zero_division=0), test_f1=f1_score(yt, pt, zero_division=0),
                           test_window_precision=precision_score(yw, pw, zero_division=0),
                           test_window_recall=recall_score(yw, pw, zero_division=0),
                           test_window_f1=f1_score(yw, pw, zero_division=0))
            self.history.append(rec)
            if self.log:
                self.log.info("  epoch %2d loss %.4f | val P %.3f R %.3f F1 %.3f | %.1fs", ep, rec["loss"],
                              rec["val_precision"], rec["val_recall"], f1, rec["time_s"])
            if f1 > best[0]:
                best = (f1, copy.deepcopy(self.net.state_dict()))
                self.best_epoch = ep
        self.net.load_state_dict(best[1])
        if self.log:
            self.log.info("LSTM: best epoch %d (val F1 %.3f)", self.best_epoch, best[0])
        return self

    def predict_scores(self, sessions, return_windows=False):
        W, L, S, _ = make_windows(sessions, self.win, self.stride, self.package)
        wp = self._window_probs(W, L)
        scores = self._session_max(wp, S, len(sessions))
        if return_windows:
            return scores, wp, S
        return scores

    def save(self, path):
        torch.save(dict(state_dict=self.net.state_dict(), params=self.p, window=(self.win, self.stride),
                        seed=self.seed, best_epoch=self.best_epoch, history=self.history), path)
