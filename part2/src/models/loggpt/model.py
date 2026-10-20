#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#
# Port of LogGPT (Han, Yuan & Trabelsi, IEEE BigData 2023; arXiv 2309.14482) from
# https://github.com/nokia/LogGPT (commit 28b15f9, identical to github.com/hanxiao0607/LogGPT):
# model/initGPT.py (pre-training, Top-K detection) and model/logGPT.py (RL fine-tuning).
# Modified for this project (modern PyTorch / transformers, Part 1's template-id sessions as input);
# every change is listed in NOTICE.md next to this file and in DECISIONS.md (P2-xx).
"""LogGPT: GPT-2 trained from scratch on template-id sequences, then fine-tuned with the Top-K reward.

fit(normal_sessions): the 5,000 normal training sessions, in sampled order. As in the original,
the first int(0.9 n) are used for training and the last 10% for validation (pre-training checkpoint
loss, RL early stopping). No anomalous session and no test session is ever seen during fit.
Session score = maximum, over predicted positions, of the rank of the actual next key in the
predicted distribution (1 = most likely; <unk> = vocabulary size + 1); anomalous <=> score > K.
"""
import copy
import math
import time

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging as hf_logging

from .vocab import WordVocab

hf_logging.set_verbosity_error()


def _pad_batch(seqs, pad, device):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
        m[i, :len(s)] = 1
    return x.to(device), m.to(device)


class LogGPT:
    name = "LogGPT"

    def __init__(self, params, seed=0, device="cuda", log=None, ckpt_dir=None):
        self.p = dict(params)
        self.seed = seed
        self.device = torch.device(device)
        self.log = log
        self.ckpt_dir = ckpt_dir
        self.vocab = None
        self.K = None
        self.pre_model = None       # after pre-training (= "LogGPT w/o RL")
        self.ft_model = None        # after RL fine-tuning (= "LogGPT")
        self.pre_history, self.rl_history = [], []
        self.times = {}
        self.meta = {}

    # ------------------------------------------------------------------ helpers
    def _info(self, msg, *a):
        if self.log:
            self.log.info(msg, *a)

    def _seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True

    def _tokens(self, sessions, i):
        return [str(int(t)) for t in sessions.seq(i)]

    def _new_model(self):
        v = self.vocab
        config = GPT2Config(vocab_size=len(v), bos_token_id=v.bos_index, eos_token_id=v.eos_index,
                            pad_token_id=v.pad_index, unk_token_id=v.unk_index, mask_token_id=v.mask_index,
                            n_positions=self.p["max_lens"], output_hidden_states=True,
                            n_layer=self.p["n_layers"], n_head=self.p["n_heads"], n_embd=self.p["n_embd"])
        model = GPT2LMHeadModel(config)
        model.resize_token_embeddings(len(v))
        return model.to(self.device)

    def _ckpt(self, name):
        return None if self.ckpt_dir is None else self.ckpt_dir / name

    # ------------------------------------------------------------------ fit
    def fit(self, normal_sessions):
        assert int(normal_sessions.labels.sum()) == 0, "LogGPT must be trained on normal sessions only"
        seqs = [self._tokens(normal_sessions, i) for i in range(len(normal_sessions))]
        self.vocab = WordVocab(seqs)                      # original: built on the whole train_df (5,000)
        self.K = int(self.vocab.n_keys * self.p["topk_ratio"])
        self.meta.update(n_train_sessions=len(seqs), n_unique_train_keys=self.vocab.n_keys,
                         vocab_size=len(self.vocab), K=self.K, topk_ratio=self.p["topk_ratio"],
                         K_original_code_default=min(int(self.vocab.n_keys * 0.95), 7),
                         max_train_len=max(len(s) for s in seqs), itos=self.vocab.itos)
        self._info("LogGPT: %d normal sessions, %d unique keys -> vocab %d, K = int(%d x %.2f) = %d",
                   len(seqs), self.vocab.n_keys, len(self.vocab), self.vocab.n_keys, self.p["topk_ratio"], self.K)
        enc = [self.vocab.forward(s) for s in seqs]
        assert max(len(e) for e in enc) <= self.p["max_lens"]
        n_fit = int(0.9 * len(enc))                       # original: valid_index = int(0.9 * len(train_df))
        self.fit_enc, self.val_enc = enc[:n_fit], enc[n_fit:]
        self.meta.update(n_fit=len(self.fit_enc), n_val=len(self.val_enc))
        t0 = time.time()
        self._pretrain()
        self.times["pretrain_s"] = time.time() - t0 + self.times.get("pretrain_resumed_s", 0.0)
        t0 = time.time()
        self._rl_finetune()
        self.times["rl_s"] = time.time() - t0 + self.times.get("rl_resumed_s", 0.0)
        return self

    def _pretrain(self):
        path = self._ckpt("pretrain.pt")
        if path is not None and path.exists():
            ck = torch.load(path, map_location=self.device, weights_only=False)
            assert ck["itos"] == self.vocab.itos
            self.pre_model = self._new_model()
            self.pre_model.load_state_dict(ck["last_state"])
            self.pre_history, self.times["pretrain_resumed_s"] = ck["history"], ck["time_s"]
            self.meta.update(pretrain_best_val_epoch=ck["best_epoch"])
            self._info("LogGPT pre-training cache hit: %s", path)
            return
        self._seed()
        model = self._new_model()
        optim = torch.optim.AdamW(model.parameters(), lr=self.p["init_lr"])
        bs, pad = self.p["init_batch_size"], self.vocab.pad_index
        best_loss, best_state, best_epoch = math.inf, None, None
        t_start = time.time()
        for epoch in range(self.p["init_num_epochs"]):
            t0 = time.time()
            model.train()
            tr_loss, nb = 0.0, 0
            for a in range(0, len(self.fit_enc), bs):     # DataLoader(shuffle=False) in the original
                x, m = _pad_batch(self.fit_enc[a:a + bs], pad, self.device)
                optim.zero_grad()
                loss = model(x, attention_mask=m, labels=x).loss
                loss.backward()
                optim.step()
                tr_loss += loss.item()
                nb += 1
            model.eval()
            val_loss, nv = 0.0, 0
            with torch.no_grad():
                for a in range(0, len(self.val_enc), bs):
                    x, m = _pad_batch(self.val_enc[a:a + bs], pad, self.device)
                    val_loss += model(x, attention_mask=m, labels=x).loss.item()
                    nv += 1
            val_loss /= nv
            if val_loss < best_loss:                      # the original saves this checkpoint ...
                best_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
            self.pre_history.append(dict(epoch=epoch, train_loss=tr_loss / nb, val_loss=val_loss, time_s=time.time() - t0))
            if epoch % 10 == 0 or epoch == self.p["init_num_epochs"] - 1:
                self._info("  pre-train epoch %3d  train loss %.4f  val loss %.4f  (%.1fs)", epoch, tr_loss / nb,
                           val_loss, time.time() - t0)
        # ... but keeps training and uses the LAST-epoch weights in memory (main.py never reloads it)
        self.pre_model = model
        self.meta.update(pretrain_best_val_epoch=best_epoch)
        self._info("LogGPT pre-training done in %.1fs; best val loss at epoch %d (%.4f), last epoch %.4f is used",
                   time.time() - t_start, best_epoch, best_loss, self.pre_history[-1]["val_loss"])
        if path is not None:
            torch.save(dict(last_state=model.state_dict(), best_state=best_state, best_epoch=best_epoch,
                            history=self.pre_history, itos=self.vocab.itos, time_s=time.time() - t_start), path)

    # ------------------------------------------------------------------ RL fine-tuning (model/logGPT.py)
    def _prompt(self, seq_ids):
        lens = len(seq_ids)
        i = self.p["cut"][0]
        if math.floor(i * lens) < 2:
            return seq_ids[:2]
        if math.floor(i * lens) >= lens - 2:
            return seq_ids[:-2]
        return seq_ids[:math.floor(i * lens)]

    def _generate(self, model, seq_ids):
        lens = len(seq_ids)
        input_ids = torch.tensor([self._prompt(seq_ids)], device=self.device)
        out = model.generate(input_ids, max_length=lens, min_length=lens,
                             num_beams=self.p["num_return_sequences"], do_sample=True,
                             pad_token_id=self.vocab.pad_index, num_return_sequences=1, early_stopping=False,
                             # transformers 5 returns a dict when the config sets output_hidden_states;
                             # 4.18 returned the tensor of token ids
                             return_dict_in_generate=False, output_hidden_states=False)
        assert out.shape[1] == lens, (out.shape, lens)
        return out

    def _rl_loss(self, model, seq_ids, generated):
        """compute_loss(): -(sum_t log p(actual k_t | generated prefix)) x (sum_t +-1 Top-K reward)."""
        logits = model(generated).logits[0, :-1][1:]      # [lens-2, V]
        ys = torch.tensor(seq_ids[2:], device=logits.device)
        log_prob = torch.log_softmax(logits.float(), dim=-1).gather(1, ys[:, None]).sum()
        topk = logits.topk(self.K, dim=-1).indices
        hits = (topk == ys[:, None]).any(dim=1)
        reward = int(hits.sum().item()) * 2 - len(ys)    # +1 per hit, -1 per miss
        return -log_prob * reward

    def _rl_step(self, model, optim, seq_ids):
        model.train()
        optim.zero_grad()
        generated = self._generate(model, seq_ids)
        loss = self._rl_loss(model, seq_ids, generated)
        loss.backward()
        optim.step()

    def _valid_step(self, model, seq_ids):
        model.eval()
        with torch.no_grad():
            generated = self._generate(model, seq_ids)
            return self._rl_loss(model, seq_ids, generated).item()

    def _val_loss(self, model):
        return sum(self._valid_step(model, s) for s in self.val_enc) / len(self.val_enc)

    def _rl_finetune(self):
        path = self._ckpt("rl_state.pt")
        ft = copy.deepcopy(self.pre_model)                # original: copy.deepcopy(initGPT_model)
        optim = torch.optim.Adam(ft.parameters(), lr=self.p["logGPT_lr"])
        eps = 1e-7
        if path is not None and path.exists():
            st = torch.load(path, map_location="cpu", weights_only=False)   # RNG states must stay on the CPU
            ft.load_state_dict(st["ft_state"])
            optim.load_state_dict(st["optim_state"])
            torch.set_rng_state(st["torch_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            rs = np.random.RandomState()
            rs.set_state(st["np_state"])
            best_state, best_loss, last_loss, count = st["best_state"], st["best_loss"], st["last_loss"], st["count"]
            start, self.rl_history, done = st["next_episode"], st["history"], st["done"]
            self.times["rl_resumed_s"] = st["time_s"]
            self._info("LogGPT RL: resuming at episode %d (done=%s)", start, done)
        else:
            self._seed()
            rs = np.random.RandomState(self.seed)         # original: global np.random seeded by set_seed(seed)
            t0 = time.time()
            init_loss = self._val_loss(ft)
            best_loss = last_loss = init_loss
            best_state, count, start, done = None, 0, 0, False
            self.rl_history = [dict(episode=-1, val_loss=init_loss, time_s=time.time() - t0)]
            self._info("LogGPT RL: initial val loss %.4f (%.1fs)", init_loss, time.time() - t0)
        t_run = time.time()
        prior = self.times.get("rl_resumed_s", 0.0)
        for episode in range(start, self.p["logGPT_episode"]):
            if done:
                break
            t0 = time.time()
            random_seed = rs.randint(0, 100)              # original: np.random.randint(0, 100); np.random.seed(...)
            rs.seed(random_seed)
            order = rs.permutation(len(self.fit_enc))
            for j in order:
                self._rl_step(ft, optim, self.fit_enc[j])
            t_train = time.time() - t0
            cur = self._val_loss(ft)
            saved = False
            if cur <= best_loss or episode == 0:
                best_loss = cur
                best_state = copy.deepcopy(ft.state_dict())
                saved = True
            rel = (last_loss - cur) / (cur + eps)
            if rel < 0.01:
                count += 1
            last_loss = cur
            self.rl_history.append(dict(episode=episode, val_loss=cur, rel_improvement=rel, count=count, saved=saved,
                                        train_s=t_train, time_s=time.time() - t0))
            self._info("  RL episode %2d  val loss %.4f  rel.impr. %+.4f  count %d  %s (%.1fs)", episode, cur, rel, count,
                       "saved" if saved else "", time.time() - t0)
            if count > 3:
                self._info("  RL early stop after episode %d", episode)
                done = True
            if path is not None:
                torch.save(dict(ft_state=ft.state_dict(), optim_state=optim.state_dict(), best_state=best_state,
                                best_loss=best_loss, last_loss=last_loss, count=count, next_episode=episode + 1,
                                done=done, history=self.rl_history, torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(), np_state=rs.get_state(),
                                time_s=prior + time.time() - t_run), path)
        self.ft_model = self._new_model()
        self.ft_model.load_state_dict(best_state)         # original test(): self.FT_GPT._load('FT')
        self.meta.update(rl_episodes_run=sum(1 for h in self.rl_history if h["episode"] >= 0),
                         rl_selected_episode=max(h["episode"] for h in self.rl_history if h.get("saved")),
                         rl_early_stopped=bool(count > 3))

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def max_rank_scores(self, sessions, which="ft", batch_size=512, unk_not_in_topk=True):
        """Max rank of the actual next key over all predicted positions (Section 5.4 of the brief).
        Identical sequences get identical scores (eval mode is deterministic), so each distinct
        sequence is scored once; batches are right-padded with an attention mask.
        unk_not_in_topk=True: an <unk> target gets rank V + 1 (brief rule). False: <unk> is ranked like
        any other token, which is what the original code does (used only for verification)."""
        model = self.ft_model if which == "ft" else self.pre_model
        model.eval()
        V = len(self.vocab)
        groups, inv = {}, np.empty(len(sessions), dtype=np.int64)
        for i in range(len(sessions)):
            inv[i] = groups.setdefault(sessions.seq(i).tobytes(), len(groups))
        uniq = [None] * len(groups)
        for i in range(len(sessions)):
            if uniq[inv[i]] is None:
                uniq[inv[i]] = self.vocab.forward(self._tokens(sessions, i))
        order = np.argsort([len(u) for u in uniq], kind="stable")
        uscore = np.zeros(len(uniq), dtype=np.int64)
        for a in range(0, len(order), batch_size):
            ib = order[a:a + batch_size]
            x, m = _pad_batch([uniq[k] for k in ib], self.vocab.pad_index, self.device)
            logits = model(x, attention_mask=m).logits[:, :-1][:, 1:].float()     # [B, L-2, V]
            probs = torch.softmax(logits, dim=-1)
            tgt = x[:, 2:]
            p_t = probs.gather(-1, tgt[..., None])
            rank = 1 + (probs > p_t).sum(-1)
            if unk_not_in_topk:
                rank = torch.where(tgt == self.vocab.unk_index, torch.full_like(rank, V + 1), rank)
            lens = m.sum(1)
            valid = torch.arange(tgt.shape[1], device=self.device)[None, :] < (lens - 2)[:, None]
            rank = torch.where(valid, rank, torch.zeros_like(rank))
            uscore[ib] = rank.max(1).values.cpu().numpy()
        n_unk = sum(self.vocab.unk_index in u for u in uniq)
        if unk_not_in_topk:
            self.meta[f"score_{which}"] = dict(n_sessions=len(sessions), n_distinct_sequences=len(uniq),
                                               n_distinct_with_unk=int(n_unk), max_len=int(max(len(u) for u in uniq) - 2))
        return uscore[inv]

    @torch.no_grad()
    def repo_predict_topk(self, sessions, which="ft"):
        """Verbatim logic of the original InitGPT._predict_topk (sliding_window=False): one sequence
        at a time, no attention mask, torch.topk on the softmax. Used only to verify max_rank_scores."""
        model = self.ft_model if which == "ft" else self.pre_model
        model.eval()
        y_pred = []
        for i in range(len(sessions)):
            seq_ids = self.vocab.forward(self._tokens(sessions, i))
            lst_ys = seq_ids[2:]
            input_ids = torch.tensor([seq_ids]).to(self.device)
            logits = model(input_ids).logits[:, :-1, :]
            logits = logits[:, 1:, :]
            probabilities = torch.softmax(logits, dim=-1)
            topk = torch.topk(probabilities, k=self.K)
            lst_topk = topk.indices.tolist()[0]
            lst_hit = [1 if val in lst_topk[ind] else 0 for ind, val in enumerate(lst_ys)]
            y_pred.append(1 if sum(lst_hit) < len(lst_ys) else 0)
        return np.array(y_pred)

    def save(self, path):
        torch.save(dict(pre_state=self.pre_model.state_dict(), ft_state=self.ft_model.state_dict(),
                        itos=self.vocab.itos, K=self.K, params=self.p, seed=self.seed,
                        pre_history=self.pre_history, rl_history=self.rl_history, meta=self.meta,
                        times=self.times), path)
