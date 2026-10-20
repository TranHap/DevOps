# LogGPT — attribution and changes

This folder contains a port of **LogGPT** (Xiao Han, Shuhan Yuan, Mohamed Trabelsi, *LogGPT: Log Anomaly
Detection via GPT*, IEEE BigData 2023, arXiv 2309.14482).

- Original code: https://github.com/nokia/LogGPT (commit `28b15f9`, 2023-12-11). The mirror
  https://github.com/hanxiao0607/LogGPT is the same commit.
- Copyright © 2023 Nokia. Licensed under the Creative Commons Attribution-NonCommercial 4.0
  International license (`LICENSE`, copied unchanged). The port is used for non-commercial coursework only.
- `vocab.py` derives from the original `utils/vocab.py` (Apache-2.0 header, itself from torchtext).

## What was ported

| Original | Here |
|---|---|
| `model/initGPT.py` `InitGPT._build_vocab`, `_init_training` | `model.py` `LogGPT.fit`, `_pretrain` |
| `model/initGPT.py` `InitGPT._predict_topk` | `model.py` `repo_predict_topk` (verbatim, for verification) and `max_rank_scores` (batched score) |
| `model/logGPT.py` `LogGPT.train`, `step`, `valid_step`, `compute_loss`, `test` | `model.py` `_rl_finetune`, `_rl_step`, `_valid_step`, `_rl_loss` |
| `utils/vocab.py` `WordVocab` | `vocab.py` |
| `utils/logdataset.py` `LogDataset.collate_fn` | `model.py` `_pad_batch` |
| `args/hdfs_config.py` defaults | `part2/configs/part2_hdfs.yaml` (`loggpt:` block) |

## Changes (each one is also a `P2-xx` entry in DECISIONS.md)

1. **Input.** The original parses HDFS with its own Drain and groups lines by block id. Here the input is
   Part 1's template-id sequences (same sessions, same order the LSTM sees); a token is the template id as a string.
2. **Training data.** The original shuffles all normal sessions and takes the first 5,000; the test set is all
   remaining sessions. Here the 5,000 normal sessions are sampled (seeded) from the normal sessions of the
   resample's 70% train split, and the test set is the resample's 30% test split, shared with the other models.
3. **K.** The paper sets K to 50% of the unique training keys; the HDFS config of the code uses
   `min(int(0.95·n), 7)`. Here `K = int(0.5 · n_keys)` (same rounding as the code's `topk` argument). On HDFS
   both give 7 when there are 15 training keys.
4. **No test evaluation during RL.** The original calls `_predict_topk` on the *test set* after every RL episode.
   That output is only printed (never used for selection) but costs one full test pass per episode; it is removed.
5. **Scoring.** The original evaluates one sequence at a time and outputs a binary decision (its AUC is computed
   on the binary prediction). Here a continuous score (max rank of the actual next key) is computed for batches of
   distinct sequences; `repo_predict_topk` keeps the original logic and is used to check that `score > K`
   reproduces its predictions exactly.
6. **Vectorised loss.** `compute_loss` loops over positions with `torch.distributions.Categorical` on the CPU; here
   the same quantities (`log_softmax` gather, Top-K membership on the logits) are computed on the GPU in one call.
7. **Seeding.** `torch` is re-seeded with the resample seed at the start of pre-training and of RL fine-tuning, so
   that a run resumed from the pre-training checkpoint is identical to an uninterrupted one. The episode shuffles
   use a `numpy.random.RandomState(seed)` that is re-seeded every episode exactly like the original's global
   `np.random.seed(np.random.randint(0, 100))`.
8. **Checkpoints.** Pre-training and every RL episode are checkpointed so that runs can be resumed.
9. **Library versions.** PyTorch 2.14 / transformers 5.20 instead of PyTorch 1.11 / transformers 4.18
   (RTX 40-series GPUs need CUDA ≥ 11.8).

Kept as in the code (even where the paper says otherwise or the code looks unintended): GPT-2 with 6 layers,
6 heads, n_embd 60, n_positions 512, trained from scratch; AdamW lr 1e-4, batch 16, 100 epochs, no shuffling,
loss also on padding positions (labels = input ids); the **last** pre-training epoch is used (the best-validation
checkpoint is saved but never reloaded by `main.py`); RL = one Adam (lr 1e-6) step per training sequence on
`-(Σ log p) · (Σ ±1 Top-K reward)` with a beam-sampled continuation (20 beams) of the first 20% of the sequence
(the paper calls it PPO); validation loss on the last 10% of the 5,000 sessions; early stop when the relative
improvement is < 1% for more than 3 episodes (counter never reset); the best-validation episode (or episode 0)
is the final model.
