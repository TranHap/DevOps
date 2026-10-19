# Slides outline - LOG6309E group assignment (presentation Fri Oct 24, slides due Thu Oct 23 23:59)

Target: ~12 slides, ~10-12 min. One message per slide; the title states the finding.
Numbers = mean over 10 stratified resamples unless marked "seed 0". [F] = figure to include.

---

### 1. Title
Log-based Anomaly Detection: Replicating and Extending Wu et al. (EMSE 2023)
Team names, LOG6309E, Fall 2025.

### 2. Problem & pipeline
- Goal: detect failures from system logs automatically.
- Pipeline: raw logs -> parsing (templates) -> partition (sessions) -> representation -> model.
- Our scope: replicate RQ1 (representations x models) on HDFS + BGL, extend on HDFS.
[F] pipeline figure from the assignment (Wu et al. Fig. 1).

### 3. Setup at a glance
| | HDFS | BGL |
|---|---|---|
| Lines | 11.2 M | 4.7 M |
| Sessions | 575,061 (block id) | 826 (6 h windows) |
| Anomalous | 2.9% | 49.5% |
| Split | 70/30 | 80/20 |
- 10 stratified resamples (seeds 0-9), same splits for every model.
- Metrics: precision, recall, F1, AUC.

### 4. Parsing: preprocessing helps, but how you mask matters
- Drain (depth 4, st 0.5) + regex masking. Group accuracy on Loghub-2k: HDFS 0.998 -> **1.000**, BGL 0.963 -> **0.970**.
- Bug we found by ablation: re-spacing wildcards (`<*>:` -> `<*> :`) changed Drain's token count, BGL GA fell to 0.918.
- BGL: 1,444 -> **382** templates (drain3 put every all-wildcard message in its own cluster).
- HDFS: 46 templates = the 46 MCV dimensions reported by the paper.

### 5. Replication check: HDFS matches exactly, BGL does not
- HDFS: 575,061 sessions, 402,542 / 172,519 split, identical to the paper.
- BGL: 826 windows vs paper's 718. Evidence the paper dropped the 34,470 empty-content lines:
  4,713,493 lines / 718 = 6,564.8 = paper's 6,565 lines/session. Window boundaries still differ (not described).
- BGL 6 h windows: 49.5% anomalous -> "always anomalous" already gives F1 0.66.

### 6. RQ1 replication results: same picture as the paper
| Model | HDFS F1 (paper) | BGL F1 (paper) |
|---|---|---|
| RF + MCV | 0.998 (0.999) | 0.921 +- 0.027 (0.891) |
| LSTM + TF-IDF (Text) | 0.968 (0.955) | 0.851 +- 0.050 (0.871) |
- Within ~0.03 of the paper; the classical count-based model beats the deep one on both datasets.
- BGL varies a lot across splits (166 test windows): seed 0 alone would mislead.

### 7. Correlation + VIF: free for RF, harmful for the LSTM
| | #features | F1 none -> vif | Holm p |
|---|---|---|---|
| RF HDFS | 47 -> 28 | 0.998 -> 0.997 | 0.014 (tiny) |
| RF BGL | 366 -> 125 | 0.921 -> 0.909 | 0.82 |
| LSTM HDFS | 113 -> 22 | 0.968 -> 0.923 | 0.020 |
| LSTM BGL | 627 -> 108 | 0.851 -> 0.810 | 0.25 |
- Why the LSTM suffers: event vectors come from 45 templates, so 110 token columns have rank 39. Correlation
  removes most tokens, and 17/45 HDFS templates become all-zero, identical to padding.
- VIF suits tabular predictors (MCV), not template embeddings.

### 8. Extension: two label-free models on HDFS
- Isolation Forest on MCV (classical, unsupervised, labels never used).
- DeepLog (deep learning, semi-supervised: normal sessions only, next-event prediction, top-g = 9).
- DeepLog fix: original DeepLog has no end-of-session key -> sessions that stop early are never checked.
  Adding EOS: recall 0.41 -> 0.58, AUC 0.85 -> 0.98 (seed 0).

### 9. Scott-Knott ESD ranking (4 models x 10 resamples)
TBD after DeepLog+EOS seeds finish. Previous ranking (DeepLog without EOS):
RF (0.998) > LSTM (0.968) > Isolation Forest (0.654) > DeepLog (0.573), same order for AUC.
[F] `results/ext/ranking_f1.png` and `ranking_auc.png`.

### 10. Can we trust the near-perfect HDFS scores?
- Only 18,375 distinct sequences for 575k sessions: **97.8% of test sessions have a sequence seen in train**.
- On novel sequences RF still gets F1 0.989 (counts generalise), 0.931 when the count vector itself is unseen (80 sessions).
- Isolation Forest F1 depends on its contamination prior: 0.03 -> 0.65, label-free 'auto' -> 0.41 (AUC unchanged 0.96).

### 11. What the best model learned (SHAP on RF, VIF-selected features, seed 0)
- Top signals: `Unexpected error trying to delete block` appears; **fewer `PacketResponder ... terminating`**
  (normal blocks have exactly 3, one per replica; anomalous 2.09 on average); `Starting thread to transfer block`.
- Local example: a block with 0 PacketResponder lines -> 0.029 -> 1.0.
- The RF learned HDFS's 3-replica write protocol: many anomalies are **missing** events.
[F] `results/ext/shap/shap_beeswarm_vif_seed0.png` (+ waterfall if space)

### 12. Take-home messages
1. Simple beats complex here: RF on template counts ranks first, the paper's RQ1 finding holds.
2. Correlation/VIF analysis: use it for tabular features, not for embeddings.
3. Missing events are a key anomaly signal on HDFS: count-based models see them; sequence models need EOS.
4. Random splits + duplicate sequences make HDFS look easier than it is: report it, prefer AUC + resampling.

### Backup slides (only if asked)
- Leakage checklist (every vocabulary/IDF/feature selection fit per seed on train only).
- DeepLog top-g sensitivity table.
- BGL window-count investigation details.
