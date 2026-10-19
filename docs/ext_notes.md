# Extension (HDFS): notes for the report

Dataset: **HDFS**. Reasons: 172k test sessions (stable estimates; BGL has 166 test windows and F1 std up to
0.05), fast enough on CPU for 10 resamples x 4 models. Same 10 stratified splits (seeds 0-9) for all models.

## E1 - Two unsupervised / semi-supervised models (two categories)
| | Isolation Forest (classical, unsupervised) | DeepLog (deep learning, semi-supervised) |
|---|---|---|
| Script | `src/run_iforest.py` | `src/run_deeplog.py` |
| Input | MCV (same as RF) | event-id sequences |
| Training data | all train sessions, labels **not used** | **normal** train sessions only (random 100k of ~391k per seed) |
| Idea | anomalies are isolated by few random splits | LSTM predicts the next event from the last h=10; anomaly if the true next event is outside the top-g=9 |
| Settings (fixed a priori) | 100 trees, contamination=0.03 (loglizer default) | 2-layer LSTM(64), emb 32, 5 epochs, Adam 1e-3, batch 2048 |
| Session score (for AUC) | `-score_samples` | max over positions of the rank of the true next event (unseen event = max) |

Threats: contamination 0.03 is close to the true HDFS anomaly rate (2.9%), which favours IF's F1; g=9 comes
from DeepLog's paper where HDFS had 29 keys, here normal sessions use only **18 keys**, so top-9 = half
the vocabulary (lenient).

### DeepLog diagnosis (seed 0: P 0.977, R 0.407, F1 0.575, AUC 0.847)
Not a bug, a property of the model (score distribution, `results/ext/scores/DeepLog_HDFS_seed0.npz`):
- 1,163 anomalous test sessions contain an event never seen in normal training -> all detected.
- ~2,200 anomalous sessions have a perfectly normal *order* (true next event always in the top-3).
  Example: a block with `Unexpected error trying to delete block`, a template that also occurs in normal
  sessions. DeepLog only asks "is the next event plausible?", so it **cannot see missing events**.
- SHAP (E3) shows that the RF's strongest signal is exactly a missing event (fewer `PacketResponder`
  lines than the 3 replicas). Counting (MCV) sees it, next-event prediction does not.
- Lower g would raise recall (anomalies with max rank 4-8: ~800 vs normals: ~700) but g was not tuned:
  tuning it would need labelled anomalies, against the semi-supervised setting.

## E2 - Scott-Knott ESD ranking
`src/rank_models.py` + `src/scott_knott.R` (R package ScottKnottESD 2.0.3, `sk_esd(version="np")`:
non-parametric, merges groups with negligible Cliff's delta). 4 models x 10 resamples, on F1 and on AUC.
Supervised models are the "no corr/VIF" variants (their best in RQ1).
Results: `results/ext/ranking_summary.csv`, plots `results/ext/ranking_f1.png`, `ranking_auc.png`.

| SK-ESD rank | Model | F1 mean +- std | AUC mean +- std | Precision | Recall |
|---|---|---|---|---|---|
| 1 | RF + MCV (supervised) | 0.998 +- 0.000 | 1.000 +- 0.000 | 0.997 | 0.999 |
| 2 | LSTM + TF-IDF (supervised) | 0.968 +- 0.014 | 0.9996 +- 0.0001 | 0.938 | 0.999 |
| 3 | Isolation Forest + MCV (unsupervised) | 0.654 +- 0.020 | 0.958 +- 0.004 | ~0.72 | ~0.60 |
| 4 | DeepLog (semi-supervised) | 0.573 +- 0.015 | 0.839 +- 0.019 | ~0.97 | ~0.40 |

- 4 distinct groups, **same order for F1 and AUC**: supervised >> unsupervised/semi-supervised.
- The gap between the two supervised models is small (0.03 F1, 0.0004 AUC) but consistent; the gap to
  the label-free models is huge (0.3-0.4 F1). On HDFS, labels are worth far more than model complexity:
  the simplest supervised model (RF on counts) wins, the deep models do not.
- IF vs DeepLog: IF has a much better AUC (0.958 vs 0.839) even without labels, because it works on counts
  (MCV) and can see missing/extra events; DeepLog only checks the order of events.
- DeepLog's F1 profile is precision ~0.97 / recall ~0.40 on every seed: it rarely raises false alarms but
  misses all anomalies whose order looks normal. IF is the opposite trade-off (P ~0.72, R ~0.60).
- Practical reading: if labelled incidents exist, use a supervised count-based model; without labels, a
  count-based unsupervised model (IF) is the better start than next-event prediction on HDFS-like logs.

## E3 - Explaining the best model (RF + MCV) with SHAP
`src/explain_rf.py`: RF retrained on seed 0 with the corr/VIF-selected 28 MCV features, TreeExplainer
(exact for trees) on 1,000 anomalous + 1,000 normal test sessions.
Why the VIF-selected RF: SHAP splits credit between correlated features (e.g. the 3 per-replica events that
always co-occur), so a ranking on the 46 raw features is misleading; RQ1 showed same performance in practice
(F1 0.997 vs 0.998).

Top features (mean |SHAP|, seed 0), see `results/ext/shap/shap_importance_vif_seed0.csv`:
| Template | What anomalous blocks look like |
|---|---|
| E32 `Unexpected error trying to delete block ... BlockInfo not found in volumeMap` | appears (0.35 per anomalous block vs ~0 for normal) |
| E3 `PacketResponder <*> for block <*> <*>` (raw: `PacketResponder 1 for block blk_... terminating`) | **fewer** lines: 2.09 vs exactly 3 for normal blocks (one per replica) -> a replica pipeline did not finish |
| E11 `Starting thread to transfer block` | appears: re-replication of the block |
| E33 / E20 `addStoredBlock ... does not belong to any file` / `Redundant addStoredBlock` | appear |
| E15 `writeBlock received exception ... Could not read from stream` | appears |

Local example (`shap_waterfall_vif_seed0.png`): base rate 0.029 -> 1.0, driven by **0** PacketResponder
lines (+0.9): the block never completed any replica write.

Take-away: the RF learned HDFS's replication protocol (3 replicas -> 3 PacketResponder/Received/addStoredBlock
events). Many anomalies are *missing* or *extra* events, which is why count-based features (MCV) work so
well on HDFS and why an order-only model (DeepLog) misses them.
