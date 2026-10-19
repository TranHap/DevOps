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

### DeepLog diagnosis: why F1 0.57 when the literature reports ~0.9+ on HDFS
Our implementation already pads short sessions and checks **every** position (incl. the first event), and
trains on normal sessions only. Vocabulary: 18 keys (templates seen in normal sessions), not 46.
1. **No end-of-session (EOS) key** in the original DeepLog -> a session that *stops early* (missing events
   at the end, e.g. the 3rd `PacketResponder ... terminating`) is never checked. Adding EOS (`--eos`),
   seed 0, g=9: recall 0.407 -> 0.582, F1 0.575 -> 0.728, **AUC 0.847 -> 0.981**; +974 anomalies caught
   for +38 false alarms. A missing event in the *middle* of a sequence is visible without EOS (the next
   event becomes unlikely); the blind spot was the end.
2. **Top-g** (post-hoc sensitivity, not used for selection; `results/ext/deeplog_g_sensitivity.csv`,
   10 seeds, no EOS): g=3 F1 0.262 (P 0.17, R 0.61) | g=5 0.654 | g=9 0.573 | g=15 0.454. Even the best g
   leaves recall <= 0.61 without EOS. With EOS (seed 0): g=5 F1 0.795, g=9 0.728.
   g=9 comes from DeepLog's paper (~29 keys); with 18 keys it is lenient. We keep g=9 (fixed a priori):
   choosing g needs labelled anomalies, against the semi-supervised setting.
3. Remaining gap to ~0.9: training on 100k of ~391k normal sessions, CPU budget (5 epochs), and literature
   setups that differ in split/labels. Report the gap and these reasons explicitly.

=> Main ranking uses **DeepLog+EOS**; the original DeepLog is reported as an ablation.
   Note: EOS was added after looking at seed-0 results (a diagnosis-driven fix, not a tuned value).

## Sanity checks for the ranking (seed 0)
**Duplicate sequences (Le & Zhang, ICSE'22).** HDFS has only 18,375 distinct event sequences for 575,061
sessions. **97.8% of test sessions have an exact sequence that also occurs in train** (normal 98.4%,
anomalous 79.4%); at the MCV level (what the RF sees) 100% / 98.9%. 10 sequences carry both labels (46
sessions). Performance on test sessions whose sequence is NOT in train (3,789 sessions, 27.4% anomalous):
| Model | seen sequences F1 / AUC | **novel sequences** F1 / AUC |
|---|---|---|
| RF (sup.) | 0.9996 / 1.000 | 0.989 / 0.9998 |
| Isolation Forest | 0.687 / 0.965 | 0.657 / 0.830 |
| DeepLog (no EOS) | 0.573 / 0.822 | 0.580 / 0.856 |
RF on the 80 test sessions whose MCV vector is unseen: F1 0.931. -> Duplicates inflate the supervised scores a
little (most "novel" sequences are re-orderings with known counts) but do not explain the 0.3+ F1 gap.
Threat to validity: random session split + duplicates = near look-up for supervised models.

**Isolation Forest threshold.** contamination=0.03 (loglizer default) ~ the true anomaly rate (2.9%): indirect
label information. Label-free `contamination='auto'` flags ~9.7% of sessions: F1 0.41 (P 0.27, R 0.89),
AUC unchanged (0.96). -> IF's F1 rank depends on that prior; the AUC ranking is threshold-free and robust.

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

(table above = original DeepLog without EOS; final ranking with DeepLog+EOS: see below)

Wording for the report (claims kept to what the data supports):
- "On HDFS, supervised models outperform label-free ones by a large margin (F1 gap 0.3+), although random
  splitting with 97.8% duplicate sequences inflates the supervised scores."
- RF vs LSTM: small but consistent gap; the simplest supervised model (RF on counts) ranks first.
- IF vs DeepLog differ in learning paradigm, representation and model, so their order *suggests*, but does
  not show, that count- vs order-based views matter. IF's F1 also depends on its contamination prior.
- DeepLog without EOS: precision ~0.97 / recall ~0.40 on every seed, blind to sessions that stop early.

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
well on HDFS. A missing event at the *end* of a block (e.g. the 3rd PacketResponder) is invisible to the
original DeepLog, which has no end-of-session key; adding EOS recovers many of them (recall 0.41 -> 0.58).
