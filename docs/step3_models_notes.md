# Step 3 - Supervised models + correlation/VIF analysis: notes for the report

Scripts: `src/run_rf.py`, `src/run_lstm.py`, `src/feature_selection.py`, `src/summarize_rq1.py`.
Results: `results/rq1/*.json` (one per run), `results/rq1_all_runs.csv`, `results/rq1_summary.csv`.

## Setup
| | Classical | Deep learning |
|---|---|---|
| Model | Random Forest (sklearn defaults, 100 trees) | LSTM(hidden 8) -> FC(2) -> softmax (paper's network) |
| Representation | MCV (raw counts, train vocabulary + UNK) | TF-IDF (Text): TF-IDF of template tokens, fitted on train sessions; one vector per event |
| Input | 1 vector per session | sliding windows (HDFS 30/1, BGL 50/50, left-padded); window label = session label; session score = max window probability |
| Training | - | Adam lr 1e-3, batch 1024, HDFS 5 epochs, BGL 30 epochs, class-weighted cross-entropy (paper: imbalanced sampler) |
| Threshold | 0.5 on predicted probability | 0.5 on session score |

Why TF-IDF (Text) for the LSTM: the paper reports this pairing (comparable), it needs no multi-GB pre-trained
vectors, and its features are real words, so the correlation/VIF analysis stays interpretable.

**Correlation + redundancy analysis** (train only): (1) drop columns constant in train; (2) hierarchical
clustering on Spearman |rho| (average linkage, cut at |rho| = 0.7), keep the member with the lowest mean |rho|
(AutoSpearman rule); (3) drop the max-VIF feature until all VIF <= 5. For the LSTM the analysis runs on the
token features (rows = train sessions' mean vectors) and the LSTM then only sees the kept tokens.
Pitfall found: running it on a 100k-row subsample made rare templates look constant and dropped them
(HDFS: 6 "constant" columns vs 2 on the full train set). Now uses all train rows.

## Results: 10 stratified splits (seed 0-9), mean +- std
| Model | Dataset | #features (none -> vif) | F1 none | F1 vif | AUC none | AUC vif | Paper F1 |
|---|---|---|---|---|---|---|---|
| RF + MCV | HDFS | 47 -> 28 | 0.998 +- 0.000 | 0.997 +- 0.001 | 1.000 | 1.000 | 0.999 |
| RF + MCV | BGL | 366 -> 125 | 0.921 +- 0.027 | 0.909 +- 0.045 | 0.983 | 0.973 | 0.891 |
| LSTM + TF-IDF | HDFS | 113 -> 22 | 0.968 +- 0.014 | 0.923 +- 0.042 | 1.000 | 0.999 | 0.955 |
| LSTM + TF-IDF | BGL | 627 -> 108 | 0.851 +- 0.050 | 0.810 +- 0.035 | 0.976 | 0.949 | 0.871 |

Paired comparison none vs vif: Wilcoxon signed-rank paired by seed (same split for both arms), Holm-corrected
over the 7 tests; Cliff's delta (positive = "none" better).
| Model | Dataset | F1 p (Holm) | F1 delta | AUC p (Holm) | AUC delta |
|---|---|---|---|---|---|
| RF | HDFS | 0.002 (**0.014**) | +0.92 large | - (both 1.000) | - |
| RF | BGL | 0.824 (0.824) | +0.14 negligible | 0.375 (0.750) | +0.39 medium |
| LSTM | HDFS | 0.004 (**0.020**) | +0.84 large | 0.002 (**0.014**) | +1.00 large |
| LSTM | BGL | 0.084 (0.252) | +0.58 large | 0.014 (0.055) | +0.76 large |

Significant after Holm: RF-HDFS F1 (but tiny, ~0.001), LSTM-HDFS F1 and AUC. LSTM-BGL AUC (raw p = 0.014)
is **not** significant after correction: say "consistent drop, not significant on BGL's small test set".
Cliff's delta is computed between the two sets of 10 values (unpaired effect size, common in SE studies).

## Take-aways
1. **Replication holds.** Within ~0.02 F1 of the paper everywhere, and the ordering is the same:
   RF + MCV beats LSTM + TF-IDF on both datasets, and HDFS is much easier than BGL.
   Paper (RQ1): classical representations such as MCV work as well as or better than semantic ones.
2. **Correlation/VIF removes 40-85% of the features.**
   - RF: practically no loss. HDFS drop is statistically significant but only ~0.001 F1 (every seed loses
     a tiny bit). BGL difference is not significant. -> a 3x smaller, easier-to-explain model for free.
   - LSTM: hurts (HDFS F1 -0.045, significant after Holm; BGL same direction, not significant).
     **Mechanism (measured, seed 0):** every event vector is the TF-IDF vector of one of the 45 (HDFS) / 370
     (BGL) train templates, so the session x token matrix has rank <= #templates (HDFS: 110 tokens, rank 39;
     BGL: 634 tokens, rank 251) -> massive perfect multicollinearity by construction. Tokens of the same
     template are perfectly correlated, so the **correlation step** removes most of them (HDFS: -89 by
     correlation, -0 by VIF; BGL: -491 by correlation, -37 by VIF with VIF up to 3e13). The selection
     ignores whether a token helps tell templates apart: distinct event vectors drop from 44/45 to 29/45
     (HDFS) and from 343/370 to 203/370 (BGL), and 17 / 152 templates become **all-zero, i.e. identical to
     the padding vector**. The LSTM can no longer see those events.
     -> Correlation/VIF analysis is designed for tabular predictors (MCV); it is ill-suited to template
     embeddings, whose dimensions are not independent predictors.
3. **LSTM: F1 moves much more than AUC.** At seed 0 HDFS+VIF F1 is 0.843 but AUC 0.999: ranking is fine,
   the 0.5 threshold is badly placed (precision drops, recall stays ~1.0). Max-over-windows scoring +
   class weighting push scores up. Reporting AUC next to F1 matters.
4. **BGL is unstable:** 166 test windows, std of F1 0.03-0.05; seed 0 alone would have exaggerated the LSTM
   VIF drop (0.906 -> 0.766). Single-split results on BGL are unreliable; this motivates the 10 resamples.

## Leakage / method checklist (verified in code)
| Check | Answer |
|---|---|
| MCV vocabulary per seed, train only | yes: `mcv.build_mcv` uses `train_idx` of that seed; unseen events -> UNK |
| TF-IDF IDF per seed, train only | yes: `tfidf_text.event_vectors` fits on the seed's train sessions |
| Corr/VIF per seed, train only | yes: run inside `run_rf` / `run_lstm` on that seed's x_train; the kept set differs across seeds (9-10 distinct sets out of 10) |
| LSTM epochs / early stopping | fixed in advance (HDFS 5, BGL 30), no early stopping, nothing tuned on test (no hyper-parameter tuning at all) |
| VIF matrix for LSTM | train sessions x tokens, mean of event vectors (HDFS 402,542 x 110, BGL 660 x 634); singular -> VIF via pseudo-inverse of the correlation matrix |
| Window -> session | session score = max window probability; predicted anomalous if score > 0.5 (= at least one window > 0.5); AUC on the continuous score |
| Short sessions | left-padded to the window size with a zero vector; the last window is always added so the tail is covered |
| Paired test | yes, Wilcoxon signed-rank paired by seed + Holm + Cliff's delta |
| What a seed changes | both the split and the model initialisation (RF `random_state`, `torch.manual_seed`) |

## Threats
- LSTM epochs / lr / class weighting not reported by the paper; chosen by us, not tuned.
- Window labels inherit the session label (paper's choice): many "anomalous" BGL windows contain no alert line.
- Seeds also change the RF / LSTM initialisation, so the std mixes split variance and training variance.
