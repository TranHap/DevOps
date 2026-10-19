# Step 2 - Partition, split, MCV: notes for the report

Scripts: `src/partition.py` -> `src/split.py` -> `src/mcv.py` (run per dataset).

## Design decisions
- **Sessions keep the full event sequence** (`data/sessions/<DS>_sessions.pkl`: session_id, events in LineId
  order, label, n_events), not only MCV. Needed by LSTM (replication) and DeepLog (extension).
- **One split shared by all 4 models.** `split(labels, ratio, seed)`: shuffled like the paper, plus
  **stratified** on the label (deviation from the paper, to keep the same anomaly rate in train/test).
  seed 0 = main split; seeds 1-9 = resamples for the Scott-Knott ranking. Saved as `data/splits/<DS>_seed<k>.npz`.
- **MCV = raw counts** (no TF-IDF / scaling), vocabulary from **train only**, plus an `UNK` column for
  templates never seen in train. Same npz keys as Wu et al.'s package (x_train, y_train, x_test, y_test).

## Results (seed 0)
| | HDFS | BGL | Paper |
|---|---|---|---|
| Grouping | block_id | 6 h window on epoch timestamp, from first log | |
| Sessions | 575,061 | **826** | HDFS 575,061 / BGL **718** |
| Anomalous sessions | 16,838 (2.93%) | **409 (49.5%)** | |
| Train / test | 402,542 / 172,519 | 660 / 166 | HDFS 402,542 / 172,519 (exact match); BGL 575 / 143 |
| Events per session (mean) | 19.4 | 5,748 | HDFS 22 / BGL 6,565 |
| MCV columns | 45 templates + UNK | 370 templates + UNK | HDFS 46 |
| Columns constant in train | 2 | 1 | |
| Test sessions with UNK events | 1 | 8 | |

## BGL: why 826 sessions and not 718
- Other window alignments do not give 718 either: UTC-aligned 824, PDT-aligned 820, the `Time` column 828.
- Paper's average of 6,565 lines/session x 718 = ~4.71 M lines. Our total without the 34,470 empty-content
  lines is 4,713,493 and 4,713,493 / 718 = **6,564.8**. So the paper most likely **dropped the empty-content
  lines** (a header regex that requires a Content field; we hit the same bug in our first parse).
- Dropping those lines still gives 826 windows, so the paper's window boundaries differ in some way it does not
  describe, and the replication package has no partition code. We keep 826 windows, keep the empty lines, and
  report this as a **threat to validity** (construct / replication).

## Points for the discussion section
- **BGL: half of the 6 h windows are anomalous (49.5%)** vs 7.34% of lines. A 6 h window almost always
  contains some alert line, so the task becomes easy to "guess" (always predicting anomaly gives F1 = 0.66)
  and precision/recall are less informative. Report it next to the F1.
- **BGL test set = 166 sessions**: one misclassified window changes F1 by ~0.01. Expect high variance across
  seeds; this is why the extension uses 10 resamples + Scott-Knott.
- HDFS: 2 of 46 MCV columns are constant in train (incl. UNK) -> they must be dropped **before** VIF
  (VIF is undefined for constant columns). BGL: 371 columns vs 660 train sessions -> also filter constant /
  near-constant columns before VIF.
- Shuffled split (paper's choice) can leak future information (Le & Zhang 2022). Kept to stay comparable;
  mention as a threat.
