# Part 1 — Replication of RQ1 (Wu, Li & Khomh, EMSE 2023)

Log-based anomaly detection on **HDFS** and **BGL**: Drain parsing with the Qin et al. (SANER 2025)
preprocessing framework, then two supervised models:

| model | representation | input level |
|---|---|---|
| Random Forest | Message Count Vector (MCV) | session |
| LSTM | BERT template embeddings (frozen) | event sequence (sliding windows) |

Step 3a (correlation analysis followed by VIF redundancy analysis) is applied to RF + MCV, with and without.
Results are in `results/RESULTS.md`. Every choice and every deviation from the paper is in `DECISIONS.md`.

## Layout

```
part1/
  configs/           hdfs.yaml, bgl.yaml (main); smoke_*.yaml (2k samples); bgl_qinfull.yaml (sensitivity)
  src/
    preprocess.py    header split, identifier extraction, masking (Qin framework vendored, regex modes)
    drain.py         logpai Drain, streaming
    parse.py         stage 1: raw log -> lines.npz, templates.csv, structured.parquet
    parsing_eval.py  GA / PA on the Loghub-2.0 2k samples -> results/parsing_eval.csv
    partition.py     stage 2: sessions (HDFS block id, BGL 6 h windows) + Sessions container
    split.py         stage 3: seeded session-level split (any seed; reused by Part 2)
    represent_mcv.py MCV (vocabulary fitted on train)
    represent_bert.py BERT template embeddings + rank evidence
    feature_selection.py step 3a
    models/rf.py     RFMCV:    fit(train_sessions) / predict_scores(sessions)
    models/lstm.py   LSTMBert: fit(train_sessions) / predict_scores(sessions)
    evaluate.py      session-level metrics, metrics.csv, comparison with the paper
    report.py        builds RESULTS.md, seed_summary.csv and figures from the result files
    run_part1.py     driver
  extra_runs.sh      optional runs: seeds 0-4, loglizer split, LSTM test-curve diagnostic
  data/              (git-ignored) raw/ parsed/ sessions/ splits/ features/ models/ predictions/
  results/           metrics.csv, comparison_with_paper.csv, parsing_eval.csv, feature_selection_*.csv,
                     bert_rank.json, seed_summary.csv, figures/, RESULTS.md, discrepancy_notes.md
  logs/              one log file per dataset and seed
```

## Reproduce from raw data

All commands run from `part1/`. Total runtime on an RTX 4060 laptop: about 10 minutes for the required runs
and about 35 minutes for `extra_runs.sh`.

### 1. Environment

```bash
# The venv is ~/.venvs/GroupAssignment (Python 3.13). With the zsh helper: `workon GroupAssignment`
uv venv --python 3.13 ~/.venvs/GroupAssignment
source ~/.venvs/GroupAssignment/bin/activate
uv pip install -r requirements.txt          # torch comes from the cu130 index listed in the file
```

The BERT checkpoint (`bert-base-uncased`, about 440 MB) is downloaded on first use into `part1/.hf_cache`
(`src/common.py` sets `HF_HOME`).

### 2. Data

```bash
mkdir -p data/raw && cd data/raw
curl -L -o HDFS_v1.zip 'https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1'
curl -L -o BGL.zip     'https://zenodo.org/records/8196385/files/BGL.zip?download=1'
unzip HDFS_v1.zip -d HDFS_v1 && unzip BGL.zip -d BGL && cd ../..
wc -l data/raw/HDFS_v1/HDFS.log data/raw/BGL/BGL.log    # 11175629 and 4747963

# 2k samples with corrected ground truth (smoke test and parsing evaluation)
git clone --depth 1 https://github.com/logpai/loghub-2.0.git ../external/loghub-2.0
```

`../external/` also holds the replication package, the Qin et al. artifact and logparser. They are only needed
for reading; their code is vendored where it is used.
`external/` is not part of the repository (git-ignored). Only the Loghub-2.0 clone above is read by the code
(smoke test and parsing evaluation); the main runs do not need it.

### 3. Smoke test (2k samples, about 1 minute)

```bash
python -m src.run_part1 --dataset configs/smoke_hdfs.yaml --stage all
python -m src.run_part1 --dataset configs/smoke_bgl.yaml  --stage all
```

The outputs go to `data/smoke/` and `results/smoke/`. The metrics are meaningless here, because 2k random
lines give sessions of about one event; the smoke test only checks that every stage runs.

### 4. Parsing evaluation

```bash
python -m src.parsing_eval                  # -> results/parsing_eval.csv
```

### 5. Required runs (seed 42; 3 runs per dataset)

```bash
for d in hdfs bgl; do
  python -m src.run_part1 --dataset $d --seed 42 --stage all
done
```

`--stage all` runs `parse → partition → split → features → train → eval`. Each stage caches its output and is
skipped when the output exists; `--force` recomputes it. A single stage or model can be run alone:

```bash
python -m src.run_part1 --dataset bgl --seed 42 --stage train --model rf_fs   # rf | rf_fs | lstm | all
python -m src.run_part1 --dataset bgl --seed 42 --stage eval
```

### 6. Optional runs and report

```bash
./extra_runs.sh       # seeds 0-4 on both datasets, loglizer split on HDFS, LSTM diagnostics and ablations,
                      # BGL parse without Qin's estimation narrowing, then the report (about 40 minutes)
python -m src.report  # -> results/RESULTS.md, comparison_with_paper.csv, seed_summary.csv, figures/
```

Diagnostics (they never write `metrics.csv`):

```bash
python -m src.run_part1 --dataset bgl --lstm-test-curve         # test F1 per epoch (not used for selection)
python -m src.run_part1 --dataset hdfs --lstm-package-windows   # package windowing + padding
python -m src.run_part1 --dataset bgl --lstm-package-faithful   # + one optimiser step per session
```

## Outputs

| file | content |
|---|---|
| `results/metrics.csv` | one row per (dataset, model, representation, feature_selection, split, seed): n_features, P, R, F1, AUC, TP/FP/TN/FN, train time |
| `results/comparison_with_paper.csv` | seed-42 P/R/F1 next to paper Tables 2 and 3, with deltas |
| `results/parsing_eval.csv` | GA / PA / PA_relaxed of 4–5 preprocessing configurations on the 2k samples |
| `results/feature_selection_{hdfs,bgl}.csv` | every template removed by step 3a, the step that removed it and the reason (ρ or VIF) |
| `results/feature_selection_*_info.json` | counts per step, VIF per iteration, kept ids |
| `results/bert_rank.json` | number of templates, embedding dimension and rank of the embedding matrix |
| `results/seed_summary.csv` | mean ± std over seeds 0–4 and 42 |
| `results/lstm_test_curve_*.json` | diagnostic only: test F1 per epoch (never used to select a model) |
| `results/lstm_package_*.json` | LSTM ablations reproducing the package's windowing / padding / per-session steps |
| `results/discrepancy_notes.md` | hand-written discrepancy analysis, included in RESULTS.md |
| `results/figures/` | dendrograms with cut line, F1 vs paper, LSTM training curves |
| `data/models/<ds>/*.joblib` | trained RF; `feature_names_in_` are the template texts (SHAP-ready for Part 2) |
| `data/predictions/<ds>/*.npz` | test session scores and labels behind every metrics row |

## Interface for Part 2

```python
from src.common import load_config
from src.partition import load_sessions
from src.split import get_split
from src.models.rf import RFMCV
import pandas as pd

cfg = load_config("hdfs")
s = load_sessions(cfg)
train_idx, test_idx = get_split(cfg, seed=7, log=None)     # any seed; cached in data/splits/
templates = pd.read_csv("data/parsed/hdfs/templates.csv", keep_default_na=False)
model = RFMCV(templates, cfg["rf"], seed=7).fit(s.subset(train_idx))
scores = model.predict_scores(s.subset(test_idx))          # P(anomaly) per test session
```

Any new model only needs `fit(train_sessions)` and `predict_scores(sessions)`. A `Sessions` object holds the
CSR-style arrays `offsets`, `events` (template ids), `labels` and `keys`.
