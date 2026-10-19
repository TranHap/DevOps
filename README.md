# LOG6309E Group Assignment - Log-based Anomaly Detection

Replication + extension of Wu, Li & Khomh, "On the Effectiveness of Log Representation
for Log-based Anomaly Detection" (EMSE 2023). Paper: `references/Wu2023_LogRepresentation.pdf`.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows (Git Bash)
# Original replication package (reference only):
git clone https://github.com/mooselab/suppmaterial-LogRepForAnomalyDetection references/replication-package
```

Datasets (Loghub, Zenodo record 8196385) go in `data/raw/HDFS/HDFS.log` and `data/raw/BGL/BGL.log`:
```bash
curl -L -o data/raw/HDFS_v1.zip https://zenodo.org/api/records/8196385/files/HDFS_v1.zip/content
curl -L -o data/raw/BGL.zip     https://zenodo.org/api/records/8196385/files/BGL.zip/content
```

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1. Parsing (Drain + regex preprocessing) | `src/parse.py --dataset HDFS\|BGL` | `data/parsed/<DS>/` |
| 2. Partition, split (seeds 0-9), MCV | `src/partition.py`, `src/split.py`, `src/mcv.py` | `data/sessions/`, `data/splits/`, `data/features/` |
| 3. Other representations (Word2Vec, ...) | TODO | |
| 4. Models + evaluation | TODO | |

## Experimental setup (following the paper)
| Dataset | Grouping | Split |
|---|---|---|
| HDFS | session = block_id (labels: `anomaly_label.csv`) | 70/30, shuffled |
| BGL | fixed 6-hour window, anomalous if any alert line | 80/20, shuffled |
