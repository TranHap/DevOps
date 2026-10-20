#!/bin/sh
# Optional / sensitivity runs (see README "Extra runs"). Run from part1/ with the venv active.
set -e
for d in bgl hdfs; do
  for s in 0 1 2 3 4; do
    python -m src.run_part1 --dataset $d --seed $s --stage features
    python -m src.run_part1 --dataset $d --seed $s --stage train
    python -m src.run_part1 --dataset $d --seed $s --stage eval
  done
done
# HDFS with the replication package's (loglizer 'uniform', per-class chronological) split
python -m src.run_part1 --dataset hdfs --split-method loglizer_uniform --stage features
python -m src.run_part1 --dataset hdfs --split-method loglizer_uniform --stage train
python -m src.run_part1 --dataset hdfs --split-method loglizer_uniform --stage eval
# diagnostic LSTM test curves (never used for model selection)
python -m src.run_part1 --dataset hdfs --lstm-test-curve
python -m src.run_part1 --dataset bgl --lstm-test-curve
# LSTM ablations towards the replication package (RESULTS.md section 10)
python -m src.run_part1 --dataset hdfs --lstm-package-windows
python -m src.run_part1 --dataset bgl --lstm-package-windows
python -m src.run_part1 --dataset bgl --lstm-package-faithful   # per-session steps: infeasible on HDFS
# parsing sensitivity: Qin regexes without estimation-stage narrowing (BGL)
python -m src.parse --config bgl_qinfull
python -m src.report
