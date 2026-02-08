# Replication scripts and data

This directory contains the scripts used to compute the evaluation results reported in the paper.

- `replication/scripts/`:
  Analysis scripts for RQ1/RQ2 (precision/recall, runtime aggregation, plots).
- `replication/data/`:
  Input trace CSV files used by the analysis scripts (when applicable).

Generated artifacts should be written under the replication package `results/` directory:
- `replication/results/summary/` (tables such as precision/recall)
- `replication/results/timings/` (runtime tables)
- `replication/results/figures/` (plots)
