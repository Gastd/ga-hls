# How to reproduce the evaluation results

This document explains how to:
1) rerun `ga-hls` for the **main evaluation experiments (Exp1–Exp34)**,
2) rerun the **analysis scripts** for RQ1/RQ2 (precision/recall, timing stats, boxplot),
3) rerun the **sensitivity study** (RQ3).

> Notes
> - The repository already contains the **raw outputs** and extracted artifacts under `replication/results/`.
> - Re-running all experiments can be time-consuming. This guide prioritizes clarity over full automation.

---

## 0) Environment setup

### Docker (recommended to run ga-hls)
From the repository root:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm ga-hls --help
```

---

## 1) Re-running ga-hls for the main experiments (Exp1–Exp34)

Inputs for the main experiments are under:

- `replication/evaluation_inputs/effectiveness/*/`

Each requirement folder contains:
- the ThEodorE property file (`*.py`)
- experiment configs (`exp1.json`, `exp2.json`, …)
- a `run.sh` helper script in most cases

### 1.1 Run a single experiment via Docker

Example (AT1 / exp1):

```bash
docker compose -f docker/docker-compose.yml run --rm ga-hls \
  run --config replication/evaluation_inputs/effectiveness/AT1/exp1.json
```

The command should produce a fresh output folder (depending on your config’s `output_dir`).

---

## 2) RQ1 — Precision/recall (Exp1–Exp34)

The analysis scripts and trace CSVs are here:

- Scripts: `replication/analysis/scripts/`
- Traces: `replication/analysis/data/`

Each `Accuracy*.py` script computes precision/recall for one trace–requirement pair.
Some scripts support multiple experiment modes (e.g., `exp = 1` vs `exp = 2`).

### 2.1 Run one precision/recall script

Example:

```bash
python replication/analysis/scripts/AccuracyAT1.py
```

### 2.2 Re-generate the table used in the paper

If your workflow expects a CSV summary, write outputs to:

- `replication/results/summary/precision_recall.csv`

(If the CSV already exists, you can compare your outputs against it.)

---

## 3) RQ2 — Efficiency (timings over Exp1–Exp34)

The timing analysis script is:

- `replication/analysis/scripts/RQ2.py`

Run:

```bash
python replication/analysis/scripts/RQ2.py
```

---

## 4) Boxplot for precision/recall (paper figure)

The plotting script is:

- `replication/analysis/scripts/printBoxplot.py`

Run:

```bash
python replication/analysis/scripts/printBoxplot.py
```

Save the generated figure under:

- `replication/results/figures/` (create if needed)

---

## 5) RQ3 — Sensitivity study

RQ3 inputs are under:

- `replication/evaluation_inputs/sensitivity/`

Each scenario folder contains multiple configs (e.g., `*_B.json`, `*_G.json`, `*_O*.json`) and a `property_*.py`.

### 5.1 Run one sensitivity configuration

Example:

```bash
docker compose -f docker/docker-compose.yml run --rm ga-hls \
  run --config replication/evaluation_inputs/sensitivity/phi3/phi3_B.json
```

Repeat for other configurations and (if applicable) multiple seeds.

### 5.2 Record / summarize results

Store summaries (e.g., per seed / per configuration aggregates) under:

- `replication/results/summary/sensitivity.csv`

