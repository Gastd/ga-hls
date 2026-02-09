# Replication package (ga-hls)

This repository accompanies the paper and contains the material needed to inspect the reported results and, where feasible, reproduce the analyses.

## Repository structure (replication package root)

All replication artifacts are stored under the `replication/` directory:

- `replication/evaluation_inputs/`  
  Inputs to run `ga-hls` (properties, trace encodings, and configs).
  - `replication/evaluation_inputs/effectiveness/`: inputs for the **main evaluation runs** (Exp1–Exp34)
  - `replication/evaluation_inputs/sensitivity/`: inputs for the **sensitivity runs** (RQ3)

- `replication/analysis/`  
  Scripts and supporting data used to compute paper-level summaries (e.g., precision/recall computations, runtime aggregation, plots).

- `replication/results/`  
  Collected run artifacts and derived summaries:
  - `replication/results/raw/` raw outputs from `ga-hls` runs
    - `replication/results/raw/exp1` … `exp34`: **main evaluation runs** (paper Exp IDs)
    - `replication/results/raw/sensitivity/…`: **sensitivity runs** (subject × variant × seed)
  - `replication/results/trees/`: extracted decision-tree artifacts (e.g., tool-rendered PNGs, Weka outputs)
  - `replication/results/arff/`: extracted ARFF datasets (representative subsets)
  - `replication/results/timings/`: per-run timing JSON files
  - `replication/results/logs/`: per-run report/log JSON files
  - `replication/results/summary/`: derived tables used in the paper (e.g., precision/recall)

- `replication/docs/`  
  Documentation and indices:
  - `replication/docs/rq/`: documentation per research question
  - `replication/docs/experiments/effectiveness_runs.csv`: registry for Exp1–Exp34 (main evaluation runs)
  - `replication/docs/experiments/sensitivity_runs.csv`: registry for sensitivity runs

## Quick start (Docker)

Build:
```bash
docker compose -f docker/docker-compose.yml build
```

Check CLI:
```bash
docker compose -f docker/docker-compose.yml run --rm ga-hls --help
```

## Results overview

### Effectiveness/Efficiency evaluation runs (Exp1–Exp34)
- Precision/recall per experiment: `replication/results/summary/precision_recall.csv`
- Per-experiment artifacts: `replication/results/trees/`, `replication/results/arff/`, `replication/results/timings/`, `replication/results/logs/`
- Registry: `replication/docs/experiments/effectiveness_runs.csv`

### Sensitivity runs (RQ3)
- Raw outputs: `replication/results/raw/sensitivity/`
- Registry: `replication/docs/experiments/sensitivity_runs.csv`
- Derived sensitivity summaries (if used in the paper): `replication/results/summary/`

## Research questions
- RQ1 (effectiveness): `replication/docs/rq/RQ1_effectiveness.md`
- RQ2 (efficiency): `replication/docs/rq/RQ2_efficiency.md`
- RQ3 (sensitivity): `replication/docs/rq/RQ3_sensitivity.md`

## Reproducing results

See `docs/REPRODUCE.md` for step-by-step commands to rerun ga-hls and the analysis scripts.
