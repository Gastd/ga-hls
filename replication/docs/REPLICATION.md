# Replication package (ga-hls)

This repository accompanies the paper and contains:
- configs used in the evaluation (`configs/`)
- result artifacts (`results/`)
- documentation per research question (`docs/rq/`)
- scripts to verify and (where feasible) reproduce analyses (`scripts/`)

## Quick start (Docker)
1) Build:
   docker compose -f docker/docker-compose.yml build

2) Check CLI:
   docker compose -f docker/docker-compose.yml run --rm ga-hls --help

## Results
See `results/summary/` (precision/recall) and `results/timings/` (runtime tables).
Per-experiment artifacts are under `results/trees/` and `results/arff/`.

## Research questions
- RQ1: `docs/rq/RQ1_effectiveness.md`
- RQ2: `docs/rq/RQ2_efficiency.md`
- RQ3: `docs/rq/RQ3_sensitivity.md`
