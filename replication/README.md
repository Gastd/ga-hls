# Replication package contents

This directory contains the replication package for the paper, including:
- inputs to run `ga-hls` for the evaluation,
- scripts to compute the reported results, and
- the collected artifacts and derived summaries.

## Directory overview

- `analysis/`
  - `analysis/scripts/`: scripts used to compute the evaluation results (RQ1/RQ2) from the collected artifacts
  - `analysis/data/`: trace CSV files used by the analysis scripts (when applicable)

- `evaluation_inputs/`
  Inputs required to run `ga-hls` (properties, trace encodings, and run configurations).
  - `evaluation_inputs/effectiveness/`: inputs for the main evaluation runs (paper Exp1–Exp34; used in RQ1 and RQ2)
  - `evaluation_inputs/sensitivity/`: inputs for the sensitivity study runs (used in RQ3)

- `docs/`
  Documentation and registries:
  - `docs/REPLICATION.md`: replication-package entry point
  - `docs/rq/`: per-research-question documentation
  - `docs/experiments/`: registries mapping runs to artifact paths

- `results/`
  Collected run artifacts and derived summaries:
  - `results/raw/`: raw `ga-hls` outputs
    - `results/raw/exp1` … `exp34`: main evaluation runs (paper Exp IDs)
    - `results/raw/sensitivity/…`: sensitivity runs (subject × variant × seed)
  - `results/summary/`: derived tables (e.g., precision/recall and sensitivity summaries)
  - `results/timings/`: per-run timing JSON files
  - `results/trees/`: extracted tree artifacts (tool-rendered PNGs, Weka outputs)
  - `results/arff/`: extracted ARFF datasets (representative subsets)
  - `results/logs/`: per-run report/log JSON files

- `scripts/`
  Utility scripts to validate the replication package (e.g., link/path checks).

## Where to start

See `docs/REPLICATION.md` for the main entry point and pointers to the artifacts used for each research question.
