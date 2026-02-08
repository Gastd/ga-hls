# RQ2 — Efficiency (runtime)

This page documents:
- total runtime per experiment
- breakdown by pipeline stage (GA, trace-checker, J48)
- where to find the timing tables

## Artifacts
- Timing table: `results/timings/table8_runtime.csv`
- Logs (if available): `results/logs/exp*/`

---

## Artifacts and provenance (RQ2)

**Scope.** RQ2 is computed over the **main evaluation runs** (paper **Exp1–Exp34**), i.e., the same runs used for RQ1.

**Per-run timing artifacts**
- Timing logs per experiment: `replication/results/timings/exp*_timespan.json`

**Experiment registry**
- `replication/docs/experiments/experiments.csv` (maps Exp IDs to timing files and raw outputs)

**Analysis script**
- Runtime aggregation/statistics script: `replication/analysis/scripts/RQ2.py`

**Raw outputs (for traceability)**
- Full raw run outputs: `replication/results/raw/exp1` … `replication/results/raw/exp34`
