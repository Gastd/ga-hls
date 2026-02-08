# RQ3 — Sensitivity (mutation choices & budget)

This page documents:
- the sensitivity study settings (mutation spaces, mutation budget)
- the outputs: runtime, stability, and DT complexity
- where to find the artifacts / summary

## Artifacts
- Sensitivity summaries: `results/summary/rq3_*.csv` (to be added)

---

## Artifacts and provenance (RQ3)

**Scope.** RQ3 is evaluated via a **sensitivity study** consisting of multiple `ga-hls` runs across:
- subjects (AT53_AT119, RR, phi2, phi3, phi4),
- configuration variants (e.g., B/G/G2/G3/O*), and
- multiple random seeds.

**Run inputs (configs and properties)**
- Sensitivity configurations and properties: `replication/evaluation_inputs/sensitivity/`

**Run registry (maps subject/variant/seed to raw outputs)**
- `replication/docs/experiments/sensitivity_runs.csv`

**Raw outputs**
- Full raw sensitivity outputs: `replication/results/raw/sensitivity/`

**Derived summaries (if referenced in the paper text)**
- Sensitivity summary tables: `replication/results/summary/` (see files prefixed with `sensitivity`)
