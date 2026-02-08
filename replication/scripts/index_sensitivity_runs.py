import csv
from pathlib import Path

ROOT = Path("results/raw/sensitivity")
OUT = Path("docs/experiments/sensitivity_runs.csv")

def first(glob):
    return next(iter(glob), None)

def main():
    if not ROOT.exists():
        raise SystemExit(f"Missing {ROOT}. Did you copy sensitivity raw outputs?")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # structure: subject/variant/seedX/timestamped_run_dir/
    for subject_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        subject = subject_dir.name
        for variant_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            variant = variant_dir.name
            for seed_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
                seed = seed_dir.name  # e.g., seed0
                # timestamped run dirs under seed
                run_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir()])
                for run_dir in run_dirs:
                    timespan = run_dir / "timespan.json"
                    report = run_dir / "report.json"
                    summary = run_dir / "summary.json"
                    arff_per100 = first(run_dir.glob("*_per100.arff"))
                    weka_outs = list(run_dir.glob("J48-data-*.out"))

                    rows.append({
                        "subject": subject,
                        "variant": variant,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "timespan_json": str(timespan) if timespan.exists() else "",
                        "report_json": str(report) if report.exists() else "",
                        "summary_json": str(summary) if summary.exists() else "",
                        "arff_per100": str(arff_per100) if arff_per100 else "",
                        "weka_out_count": str(len(weka_outs)),
                    })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "subject","variant","seed","run_dir",
            "timespan_json","report_json","summary_json",
            "arff_per100","weka_out_count"
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT} with {len(rows)} rows")

if __name__ == "__main__":
    main()
