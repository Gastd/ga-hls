#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Repo structure..."
test -d docs/rq
test -d results/summary
test -d results/timings
echo "OK"

echo "[2/6] Configs parse as JSON..."
python - <<'PY'
import json, glob, sys
paths = glob.glob("configs/*.json")
if not paths:
    print("No configs/*.json found", file=sys.stderr)
    sys.exit(1)
for p in paths:
    with open(p) as f:
        json.load(f)
print(f"OK: parsed {len(paths)} configs")
PY

echo "[3/6] Docker compose file exists..."
test -f docker/docker-compose.yml
echo "OK"

echo "[4/6] CLI import check (local python)..."
python - <<'PY'
import ga_hls  # package import name may differ; adjust if needed
print("Imported package OK")
PY || echo "NOTE: local import failed (ok if you run via Docker only)."

echo "[5/6] Results presence (non-fatal)..."
if ls results/summary/*.csv >/dev/null 2>&1; then
  echo "Found summary CSVs"
else
  echo "WARNING: results/summary/*.csv not found yet"
fi

echo "[6/6] Done."
