#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

TARGET_NAMES = ["exp1.json", "exp2.json", "exp3.json"]
ROOT_DEFAULT = Path("replication/evaluation_inputs/effectiveness")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Restore exp*.json from exp*.json.v1 under replication/evaluation_inputs/effectiveness/"
    )
    ap.add_argument("--root", default=str(ROOT_DEFAULT), help="Root directory to operate on")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without modifying files")
    ap.add_argument("--keep-missing", action="store_true",
                    help="Do not fail if some exp*.json.v1 backups are missing (still reports them)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    # Collect all directories that contain any exp*.json.v1
    dirs = set()
    for t in TARGET_NAMES:
        for p in root.rglob(t + ".v1"):
            dirs.add(p.parent)

    if not dirs:
        print(f"ERROR: no backups found matching exp*.json.v1 under {root}", file=sys.stderr)
        return 3

    dirs = sorted(dirs)

    removed = []
    restored = []
    missing_backups = []
    skipped_existing = []

    for d in dirs:
        for name in TARGET_NAMES:
            cur = d / name
            bak = d / (name + ".v1")

            # Remove current file if present
            if cur.exists():
                removed.append(cur)
                if not args.dry_run:
                    cur.unlink()

            # Restore from backup
            if bak.exists():
                # rename backup to current
                restored.append((bak, cur))
                if not args.dry_run:
                    bak.rename(cur)
            else:
                missing_backups.append(bak)
                if not args.keep_missing:
                    # We'll report later and exit non-zero
                    pass

    # Report
    print("== SUMMARY ==")
    print(f"Root: {root}")
    print(f"Dirs touched: {len(dirs)}")
    print(f"Removed: {len(removed)} exp*.json")
    print(f"Restored: {len(restored)} exp*.json.v1 -> exp*.json")
    if missing_backups:
        print(f"Missing backups: {len(missing_backups)} (first 20 shown)")
        for p in missing_backups[:20]:
            print(f"  - {p}")

    if args.dry_run:
        print("\n(DRY RUN) Actions that would be taken:")
    else:
        print("\nActions taken:")

    if removed:
        print("\nRemoved:")
        for p in removed[:50]:
            print(f"  - {p}")
        if len(removed) > 50:
            print(f"  ... ({len(removed)-50} more)")

    if restored:
        print("\nRestored:")
        for src, dst in restored[:50]:
            print(f"  - {src}  ->  {dst}")
        if len(restored) > 50:
            print(f"  ... ({len(restored)-50} more)")

    # Exit code: fail if backups missing and keep-missing is not enabled
    if missing_backups and not args.keep_missing:
        print("\nERROR: Some exp*.json.v1 backups were missing. Use --keep-missing to ignore.", file=sys.stderr)
        return 4

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
