from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_j48_out(out_text: str) -> Dict[str, Any]:
    """
    Parse Weka J48 textual output (.out) to extract coarse tree statistics.
    This is heuristic but stable enough for sensitivity comparisons.
    """
    lines = [ln.rstrip("\n") for ln in out_text.splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]

    # Root split: first “rule-like” line after the model header
    root = None
    seen_model_header = False
    for ln in nonempty:
        low = ln.strip().lower()
        if "classifier model" in low:
            seen_model_header = True
            continue
        if not seen_model_header:
            continue
        # skip obvious footer/meta lines
        if low.startswith(("=== ", "number of leaves", "size of the tree", "time taken", "correctly classified")):
            continue
        # heuristic: tree rule lines usually contain ":" or a comparison operator
        if (":" in ln) or any(op in ln for op in ["<=", ">=", "<", ">", "="]):
            root = ln.strip()
            break

    # If the first "root" line starts with ":" it is a trivial (no-split) tree.
    if root is not None and root.lstrip().startswith(":"):
        root = None

    # Depth: maximum number of '|' markers on any tree line
    depth = 0
    nodes = 0
    leaves = 0
    for ln in nonempty:
        if "|" in ln or ":" in ln:
            nodes += 1
        if ":" in ln:
            leaves += 1
        depth = max(depth, ln.count("|"))
    
    # If there's no root split, treat as depth 0.
    if root is None:
        depth = 0

    return {
        "tree_depth": depth if nodes > 0 else None,
        "tree_nodes": nodes if nodes > 0 else None,
        "tree_leaves": leaves if leaves > 0 else None,
        "root_split": root,
    }

def write_summary(output_dir: str, summary: Dict[str, Any]) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "summary.json"
    p.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return str(p)
