import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO = Path(__file__).resolve().parents[2]
EFF = REPO / "replication" / "evaluation_inputs" / "effectiveness"
OUT = REPO / "replication" / "evaluation_inputs" / "effectiveness" / "migration_verification.csv"

DOCKER_COMPOSE = ["docker", "compose", "-f", "docker/docker-compose.yml", "run", "--rm", "ga-hls"]

REL_SYMS = {">", "<", ">=", "<=", "==", "!="}
LOGICALS = {"And", "Or", "Implies"}
QUANTS = {"ForAll", "Exists"}

@dataclass(frozen=True)
class V1Spec:
    kind: str  # "numeric" or "op"
    domain: Tuple  # (lo,hi) or tuple(domains)

@dataclass(frozen=True)
class V2Change:
    kind: str  # "numeric"|"quantifier"|"logical"|"relational"
    domain: Tuple  # (lo,hi) or tuple(domains)
    pos: int

def have_local_cli() -> bool:
    return shutil.which("ga-hls") is not None

def run_ga_hls(args: List[str]) -> str:
    if have_local_cli():
        cmd = ["ga-hls"] + args
        return subprocess.check_output(cmd, cwd=str(REPO), text=True)
    cmd = DOCKER_COMPOSE + args
    return subprocess.check_output(cmd, cwd=str(REPO), text=True)

def parse_explain_positions(output: str) -> Dict[int, Dict[str,str]]:
    """
    Parses the 'explain-positions' table to map position -> {nodetype,node,features}.
    """
    rows: Dict[int, Dict[str,str]] = {}
    for ln in output.splitlines():
        s = ln.rstrip()
        if not s.strip():
            continue
        m = re.match(r"^\s*(\d+)\s+(\w+)\s+(.*?)\s+(\S+)\s*$", s)
        if not m:
            continue
        pos = int(m.group(1))
        nodetype = m.group(2)
        node = m.group(3)
        feat = m.group(4)
        rows[pos] = {"nodetype": nodetype, "node": node, "features": feat}
    return rows

def parse_number_from_node_text(n: str) -> Optional[float]:
    # Common patterns: "8", "120.0", maybe "IntVal(5)" or "RealVal(0.0)" etc.
    n = n.strip()
    # prefer plain float/int
    if re.fullmatch(r"-?\d+(\.\d+)?", n):
        return float(n)
    # fallback: capture a number in the string
    m = re.search(r"(-?\d+(?:\.\d+)?)", n)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def classify_v1_op_domain(dom: Tuple[str, ...]) -> str:
    s = set(dom)
    if s.issubset(QUANTS):
        return "quantifier"
    if s.issubset(LOGICALS):
        return "logical"
    if s.issubset(REL_SYMS):
        return "relational"
    return "unknown"

def load_v1(v1_path: Path) -> List[V1Spec]:
    d = json.loads(v1_path.read_text())
    specs: List[V1Spec] = []
    for _, spec in d.items():
        if not (isinstance(spec, list) and len(spec) == 2):
            continue
        typ, dom = spec[0], spec[1]
        if typ in ("int", "float") and isinstance(dom, list) and len(dom) == 2:
            lo, hi = float(dom[0]), float(dom[1])
            specs.append(V1Spec("numeric", (lo, hi)))
        elif typ == "op" and isinstance(dom, list):
            specs.append(V1Spec("op", tuple(dom)))
        else:
            specs.append(V1Spec("unknown", tuple(dom) if isinstance(dom, list) else (str(dom),)))
    return specs

def load_v2(v2_path: Path) -> Tuple[List[V2Change], Dict]:
    d = json.loads(v2_path.read_text())
    changes = []
    allowed = d["mutation"]["allowed_changes"]
    for k, v in allowed.items():
        pos = int(k)
        if "numeric" in v:
            lo, hi = float(v["numeric"][0]), float(v["numeric"][1])
            changes.append(V2Change("numeric", (lo, hi), pos))
        elif "quantifier" in v:
            changes.append(V2Change("quantifier", tuple(v["quantifier"]), pos))
        elif "logical" in v:
            changes.append(V2Change("logical", tuple(v["logical"]), pos))
        elif "relational" in v:
            changes.append(V2Change("relational", tuple(v["relational"]), pos))
        else:
            changes.append(V2Change("unknown", tuple(), pos))
    return changes, d

def multiset(xs: List[Tuple]) -> Dict[Tuple,int]:
    m: Dict[Tuple,int] = {}
    for x in xs:
        m[x] = m.get(x, 0) + 1
    return m

def verify_one(v2_path: Path) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    v1_path = Path(str(v2_path) + ".v1")
    if not v1_path.exists():
        return False, [f"missing_v1_backup:{v1_path.name}"]

    v1_specs = load_v1(v1_path)
    v2_changes, v2 = load_v2(v2_path)

    # 1) Count match
    if len(v1_specs) != len(v2_changes):
        notes.append(f"count_mismatch:v1={len(v1_specs)} v2={len(v2_changes)}")

    # 2) Domain/range equivalence (ignore positions; compare as multisets)
    v1_norm: List[Tuple] = []
    for s in v1_specs:
        if s.kind == "numeric":
            v1_norm.append(("numeric", s.domain))
        elif s.kind == "op":
            k = classify_v1_op_domain(s.domain)
            v1_norm.append((k, tuple(s.domain)))
        else:
            v1_norm.append((s.kind, s.domain))

    v2_norm: List[Tuple] = []
    for c in v2_changes:
        v2_norm.append((c.kind, c.domain))

    if multiset(v1_norm) != multiset(v2_norm):
        notes.append("domain_mismatch")
        notes.append(f"v1={multiset(v1_norm)}")
        notes.append(f"v2={multiset(v2_norm)}")

    # 3) AST node type compatibility checks via explain-positions
    try:
        explain = run_ga_hls(["explain-positions", "--config", str(v2_path)])
        rows = parse_explain_positions(explain)
    except Exception as e:
        return False, [f"explain_positions_failed:{e!r}"]

    def want_nodetypes(kind: str) -> Tuple[str, ...]:
        if kind == "numeric":
            return ("IntConst", "RealConst")
        if kind == "quantifier":
            return ("ForAll", "Exists")
        if kind == "logical":
            return ("And", "Or", "Implies")
        if kind == "relational":
            return ("RelOp",)
        return tuple()

    for ch in v2_changes:
        if ch.pos not in rows:
            notes.append(f"missing_pos_in_explain:{ch.pos}")
            continue
        nt = rows[ch.pos]["nodetype"]
        allowed = want_nodetypes(ch.kind)
        if allowed and nt not in allowed:
            notes.append(f"type_mismatch:pos{ch.pos} kind={ch.kind} nodetype={nt}")

        # 4) numeric sanity: constant at pos within allowed range
        if ch.kind == "numeric":
            val = parse_number_from_node_text(rows[ch.pos]["node"])
            if val is None:
                notes.append(f"numeric_parse_failed:pos{ch.pos} node={rows[ch.pos]['node']!r}")
            else:
                lo, hi = ch.domain
                if not (lo <= val <= hi):
                    notes.append(f"numeric_out_of_range:pos{ch.pos} val={val} range=[{lo},{hi}]")

    ok = len(notes) == 0
    return ok, notes

def main():
    rows_out = []
    total = 0
    passed = 0

    for req_dir in sorted([p for p in EFF.iterdir() if p.is_dir()]):
        for v2_path in sorted(req_dir.glob("exp*.json")):
            if v2_path.name.endswith(".v1"):
                continue
            total += 1
            ok, notes = verify_one(v2_path)
            if ok:
                passed += 1
            rows_out.append({
                "requirement": req_dir.name,
                "config": v2_path.name,
                "ok": "YES" if ok else "NO",
                "notes": " | ".join(notes) if notes else ""
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["requirement","config","ok","notes"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {OUT}")
    print(f"Passed {passed}/{total}")
    if passed != total:
        print("Failures:")
        for r in rows_out:
            if r["ok"] == "NO":
                print(f" - {r['requirement']}/{r['config']}: {r['notes']}")

if __name__ == "__main__":
    main()
