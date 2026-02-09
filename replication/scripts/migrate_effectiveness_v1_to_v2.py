import json, re, shutil, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO = Path(__file__).resolve().parents[2]
EFF = REPO / "replication" / "evaluation_inputs" / "effectiveness"

TR_AT = REPO / "replication" / "analysis" / "data" / "tracesAT.csv"
TR_CC = REPO / "replication" / "analysis" / "data" / "tracesCC.csv"
TR_RR = REPO / "replication" / "analysis" / "data" / "traceRR.csv"

GA_DEFAULTS = dict(
    population_size=250,
    generations=10,
    crossover_rate=0.95,
    mutation_rate=0.10,
    seed=0,
)

REL_SYMS = ["<=", ">=", "<", ">", "==", "!="]

def traces_for(req: str) -> Path:
    u = req.upper()
    if u.startswith("AT"):
        return TR_AT
    if u.startswith("CC"):
        return TR_CC
    if u.startswith("RR"):
        return TR_RR
    raise ValueError(f"Cannot infer traces for {req}")

def find_requirement_py(req_dir: Path) -> Path:
    cands = sorted(req_dir.glob("*.py"))
    if not cands:
        raise FileNotFoundError(f"No .py requirement in {req_dir}")
    # Prefer files that look like requirement exports
    for p in cands:
        if p.stem.upper().startswith(req_dir.name.upper()):
            return p
    return cands[0]

def run_explain_positions(cfg: Path) -> str:
    # Prefer local CLI if installed; else docker compose.
    if shutil.which("ga-hls"):
        cmd = ["ga-hls", "explain-positions", "--config", str(cfg)]
        return subprocess.check_output(cmd, cwd=str(REPO), text=True)
    cmd = ["docker", "compose", "-f", "docker/docker-compose.yml",
           "run", "--rm", "ga-hls",
           "explain-positions", "--config", str(cfg)]
    return subprocess.check_output(cmd, cwd=str(REPO), text=True)

def parse_positions(out: str) -> List[Dict]:
    rows = []
    for ln in out.splitlines():
        s = ln.strip()
        if not s:
            continue
        if not re.match(r"^\d+\s+\w+", s):
            continue
        parts = s.split()
        pos = int(parts[0])
        nodetype = parts[1]
        feature = parts[-1]
        node = " ".join(parts[2:-1])
        rows.append(dict(pos=pos, nodetype=nodetype, node=node, feature=feature))
    return rows

def extract_rel_sym(expr: str) -> Optional[str]:
    for sym in REL_SYMS:
        if sym in expr:
            return sym
    # handle common pretty-prints
    if " <= " in expr: return "<="
    if " >= " in expr: return ">="
    if " < " in expr: return "<"
    if " > " in expr: return ">"
    return None

def best_numeric_match(cands: List[Tuple[int, float]], lo: float, hi: float, used: set) -> Tuple[int, str]:
    mid = (lo + hi) / 2.0
    usable = [(p,v) for (p,v) in cands if p not in used]
    if not usable:
        usable = cands
    in_range = [(p,v) for (p,v) in usable if lo <= v <= hi]
    pool = in_range if in_range else usable

    # Prefer exact endpoint matches
    for endpoint in (lo, hi):
        exact = [(p,v) for (p,v) in pool if abs(v - endpoint) < 1e-9]
        if len(exact) == 1:
            return exact[0][0], "exact-endpoint"

    # Closest to midpoint
    pool_sorted = sorted(pool, key=lambda pv: (abs(pv[1]-mid), pv[0]))
    chosen = pool_sorted[0][0]
    note = "in-range" if in_range else "closest-out-of-range"
    if in_range and len(in_range) > 1:
        note += f"-ambiguous({len(in_range)})"
    return chosen, note

def classify_op_domain(domain: List[str]) -> str:
    s = set(domain)
    if s.issubset({"ForAll", "Exists"}):
        return "quantifier"
    if s.issubset({"And", "Or", "Implies"}):
        return "logical"
    if s.issubset({">", "<", ">=", "<=", "==", "!="}):
        return "relational"
    return "unknown"

def choose_quant_pos(quant_nodes: List[Tuple[int,str]], domain: List[str], used: set) -> Tuple[int,str]:
    # Prefer ForAll positions first when flipping ForAll/Exists (matches CC3 intent)
    order = []
    for p,t in quant_nodes:
        if p in used: continue
        order.append((0 if t=="ForAll" else 1, p, t))
    if not order:
        # fallback: allow reuse
        for p,t in quant_nodes:
            order.append((0 if t=="ForAll" else 1, p, t))
    order.sort()
    return order[0][1], f"prefer-ForAll({order[0][2]})"

def choose_logical_pos(logical_nodes: List[Tuple[int,str]], domain: List[str], used: set) -> Tuple[int,str]:
    # Prefer Or if domain includes And/Or (common “flip OR→AND”)
    want_or = set(domain) == {"And","Or"} or ("And" in domain and "Or" in domain)
    order = []
    for p,t in logical_nodes:
        if p in used: continue
        prio = 0 if (want_or and t=="Or") else 1
        order.append((prio, p, t))
    if not order:
        for p,t in logical_nodes:
            prio = 0 if (want_or and t=="Or") else 1
            order.append((prio, p, t))
    order.sort()
    return order[0][1], f"prefer-Or({order[0][2]})" if want_or else f"first({order[0][2]})"

def choose_relop_pos(relops: List[Tuple[int,str]], domain: List[str], used: set) -> Tuple[int,str]:
    # Prefer nodes that currently contain > or < if domain is {>,<}
    want = set(domain)
    order = []
    for p,expr in relops:
        if p in used: continue
        sym = extract_rel_sym(expr) or ""
        prio = 0 if sym in want else 1
        order.append((prio, p, sym))
    if not order:
        for p,expr in relops:
            sym = extract_rel_sym(expr) or ""
            prio = 0 if sym in want else 1
            order.append((prio, p, sym))
    order.sort()
    return order[0][1], f"match-relop({order[0][2]})"

def migrate_file(req_dir: Path, exp_json: Path, dry_run: bool) -> Tuple[bool,str]:
    v1 = json.loads(exp_json.read_text())

    # skip already-v2
    if isinstance(v1, dict) and "input" in v1 and "mutation" in v1 and "ga" in v1:
        return False, "already-v2"

    req = req_dir.name
    py = find_requirement_py(req_dir)
    tr = traces_for(req)

    tmp = req_dir / ".tmp_explain.json"
    tmp.write_text(json.dumps({
        "input": {"requirement_file": str(py), "traces_file": str(tr), "output_dir": str(REPO/"replication/results/_tmp")},
        "ga": {"population_size": 1, "generations": 1, "crossover_rate": 0.0, "mutation_rate": 0.0, "seed": 0},
        "mutation": {
            "max_mutations": 1,
            "enable_numeric_perturbation": True,
            "enable_relop_flip": True,
            "enable_logical_flip": True,
            "enable_quantifier_flip": True,
            "allowed_positions": [],
            "allowed_changes": {}
        }
    }, indent=2))

    out = run_explain_positions(tmp)
    tmp.unlink(missing_ok=True)

    rows = parse_positions(out)

    int_nodes: List[Tuple[int,float]] = []
    real_nodes: List[Tuple[int,float]] = []
    quant_nodes: List[Tuple[int,str]] = []
    logical_nodes: List[Tuple[int,str]] = []
    relop_nodes: List[Tuple[int,str]] = []

    for r in rows:
        t = r["nodetype"]
        p = r["pos"]
        n = r["node"]

        if t == "IntConst":
            try: int_nodes.append((p, float(int(n))))
            except: pass
        if t == "RealConst":
            try: real_nodes.append((p, float(n)))
            except: pass
        if t in ("ForAll","Exists"):
            quant_nodes.append((p,t))
        if t in ("And","Or","Implies"):
            logical_nodes.append((p,t))
        if t == "RelOp":
            relop_nodes.append((p,n))

    used_num=set(); used_q=set(); used_l=set(); used_r=set()
    allowed_positions: List[int] = []
    allowed_changes: Dict[str, Dict] = {}
    notes: List[str] = []

    def add_pos(pos: int, change: Dict, note: str):
        allowed_positions.append(pos)
        allowed_changes[str(pos)] = change
        notes.append(note)

    for k, spec in v1.items():
        if not (isinstance(spec, list) and len(spec)==2):
            notes.append(f"{k}:bad-format")
            continue
        typ, dom = spec[0], spec[1]

        if typ in ("int","float") and isinstance(dom, list) and len(dom)==2:
            lo,hi = float(dom[0]), float(dom[1])
            cands = int_nodes if typ=="int" else real_nodes
            if not cands:
                # fallback: allow matching across types if only one exists
                cands = real_nodes if real_nodes else int_nodes
            if not cands:
                notes.append(f"{k}:no-numeric-nodes")
                continue
            pos, why = best_numeric_match(cands, lo, hi, used_num)
            used_num.add(pos)
            add_pos(pos, {"numeric":[lo,hi]}, f"{k}->{pos}:{why}")
            continue

        if typ == "op" and isinstance(dom, list):
            kind = classify_op_domain(dom)
            if kind == "quantifier":
                if not quant_nodes:
                    notes.append(f"{k}:no-quantifiers")
                    continue
                pos, why = choose_quant_pos(quant_nodes, dom, used_q)
                used_q.add(pos)
                add_pos(pos, {"quantifier": dom}, f"{k}->{pos}:{why}")
                continue
            if kind == "logical":
                if not logical_nodes:
                    notes.append(f"{k}:no-logicals")
                    continue
                pos, why = choose_logical_pos(logical_nodes, dom, used_l)
                used_l.add(pos)
                add_pos(pos, {"logical": dom}, f"{k}->{pos}:{why}")
                continue
            if kind == "relational":
                if not relop_nodes:
                    notes.append(f"{k}:no-relops")
                    continue
                pos, why = choose_relop_pos(relop_nodes, dom, used_r)
                used_r.add(pos)
                add_pos(pos, {"relational": dom}, f"{k}->{pos}:{why}")
                continue
            notes.append(f"{k}:unknown-op-domain")
            continue

        notes.append(f"{k}:unknown-type-{typ}")

    enable_numeric = any("numeric" in d for d in allowed_changes.values())
    enable_relop = any("relational" in d for d in allowed_changes.values())
    enable_logical = any("logical" in d for d in allowed_changes.values())
    enable_quant = any("quantifier" in d for d in allowed_changes.values())

    v2 = {
        "input": {
            "requirement_file": str(py),
            "traces_file": str(tr),
            "output_dir": str(REPO / "replication" / "results" / "_rerun" / req / exp_json.stem),
        },
        "ga": dict(GA_DEFAULTS),
        "mutation": {
            "max_mutations": 1,
            "enable_numeric_perturbation": bool(enable_numeric),
            "enable_relop_flip": bool(enable_relop),
            "enable_logical_flip": bool(enable_logical),
            "enable_quantifier_flip": bool(enable_quant),
            "allowed_positions": allowed_positions,
            "allowed_changes": allowed_changes,
        }
    }

    if dry_run:
        return True, "DRY " + " ; ".join(notes)

    backup = exp_json.with_suffix(exp_json.suffix + ".v1")
    if not backup.exists():
        exp_json.replace(backup)
    exp_json.write_text(json.dumps(v2, indent=2))
    return True, "OK " + " ; ".join(notes)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report_lines = ["req,config,changed,note"]
    for req_dir in sorted([p for p in EFF.iterdir() if p.is_dir()]):
        for exp_json in sorted(req_dir.glob("exp*.json")):
            changed, note = migrate_file(req_dir, exp_json, args.dry_run)
            report_lines.append(f"{req_dir.name},{exp_json.name},{changed},{note.replace(',', ';')}")
    report = EFF / "migration_report.csv"
    report.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {report}")

if __name__ == "__main__":
    main()
