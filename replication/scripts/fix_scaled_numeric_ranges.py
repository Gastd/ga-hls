import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO = Path(__file__).resolve().parents[2]
EFF = REPO / "replication" / "evaluation_inputs" / "effectiveness"

DOCKER = ["docker", "compose", "-f", "docker/docker-compose.yml", "run", "--rm", "ga-hls"]

REL_SYMS = ["<=", ">=", "<", ">", "==", "!="]

def have_local_cli() -> bool:
    return shutil.which("ga-hls") is not None

def run_ga_hls(args: List[str]) -> str:
    if have_local_cli():
        cmd = ["ga-hls"] + args
        return subprocess.check_output(cmd, cwd=str(REPO), text=True)
    cmd = DOCKER + args
    return subprocess.check_output(cmd, cwd=str(REPO), text=True)

def parse_positions(out: str) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str,str]] = {}
    for ln in out.splitlines():
        s = ln.rstrip()
        if not s.strip():
            continue
        m = re.match(r"^\s*(\d+)\s+(\w+)\s+(.*?)\s+(\S+)\s*$", s)
        if not m:
            continue
        pos = int(m.group(1))
        nodetype = m.group(2)
        node = m.group(3).strip()
        feat = m.group(4).strip()
        rows[pos] = {"nodetype": nodetype, "node": node, "feature": feat}
    return rows

def parse_num(node_text: str) -> Optional[float]:
    t = node_text.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return float(t)
    m = re.search(r"(-?\d+(?:\.\d+)?)", t)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def best_match_numeric(cands: List[Tuple[int, float]], lo: float, hi: float, avoid: set) -> Optional[int]:
    mid = (lo + hi) / 2.0
    usable = [(p,v) for (p,v) in cands if p not in avoid]
    if not usable:
        usable = cands
    in_range = [(p,v) for (p,v) in usable if lo <= v <= hi]
    pool = in_range if in_range else usable
    pool.sort(key=lambda pv: (abs(pv[1]-mid), pv[0]))
    return pool[0][0] if pool else None

def is_microseconds_range(lo: float, hi: float) -> bool:
    # Heuristic: big numbers, divisible by 1e6-ish
    if lo < 1_000_000 and hi < 1_000_000:
        return False
    return (abs(lo) % 1_000_000 == 0) and (abs(hi) % 1_000_000 == 0)

def fix_config(cfg_path: Path) -> Tuple[bool, List[str]]:
    d = json.loads(cfg_path.read_text())
    if not isinstance(d, dict) or "input" not in d or "mutation" not in d:
        return False, ["not_a_v2_config"]

    mut = d["mutation"]
    allowed_changes: Dict[str, Dict] = mut.get("allowed_changes", {})
    allowed_positions: List[int] = mut.get("allowed_positions", [])

    # get AST positions
    try:
        out = run_ga_hls(["explain-positions", "--config", str(cfg_path)])
    except Exception as e:
        return False, [f"explain_failed:{e!r}"]
    rows = parse_positions(out)

    # numeric candidates
    numeric_nodes: List[Tuple[int, float]] = []
    for pos, r in rows.items():
        if r["nodetype"] in ("IntConst", "RealConst"):
            val = parse_num(r["node"])
            if val is not None:
                numeric_nodes.append((pos, val))

    if not numeric_nodes:
        return False, ["no_numeric_nodes"]

    changed = False
    notes: List[str] = []
    used = set()

    # Pre-mark all positions used by numeric changes (so we can avoid duplicates when remapping)
    for k, v in allowed_changes.items():
        if "numeric" in v:
            used.add(int(k))

    for k, v in list(allowed_changes.items()):
        if "numeric" not in v:
            continue
        pos = int(k)
        lo, hi = float(v["numeric"][0]), float(v["numeric"][1])

        # sanity: does current node value satisfy range?
        node_val = None
        if pos in rows:
            node_val = parse_num(rows[pos]["node"])

        if node_val is not None and (lo <= node_val <= hi):
            continue  # ok

        # Try to fix: if microseconds domain but AST has seconds constants
        if is_microseconds_range(lo, hi):
            lo2, hi2 = lo / 1_000_000.0, hi / 1_000_000.0
            # Look for any numeric node inside rescaled range
            in_range = [(p,val) for (p,val) in numeric_nodes if lo2 <= val <= hi2]
            if in_range:
                new_pos = best_match_numeric(numeric_nodes, lo2, hi2, avoid=set())
                if new_pos is None:
                    continue

                # update keys in allowed_changes and allowed_positions
                # remove old mapping
                old_key = str(pos)
                new_key = str(new_pos)

                # if new_pos already has a numeric mapping, we'll overwrite only if it's the same domain,
                # otherwise keep the first and mark conflict.
                if new_key in allowed_changes and "numeric" in allowed_changes[new_key] and new_key != old_key:
                    # conflict; skip to avoid silently clobbering
                    notes.append(f"conflict_skip:{cfg_path.name}:pos{pos}->pos{new_pos}")
                    continue

                # rewrite
                del allowed_changes[old_key]
                allowed_changes[new_key] = {"numeric": [lo2, hi2]}
                changed = True
                notes.append(f"rescaled:{cfg_path.name}:{pos}->{new_pos} [{lo},{hi}]=>[{lo2},{hi2}]")
                # update allowed_positions list
                allowed_positions = [p for p in allowed_positions if p != pos]
                if new_pos not in allowed_positions:
                    allowed_positions.append(new_pos)
                continue

        # Otherwise: try remapping without rescaling (find node within original range)
        in_range = [(p,val) for (p,val) in numeric_nodes if lo <= val <= hi]
        if in_range:
            new_pos = best_match_numeric(numeric_nodes, lo, hi, avoid=set())
            if new_pos is None:
                continue
            old_key = str(pos)
            new_key = str(new_pos)
            if new_key in allowed_changes and "numeric" in allowed_changes[new_key] and new_key != old_key:
                notes.append(f"conflict_skip:{cfg_path.name}:pos{pos}->pos{new_pos}")
                continue
            del allowed_changes[old_key]
            allowed_changes[new_key] = {"numeric": [lo, hi]}
            changed = True
            notes.append(f"remap:{cfg_path.name}:{pos}->{new_pos} [{lo},{hi}]")
            allowed_positions = [p for p in allowed_positions if p != pos]
            if new_pos not in allowed_positions:
                allowed_positions.append(new_pos)
            continue

        # If nothing found, leave it for manual inspection
        notes.append(f"unfixed:{cfg_path.name}:pos{pos} val={node_val} range=[{lo},{hi}]")

    if changed:
        mut["allowed_changes"] = {k: allowed_changes[k] for k in sorted(allowed_changes.keys(), key=lambda x: int(x))}
        mut["allowed_positions"] = sorted(set(allowed_positions))
        d["mutation"] = mut
        cfg_path.write_text(json.dumps(d, indent=2))

    return changed, notes

def main():
    cfgs = []
    for req_dir in sorted([p for p in EFF.iterdir() if p.is_dir()]):
        cfgs.extend(sorted(req_dir.glob("exp*.json")))
    total = 0
    changed_n = 0
    all_notes: List[str] = []
    for cfg in cfgs:
        if cfg.name.endswith(".v1"):
            continue
        total += 1
        changed, notes = fix_config(cfg)
        if changed:
            changed_n += 1
        all_notes.extend(notes)
    print(f"Processed {total} configs, changed {changed_n}")
    if all_notes:
        print("Notes:")
        for n in all_notes:
            print(" -", n)

if __name__ == "__main__":
    main()
