import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "replication" / "evaluation_inputs"

def rel(p: str) -> str:
    # Convert absolute/normalized paths to repo-relative (POSIX) when possible.
    path = Path(p)
    try:
        rp = path.resolve().relative_to(REPO.resolve())
        return rp.as_posix()
    except Exception:
        # If already relative, keep as-is.
        return p.replace("\\", "/")

def fix_one(cfg_path: Path) -> bool:
    data = json.loads(cfg_path.read_text())
    if not isinstance(data, dict) or "input" not in data:
        return False
    changed = False

    inp = data.get("input", {})
    for key in ("requirement_file", "traces_file", "output_dir"):
        if key in inp and isinstance(inp[key], str):
            newv = rel(inp[key])
            if newv != inp[key]:
                inp[key] = newv
                changed = True
    data["input"] = inp

    if changed:
        cfg_path.write_text(json.dumps(data, indent=2))
    return changed

def main():
    # Only touch v2 configs (exp*.json) under effectiveness + sensitivity
    cfgs = list((ROOT / "effectiveness").rglob("exp*.json")) + list((ROOT / "sensitivity").rglob("*.json"))
    cfgs = [p for p in cfgs if p.is_file() and not p.name.endswith(".v1")]
    touched = 0
    for p in cfgs:
        if fix_one(p):
            touched += 1
    print(f"Touched {touched} configs")

if __name__ == "__main__":
    main()
