import re
from pathlib import Path

ROOT = Path("replication/evaluation_inputs/effectiveness")

# Matches: ToInt(RealVal(0)+(((<num>*1000000)-0.0)/10000.0))
# Captures <num> which can be int or float (e.g., 30, 2.5)
TOINT_TIMEIDX_RE = re.compile(
    r"ToInt\(\s*RealVal\(0\)\s*\+\s*\(\(\(\s*([0-9]+(?:\.[0-9]+)?)\s*\*\s*1000000\s*\)\s*-\s*0\.0\s*\)\s*/\s*10000\.0\s*\)\s*\)\s*\)"
)

# Matches: (<int>*1000000) e.g., (10*1000000) -> 10000000
INT_US_RE = re.compile(r"\(\s*([0-9]+)\s*\*\s*1000000\s*\)")

def repl_toint(m: re.Match) -> str:
    s = float(m.group(1))
    # (s*1e6)/10000 = s*100
    v = s * 100.0
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return str(v)

def normalize_text(txt: str) -> str:
    txt = TOINT_TIMEIDX_RE.sub(lambda m: repl_toint(m), txt)
    txt = INT_US_RE.sub(lambda m: str(int(m.group(1)) * 1_000_000), txt)
    return txt

def main():
    py_files = sorted(ROOT.rglob("*.py"))
    if not py_files:
        print("No property .py files found under", ROOT)
        return

    changed = 0
    for p in py_files:
        out = p.with_name(p.stem + "_norm.py")
        src = p.read_text(encoding="utf-8")
        norm = normalize_text(src)
        if norm != src:
            out.write_text(norm, encoding="utf-8")
            changed += 1
            print("WROTE", out)
        else:
            # still write a copy? no—avoid noise
            pass

    print(f"Done. Normalized {changed} files (created *_norm.py).")

if __name__ == "__main__":
    main()
