from __future__ import annotations

import re
from pathlib import Path


_Z3_ADD_RE = re.compile(
    r"""(z3solver\.add\s*\()   # group 1: 'z3solver.add(' with optional spaces
        (.*)                   # group 2: the formula expression (greedy)
        (\)\s*)$               # group 3: closing ')' and optional trailing spaces
    """,
    re.VERBOSE,
)


def build_z3check_script(
    template_path: str | Path,
    formula_py_expr: str,
    output_path: str | Path,
) -> None:
    """
    Build a z3check.py file by taking the original property script as
    a template and replacing only the argument of z3solver.add(...).
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    lines = template_path.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    replaced = False

    for line in lines:
        if "z3solver.add" in line:
            m = _Z3_ADD_RE.search(line)
            if m:
                prefix, _, suffix = m.groups()
                newline = f"{prefix}{formula_py_expr}{suffix}"
                new_lines.append(newline)
                replaced = True
                continue

        new_lines.append(line)

    if not replaced:
        raise RuntimeError(
            f"Could not find a z3solver.add(...) line to rewrite in {template_path}"
        )

    output_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
