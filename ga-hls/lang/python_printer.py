from __future__ import annotations

from typing import List

from .ast import (
    And,
    ArithOp,
    BoolConst,
    Exists,
    ForAll,
    Formula,
    Implies,
    IntConst,
    Not,
    Or,
    RealConst,
    RelOp,
    Var,
    FuncCall,
    Subscript,
)


def formula_to_python_expr(f: Formula) -> str:
    """
    Render a Formula AST back into a Python expression string in the
    ThEodorE / property style, e.g.:

        Not(ForAll([t], Implies(interval_t, conditions_t)))
        And(x > 0, x < 10)
        ForAll([s], Implies(And(...), And(...)))
    """
    if isinstance(f, BoolConst):
        return "True" if f.value else "False"

    if isinstance(f, IntConst):
        return str(f.value)

    if isinstance(f, RealConst):
        return repr(f.value)

    if isinstance(f, Var):
        return f.name

    if isinstance(f, RelOp):
        left = formula_to_python_expr(f.left)
        right = formula_to_python_expr(f.right)
        return f"({left} {f.op} {right})"

    if isinstance(f, ArithOp):
        left = formula_to_python_expr(f.left)
        right = formula_to_python_expr(f.right)
        return f"({left} {f.op} {right})"

    if isinstance(f, Not):
        inner = formula_to_python_expr(f.arg)
        return f"Not({inner})"

    if isinstance(f, And):
        # Render as And(a, b, c, ...)
        args: List[str] = [formula_to_python_expr(a) for a in f.args]
        return "And(" + ", ".join(args) + ")"

    if isinstance(f, Or):
        args: List[str] = [formula_to_python_expr(a) for a in f.args]
        return "Or(" + ", ".join(args) + ")"

    if isinstance(f, Implies):
        left = formula_to_python_expr(f.left)
        right = formula_to_python_expr(f.right)
        return f"Implies({left}, {right})"

    if isinstance(f, ForAll):
        vars_list = "[" + ", ".join(v for v in f.vars) + "]"
        body = formula_to_python_expr(f.body)
        return f"ForAll({vars_list}, {body})"

    if isinstance(f, Exists):
        vars_list = "[" + ", ".join(v for v in f.vars) + "]"
        body = formula_to_python_expr(f.body)
        return f"Exists({vars_list}, {body})"
    
    if isinstance(f, FuncCall):
        args = ", ".join(formula_to_python_expr(a) for a in f.args)
        return f"{f.func}({args})"

    if isinstance(f, Subscript):
        return f"{formula_to_python_expr(f.base)}[{formula_to_python_expr(f.index)}]"

    # Fallback – shouldn't really happen if the AST is well-formed
    return repr(f)
