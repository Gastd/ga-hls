from __future__ import annotations

from typing import Any, List

from .python_printer import formula_to_python_expr
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
    Subscript,
    FuncCall,
)


class InternalEncodeError(RuntimeError):
    """Raised when we cannot encode a Formula into the internal JSON format."""


def formula_to_internal_obj(formula: Formula) -> Any:
    """
    Encode a `Formula` AST node into the original internal list-of-lists format
    expected by `treenode.parse`.

    This is the inverse of `ga_hls.lang.internal_parser.parse_internal_obj` for
    the node types we support.
    """
    if isinstance(formula, BoolConst):
        return bool(formula.value)

    if isinstance(formula, IntConst):
        return int(formula.value)

    if isinstance(formula, RealConst):
        return float(formula.value)

    if isinstance(formula, Var):
        return formula.name

    if isinstance(formula, RelOp):
        return [
            formula.op,
            [
                formula_to_internal_obj(formula.left),
                formula_to_internal_obj(formula.right),
            ],
        ]

    if isinstance(formula, And):
        return [
            "And",
            [formula_to_internal_obj(a) for a in formula.args],
        ]

    if isinstance(formula, Or):
        return [
            "Or",
            [formula_to_internal_obj(a) for a in formula.args],
        ]

    if isinstance(formula, Not):
        return [
            "Not",
            [formula_to_internal_obj(formula.arg)],
        ]

    if isinstance(formula, Implies):
        return [
            "Implies",
            [
                formula_to_internal_obj(formula.left),
                formula_to_internal_obj(formula.right),
            ],
        ]

    if isinstance(formula, ForAll):
        # ["ForAll", [[vars...], body]]
        return [
            "ForAll",
            [
                list(formula.vars),
                formula_to_internal_obj(formula.body),
            ],
        ]

    if isinstance(formula, Exists):
        # ["Exists", [[vars...], body]]
        return [
            "Exists",
            [
                list(formula.vars),
                formula_to_internal_obj(formula.body),
            ],
        ]

    if isinstance(formula, ArithOp):
        return [
            formula.op,
            [
                formula_to_internal_obj(formula.left),
                formula_to_internal_obj(formula.right),
            ],
        ]
    
    if isinstance(formula, Subscript) or isinstance(formula, FuncCall):
        return formula_to_python_expr(formula)

    raise InternalEncodeError(
        f"Unsupported Formula node type for internal encoding: {type(formula)!r}"
    )
