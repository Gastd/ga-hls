from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Sequence

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
)


class InternalFormatError(RuntimeError):
    """Raised when the internal JSON-style formula format is malformed."""


# --- Public API -------------------------------------------------------------


def parse_internal_json(s: str) -> Formula:
    """
    Parse an internal JSON string (list-of-lists, operator-first) into an AST.

    Example input string (simplified):
        '["ForAll", [["s"], ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18]]]], ["==", ["signal_5[s]",0]]]]]]]'
    """
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        raise InternalFormatError(f"Invalid JSON formula: {exc}") from exc

    return parse_internal_obj(data)


def parse_internal_obj(obj: Any) -> Formula:
    """
    Parse a Python object representing the internal formula format
    (as obtained from json.loads) into an AST Formula.
    """
    # Base cases: numbers, booleans, strings.
    if isinstance(obj, bool):
        return BoolConst(obj)

    if isinstance(obj, int):
        return IntConst(obj)

    if isinstance(obj, float):
        return RealConst(obj)

    if isinstance(obj, str):
        return Var(obj)

    if not isinstance(obj, list) or not obj:
        raise InternalFormatError(f"Expected non-empty list or atom, got {obj!r}")

    # We expect an operator-first encoding: [op, args]
    op = obj[0]
    if not isinstance(op, str):
        raise InternalFormatError(f"Operator must be a string, got {op!r}")

    if len(obj) != 2:
        raise InternalFormatError(
            f"Operator node must have exactly 2 elements [op, args], got {obj!r}"
        )

    args = obj[1]

    # Quantifiers: ["ForAll", [[vars...], body]] or ["Exists", [[vars...], body]]
    if op in ("ForAll", "Exists"):
        if not isinstance(args, list) or len(args) != 2:
            raise InternalFormatError(
                f"{op} expects [ [vars...], body ], got {args!r}"
            )

        vars_part, body_part = args
        if not isinstance(vars_part, list):
            raise InternalFormatError(
                f"{op} vars part must be a list of variable names, got {vars_part!r}"
            )

        var_names: List[str] = []
        for v in vars_part:
            if not isinstance(v, str):
                raise InternalFormatError(
                    f"{op} variable names must be strings, got {v!r}"
                )
            var_names.append(v)

        body = parse_internal_obj(body_part)

        if op == "ForAll":
            return ForAll(vars=var_names, body=body)
        else:
            return Exists(vars=var_names, body=body)

    # Logical connectives
    if op == "And":
        _ensure_seq(args, op)
        return And(args=[parse_internal_obj(a) for a in args])

    if op == "Or":
        _ensure_seq(args, op)
        return Or(args=[parse_internal_obj(a) for a in args])

    if op == "Not":
        _ensure_seq(args, op, expected_len=1)
        return Not(arg=parse_internal_obj(args[0]))

    if op == "Implies":
        _ensure_seq(args, op, expected_len=2)
        left = parse_internal_obj(args[0])
        right = parse_internal_obj(args[1])
        return Implies(left=left, right=right)

    # Relational operators
    if op in ("<", "<=", ">", ">=", "==", "!="):
        _ensure_seq(args, op, expected_len=2)
        left = parse_internal_obj(args[0])
        right = parse_internal_obj(args[1])
        return RelOp(op=op, left=left, right=right)

    # Arithmetic operators (if present)
    if op in ("+", "-", "*", "/"):
        _ensure_seq(args, op, expected_len=2)
        left = parse_internal_obj(args[0])
        right = parse_internal_obj(args[1])
        return ArithOp(op=op, left=left, right=right)

    # Fallback: treat unknown op as function-like application encoded as Var(op)(args...)
    _ensure_seq(args, op)
    # Simple textual encoding:
    inner = ", ".join(str(parse_internal_obj(a)) for a in args)
    return Var(f"{op}({inner})")


# --- Helpers ----------------------------------------------------------------


def _ensure_seq(args: Any, op: str, expected_len: int | None = None) -> None:
    if not isinstance(args, Sequence):
        raise InternalFormatError(f"{op} expects a sequence of arguments, got {args!r}")
    if expected_len is not None and len(args) != expected_len:
        raise InternalFormatError(
            f"{op} expects {expected_len} arguments, got {len(args)} in {args!r}"
        )
