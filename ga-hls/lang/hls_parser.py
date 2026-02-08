from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional, Union

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


class HLSParseError(RuntimeError):
    """Raised when we cannot extract or interpret an HLS/ThEodorE .hls formula."""


class ExprParseError(RuntimeError):
    """Raised when the translated Python-like expression cannot be mapped to Formula AST."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_formula_from_hls(
    path: Union[str, Path],
    property_name: Optional[str] = None,
) -> Formula:
    """
    Load a ThEodorE/HLS `.hls` file and extract a property's Specification
    as a `Formula` AST node.

    Steps:
      - Parse the file textually.
      - Locate a `property_X::=` block (first one, or matching `property_name`).
      - Extract the `Specification::= ...;` chunk.
      - Translate its syntax to a Python-like expression string.
      - Parse that Python expression and map it to the internal Formula AST.

    Limitations:
      - Grammar support is intentionally narrow and follows the patterns
        present in the current requirements.hls file.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HLSParseError(f"Failed to read HLS file {path!s}: {exc}") from exc

    block = _extract_property_block(text, property_name)
    if block is None:
        if property_name:
            raise HLSParseError(
                f"Could not find property block '{property_name}' in {path!s}"
            )
        else:
            raise HLSParseError(f"Could not find any property block in {path!s}")

    spec_str = _extract_specification(block)
    if spec_str is None:
        raise HLSParseError(
            f"Could not find Specification::= ...; in property block in {path!s}"
        )

    expr_str = _hls_spec_to_python_expr(spec_str)

    try:
        py_ast = ast.parse(expr_str, mode="eval")
    except SyntaxError as exc:
        raise HLSParseError(
            f"Failed to parse translated HLS spec as Python:\n"
            f"  expr: {expr_str}\n  error: {exc}"
        ) from exc

    try:
        return _expr_to_formula(py_ast.body)
    except ExprParseError as exc:
        raise HLSParseError(
            f"Failed to map translated HLS spec to internal AST:\n"
            f"  expr: {expr_str}\n  error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Property block & Specification extraction
# ---------------------------------------------------------------------------


_PROPERTY_RE = re.compile(
    r"property_(?P<num>\w+)\s*::=\s*\{(?P<body>.*?)\}",
    re.DOTALL | re.IGNORECASE,
)


def _extract_property_block(
    text: str, property_name: Optional[str]
) -> Optional[str]:
    """
    Extract the contents of a property block:

        property_01::=
        {
            ...
        }

    If `property_name` is given (e.g., "property_01"), we try to match it;
    otherwise we return the first property block.
    """
    matches = list(_PROPERTY_RE.finditer(text))
    if not matches:
        return None

    if property_name is None:
        return matches[0].group("body")

    # Normalize property_name like "property_01" or just "01"
    prop_suffix = property_name
    if property_name.lower().startswith("property_"):
        prop_suffix = property_name.split("_", 1)[1]

    for m in matches:
        if m.group("num") == prop_suffix:
            return m.group("body")

    return None


_SPEC_RE = re.compile(
    r"Specification\s*::=\s*(?P<spec>.*?);",
    re.DOTALL | re.IGNORECASE,
)


def _extract_specification(block: str) -> Optional[str]:
    m = _SPEC_RE.search(block)
    if not m:
        return None
    spec = m.group("spec").strip()
    return spec


# ---------------------------------------------------------------------------
# HLS spec string ---> Python expr string
# ---------------------------------------------------------------------------

def _hls_spec_to_python_expr(spec: str) -> str:
    """
    Translate a single Specification::= ...; string from HLS syntax into a
    Python-like expression which `_expr_to_formula` can understand.

    Patterns supported:

      Exists Value c In (a,b): ForAll Timestamp t In (c,d): BODY
      ForAll Index s In (a,b): BODY
      ForAll Timestamp t In (a,b): BODY
    """
    spec = spec.strip()
    # Collapse newlines / excessive whitespace
    spec = " ".join(spec.split())

    # 1) Exists Value c In (...) : ForAll Timestamp t In (...) : BODY
    m = re.match(
        r"Exists\s+Value\s+(?P<val>\w+)\s+In\s*(?P<vrange>[\(\[].*?[\)\]])\s*:\s*"
        r"ForAll\s+Timestamp\s+(?P<tvar>\w+)\s+In\s*(?P<trange>[\(\[].*?[\)\]])\s*:\s*(?P<body>.+)",
        spec,
        flags=re.IGNORECASE,
    )
    if m:
        val = m.group("val")
        vrange_str = m.group("vrange")
        tvar = m.group("tvar")
        trange_str = m.group("trange")
        body_str = m.group("body")

        v_lower, v_upper, v_inclusive = _parse_range(vrange_str)
        t_lower, t_upper, t_inclusive = _parse_range(trange_str)

        # Domain for c
        if v_inclusive == "open":
            c_range_cond = f"({val} > {v_lower}) and ({val} < {v_upper})"
        elif v_inclusive == "closed":
            c_range_cond = f"({val} >= {v_lower}) and ({val} <= {v_upper})"
        else:
            c_range_cond = f"({val} >= {v_lower}) and ({val} < {v_upper})"

        # Domain for t
        if t_inclusive == "open":
            t_range_cond = f"({tvar} > {t_lower}) and ({tvar} < {t_upper})"
        elif t_inclusive == "closed":
            t_range_cond = f"({tvar} >= {t_lower}) and ({tvar} <= {t_upper})"
        else:
            t_range_cond = f"({tvar} >= {t_lower}) and ({tvar} < {t_upper})"

        body_expr = _hls_body_to_python(body_str)

        inner_forall = (
            f"ForAll([{tvar}], Implies(({t_range_cond}), ({body_expr})))"
        )
        # For Exists, we encode the domain as an `and` with the ForAll
        expr = f"Exists([{val}], ({c_range_cond}) and ({inner_forall}))"
        return expr

    # 2) ForAll Index s In (...) : BODY
    m = re.match(
        r"ForAll\s+Index\s+(?P<var>\w+)\s+In\s*(?P<range>[\(\[].*?[\)\]])\s*:\s*(?P<body>.+)",
        spec,
        flags=re.IGNORECASE,
    )
    if m:
        var = m.group("var")
        range_str = m.group("range")
        body_str = m.group("body")

        lower, upper, inclusive = _parse_range(range_str)

        if inclusive == "open":  # (a, b)
            range_cond = f"({var} > {lower}) and ({var} < {upper})"
        elif inclusive == "closed":  # [a, b]
            range_cond = f"({var} >= {lower}) and ({var} <= {upper})"
        else:  # half-open, approximate
            range_cond = f"({var} >= {lower}) and ({var} < {upper})"

        body_expr = _hls_body_to_python(body_str)
        expr = f"ForAll([{var}], Implies(({range_cond}), ({body_expr})))"
        return expr

    # 3) ForAll Timestamp t In (...) : BODY (if it ever appears alone)
    m = re.match(
        r"ForAll\s+Timestamp\s+(?P<var>\w+)\s+In\s*(?P<range>[\(\[].*?[\)\]])\s*:\s*(?P<body>.+)",
        spec,
        flags=re.IGNORECASE,
    )
    if m:
        var = m.group("var")
        range_str = m.group("range")
        body_str = m.group("body")

        lower, upper, inclusive = _parse_range(range_str)

        if inclusive == "open":
            range_cond = f"({var} > {lower}) and ({var} < {upper})"
        elif inclusive == "closed":
            range_cond = f"({var} >= {lower}) and ({var} <= {upper})"
        else:
            range_cond = f"({var} >= {lower}) and ({var} < {upper})"

        body_expr = _hls_body_to_python(body_str)
        expr = f"ForAll([{var}], Implies(({range_cond}), ({body_expr})))"
        return expr

    # 4) Fallback: no quantifier pattern recognized, just translate body
    body_expr = _hls_body_to_python(spec)
    return body_expr

_RANGE_RE = re.compile(r"([\(\[])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\)\]])")

def _parse_range(range_str: str):
    """
    Parse a range like (0,FinalIndex) or [0, FinalIndex-1] or (0, 10 [h]).

    Returns (lower_str, upper_str, inclusive_type)
      inclusive_type ∈ {"open", "closed", "half-open"}
    """
    # First, drop any unit annotations like [h], [s], etc.
    clean = re.sub(r"\[[^\]]*\]", "", range_str)

    m = _RANGE_RE.match(clean)
    if not m:
        # Fallback: treat as opaque expressions
        return "0", "FinalIndex", "open"

    left_br, lower, upper, right_br = m.groups()
    lower = lower.strip()
    upper = upper.strip()

    if left_br == "(" and right_br == ")":
        inc = "open"
    elif left_br == "[" and right_br == "]":
        inc = "closed"
    else:
        inc = "half-open"

    return lower, upper, inc

def _hls_body_to_python(body: str) -> str:
    """
    Translate the body part of an HLS spec into a Python-like expression.

    Transformations:
      - HLS logical operators: And/Or → `and` / `or`
      - Implication: A->B or ((A->B) And (C->D)) → Implies(...)
      - signal_4(@index s) → signal_4[s]
      - signal_4(@index s+1) → signal_4[s+1]
      - signal_8(@timestamp 36000[s]) → signal_8[36000] or signal_8[t]
    """
    s = body

    # Normalize whitespace for easier regex handling
    s = " ".join(s.split())

    # Handle the known implication patterns first
    s = _replace_implies_patterns(s)

    # Signals at index: signal_4(@index s) → signal_4[s]
    s = re.sub(
        r"(\w+)\(@index\s+([^\)]+)\)",
        r"\1[\2]",
        s,
    )

    # Signals at timestamp: handle both numeric + units and variables
    def _ts_repl(m: re.Match) -> str:
        name = m.group(1)
        inner = m.group(2)
        inner = re.sub(r"\[[^\]]*\]", "", inner).strip()
        return f"{name}[{inner}]"

    s = re.sub(
        r"(\w+)\(@timestamp\s+([^\)]+)\)",
        _ts_repl,
        s,
    )

    # Logical keywords
    s = re.sub(r"\bAnd\b", " and ", s)
    s = re.sub(r"\bOr\b", " or ", s)

    return s


def _replace_implies_patterns(s: str) -> str:
    """
    Replace HLS-style implications with Implies(...) in the limited patterns
    present in requirements.hls.

    Supported:
      - ((A->B ) And (C->D))
      - A->B  (single arrow in the expression)
    """
    # 1) property_04-style: ((A->B ) And (C->D))
    m = re.match(
        r"\(\((?P<a>.+?)->(?P<b>.+?)\)\s+And\s+\((?P<c>.+?)->(?P<d>.+?)\)\)",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        a = m.group("a").strip()
        b = m.group("b").strip()
        c = m.group("c").strip()
        d = m.group("d").strip()
        return (
            f"(Implies(({a}), ({b})) and "
            f"Implies(({c}), ({d})))"
        )

    # 2) Generic single-implication case (e.g. property_06)
    if "->" in s:
        left, right = s.split("->", 1)
        left = left.strip()
        right = right.strip()
        return f"Implies(({left}), ({right}))"

    # 3) No implication
    return s

# ---------------------------------------------------------------------------
# Python expr -> Formula
# ---------------------------------------------------------------------------


def _expr_to_formula(node: ast.expr) -> Formula:
    # Constants
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return BoolConst(v)
        if isinstance(v, int):
            return IntConst(v)
        if isinstance(v, float):
            return RealConst(v)
        # Strings etc. – treat as variable-like
        return Var(repr(v))

    # Variables / identifiers
    if isinstance(node, ast.Name):
        return Var(node.id)

    # Subscript, e.g., signal_5[s]
    if isinstance(node, ast.Subscript):
        return Var(_subscript_to_str(node))

    # Unary operators: Not, unary minus
    if isinstance(node, ast.UnaryOp):
        # Logical negation
        if isinstance(node.op, ast.Not):
            return Not(_expr_to_formula(node.operand))

        # Unary minus, e.g. -15.27 or -x
        if isinstance(node.op, ast.USub):
            inner = _expr_to_formula(node.operand)
            # If the inner is a numeric constant, just flip the sign
            if isinstance(inner, IntConst):
                return IntConst(-inner.value)
            if isinstance(inner, RealConst):
                return RealConst(-inner.value)
            # Otherwise, encode as 0 - inner
            return ArithOp(op="-", left=IntConst(0), right=inner)

        raise ExprParseError(f"Unsupported unary operator: {ast.dump(node.op)}")

    # Boolean operators: And, Or
    if isinstance(node, ast.BoolOp):
        args = [_expr_to_formula(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return And(args)
        if isinstance(node.op, ast.Or):
            return Or(args)
        raise ExprParseError(f"Unsupported boolean op: {ast.dump(node.op)}")

    # Binary arithmetic operators
    if isinstance(node, ast.BinOp):
        left = _expr_to_formula(node.left)
        right = _expr_to_formula(node.right)
        op_str = _binop_to_str(node.op)
        return ArithOp(op=op_str, left=left, right=right)

    # Comparisons: <, <=, >, >=, ==, !=
    if isinstance(node, ast.Compare):
        left_expr = node.left
        ops = node.ops
        comparators = node.comparators

        if len(ops) != len(comparators):
            raise ExprParseError(
                f"Comparison ops/comparators length mismatch: {ast.dump(node)}"
            )

        rels: List[Formula] = []
        cur_left = left_expr
        for op, right_expr in zip(ops, comparators):
            rel = RelOp(
                op=_cmpop_to_str(op),
                left=_expr_to_formula(cur_left),
                right=_expr_to_formula(right_expr),
            )
            rels.append(rel)
            cur_left = right_expr

        if len(rels) == 1:
            return rels[0]
        return And(rels)

    # Calls: ForAll([...], body), Exists([...], body), Implies(a, b)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in ("ForAll", "Exists"):
                return _handle_quantifier_call(fname, node)
            if fname == "Implies":
                if len(node.args) != 2:
                    raise ExprParseError(
                        f"Implies expects two arguments, got: {ast.dump(node)}"
                    )
                left = _expr_to_formula(node.args[0])
                right = _expr_to_formula(node.args[1])
                return Implies(left=left, right=right)
            # Other function calls → treat as opaque Var
            return Var(_call_to_str(node))

        # Something like f.g(...)
        return Var(_call_to_str(node))

    raise ExprParseError(f"Unsupported expression node: {ast.dump(node)}")


def _handle_quantifier_call(fname: str, call: ast.Call) -> Formula:
    if len(call.args) != 2:
        raise ExprParseError(
            f"{fname} expects two arguments (vars list, body), got: {ast.dump(call)}"
        )

    vars_arg, body_arg = call.args

    if not isinstance(vars_arg, (ast.List, ast.Tuple)):
        raise ExprParseError(
            f"{fname} first argument must be a list/tuple of variable names, got: {ast.dump(vars_arg)}"
        )

    var_names: List[str] = []
    for elt in vars_arg.elts:
        if isinstance(elt, ast.Name):
            var_names.append(elt.id)
        else:
            raise ExprParseError(
                f"{fname} variable list elements must be simple names, got: {ast.dump(elt)}"
            )

    body_formula = _expr_to_formula(body_arg)

    if fname == "ForAll":
        return ForAll(vars=var_names, body=body_formula)
    else:
        return Exists(vars=var_names, body=body_formula)


def _subscript_to_str(node: ast.Subscript) -> str:
    """
    Render a subscript expression like `signal_5[s]` into a string.
    """
    base = _expr_to_formula(node.value)
    slice_node = node.slice
    if isinstance(slice_node, ast.Index):  # Python 3.8
        slice_node = slice_node.value
    index = _expr_to_formula(slice_node)
    return f"{base}[{index}]"


def _call_to_str(node: ast.Call) -> str:
    func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else _fallback_func_str(
        node.func
    )
    args_str = ", ".join(
        ast.unparse(a) if hasattr(ast, "unparse") else _fallback_expr_str(a)
        for a in node.args
    )
    return f"{func_str}({args_str})"


def _fallback_func_str(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    return node.__class__.__name__


def _fallback_expr_str(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return node.__class__.__name__


def _binop_to_str(op: ast.operator) -> str:
    if isinstance(op, ast.Add):
        return "+"
    if isinstance(op, ast.Sub):
        return "-"
    if isinstance(op, ast.Mult):
        return "*"
    if isinstance(op, ast.Div):
        return "/"
    raise ExprParseError(f"Unsupported binary operator: {op!r}")


def _cmpop_to_str(op: ast.cmpop) -> str:
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    raise ExprParseError(f"Unsupported comparison operator: {op!r}")
