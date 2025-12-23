from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

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


class TheodoreParseError(RuntimeError):
    """Raised when we cannot extract or interpret a ThEodorE/HLS formula."""

class TheodorePropertyExtractor(ast.NodeVisitor):
    """
    Walks a ThEodorE property file and extracts:
      - definitions of alias variables (interval_t, conditions_t, ...)
      - the z3solver.add(...) requirement formula.
    """
    def __init__(self) -> None:
        self.alias_defs: dict[str, ast.AST] = {}
        self.requirement_expr: ast.AST | None = None

    # capture assignments like: interval = <expr>, in_zone = <expr>, etc.
    def visit_Assign(self, node: ast.Assign) -> None:
        # only handle simple "NAME = expr"
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            # Treat all simple assignments as potential aliases.
            # They will only be inlined where Var(name) appears in the main formula.
            self.alias_defs[name] = node.value
        self.generic_visit(node)


    # capture: z3solver.add(<expr>)
    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "z3solver"
            and node.func.attr == "add"
            and len(node.args) == 1
        ):
            # Remember the *last* add(...) call as the requirement
            self.requirement_expr = node.args[0]
        self.generic_visit(node)

# --- Public API -------------------------------------------------------------

def load_formula_from_property(path: Union[str, Path]) -> Formula:
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TheodoreParseError(f"Failed to read property file {path!s}: {exc}") from exc

    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise TheodoreParseError(f"Failed to parse Python in {path!s}: {exc}") from exc

    # First pass: collect alias definitions and the explicit requirement expr
    extractor = TheodorePropertyExtractor()
    extractor.visit(module)

    requirement_expr: ast.AST | None = extractor.requirement_expr

    # Fallback to the old heuristic if we didn't see a z3solver.add(...) in the visitor
    if requirement_expr is None:
        call_node = _find_z3_add_call(module)
        if call_node is None:
            raise TheodoreParseError(
                f"Could not find a 'z3solver.add(...)' call in {path!s}"
            )
        if not call_node.args:
            raise TheodoreParseError(
                f"'z3solver.add' in {path!s} has no arguments; expected a formula."
            )
        requirement_expr = call_node.args[0]

    # Parse the main requirement expression into a Formula
    requirement_formula = _expr_to_formula(requirement_expr)

    # Parse alias definitions (interval_t, conditions_t, etc.) into Formulas
    alias_formulas: dict[str, Formula] = {}
    for name, expr in extractor.alias_defs.items():
        try:
            alias_formulas[name] = _expr_to_formula(expr)
        except TheodoreParseError as exc:
            raise TheodoreParseError(
                f"Failed to parse alias definition '{name}' in {path!s}: {exc}"
            ) from exc

    # Inline aliases so that Var('interval_t') / Var('conditions_t') are expanded
    if alias_formulas:
        requirement_formula = _inline_aliases(requirement_formula, alias_formulas)

    # Strips out the outer Not:
    #   Not( ForAll(...) )  -->  ForAll(...)
    if isinstance(requirement_formula, Not):
        requirement_formula = requirement_formula.arg

    return requirement_formula



# --- AST search helpers -----------------------------------------------------


def _find_z3_add_call(module: ast.Module) -> ast.Call | None:
    """
    Search the module for a Call node matching `z3solver.add(...)`.

    We may have multiple such calls (timestamps constraints, trace setup,
    main requirement, etc.). We pick the "best" one by the following
    heuristic:

      1. Collect all z3solver.add(...) calls.
      2. Prefer those whose first argument contains a ForAll(...) or Exists(...).
      3. Among those, pick the one with the largest AST size.
      4. If none have quantifiers, pick the one with the largest AST size overall.
    """
    candidates: List[ast.Call] = []

    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "z3solver"
                    and func.attr == "add"
                ):
                    if node.args:
                        candidates.append(node)

    if not candidates:
        return None

    def contains_quantifier(expr: ast.AST) -> bool:
        for sub in ast.walk(expr):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in ("ForAll", "Exists")
            ):
                return True
        return False

    def expr_size(expr: ast.AST) -> int:
        return sum(1 for _ in ast.walk(expr))

    # First, filter to those that contain quantifiers in their first argument.
    quant_candidates = [
        call for call in candidates if contains_quantifier(call.args[0])
    ]

    if quant_candidates:
        # Pick the largest one
        return max(quant_candidates, key=lambda c: expr_size(c.args[0]))

    # Fallback: pick the call with the largest argument AST
    return max(candidates, key=lambda c: expr_size(c.args[0]))


def _inline_aliases(formula: Formula, aliases: dict[str, Formula]) -> Formula:
    """
    Recursively replace Var(name) with aliases[name] wherever possible.

    This returns a *new* Formula tree; the alias definitions themselves are
    not mutated.
    """

    # Direct substitution for alias variables
    if isinstance(formula, Var) and formula.name in aliases:
        # Recurse once more in case aliases refer to other aliases
        return _inline_aliases(aliases[formula.name], aliases)

    # Structural recursion on all known node types
    if isinstance(formula, Not):
        return Not(_inline_aliases(formula.arg, aliases))

    if isinstance(formula, And):
        return And([_inline_aliases(a, aliases) for a in formula.args])

    if isinstance(formula, Or):
        return Or([_inline_aliases(a, aliases) for a in formula.args])

    if isinstance(formula, Implies):
        return Implies(
            left=_inline_aliases(formula.left, aliases),
            right=_inline_aliases(formula.right, aliases),
        )

    if isinstance(formula, ForAll):
        return ForAll(
            vars=list(formula.vars),
            body=_inline_aliases(formula.body, aliases),
        )

    if isinstance(formula, Exists):
        return Exists(
            vars=list(formula.vars),
            body=_inline_aliases(formula.body, aliases),
        )

    if isinstance(formula, RelOp):
        return RelOp(
            op=formula.op,
            left=_inline_aliases(formula.left, aliases),
            right=_inline_aliases(formula.right, aliases),
        )

    if isinstance(formula, ArithOp):
        return ArithOp(
            op=formula.op,
            left=_inline_aliases(formula.left, aliases),
            right=_inline_aliases(formula.right, aliases),
        )

    # Base cases: constants, already-opaque vars/signals, etc.
    # BoolConst, IntConst, RealConst, Var (non-alias), or any other Formula subtype
    # that doesn't contain children we care about.
    return formula

# --- Python expr -> Formula -------------------------------------------------


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

    # Subscript, e.g., v_speed[ToInt(...)]
    if isinstance(node, ast.Subscript):
        # Base: v_speed
        base_formula = _expr_to_formula(node.value)

        # Slice (index): ToInt(RealVal(0) + (t - 0.0) / 10000.0)
        slice_node = node.slice
        # Python < 3.9 uses ast.Index wrapper
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):  # type: ignore[attr-defined]
            slice_node = slice_node.value

        index_formula = _expr_to_formula(slice_node)
        return Subscript(base=base_formula, index=index_formula)


    # Unary operators (we only care about Not)
        # Unary operators: Not, unary +/- on numeric literals (e.g., -3.0)
    if isinstance(node, ast.UnaryOp):
        # Logical negation
        if isinstance(node.op, ast.Not):
            return Not(_expr_to_formula(node.operand))

        # Unary plus: +x  (just return x)
        if isinstance(node.op, ast.UAdd):
            return _expr_to_formula(node.operand)

        # Unary minus: -x
        if isinstance(node.op, ast.USub):
            operand = node.operand

            # Common case: negative numeric literal, e.g., -3 or -3.0
            if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                v = operand.value
                if isinstance(v, int):
                    return IntConst(-v)
                else:
                    return RealConst(-float(v))

            # Fallback: if operand parses to a numeric const, negate it
            inner = _expr_to_formula(operand)
            if isinstance(inner, IntConst):
                return IntConst(-inner.value)
            if isinstance(inner, RealConst):
                return RealConst(-inner.value)

            raise TheodoreParseError(
                f"Unary '-' is only supported on numeric literals/consts, got: {ast.dump(node)}"
            )

        raise TheodoreParseError(f"Unsupported unary operator: {ast.dump(node.op)}")

    # Boolean operators: And, Or
    if isinstance(node, ast.BoolOp):
        args = [_expr_to_formula(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return And(args)
        if isinstance(node.op, ast.Or):
            return Or(args)
        raise TheodoreParseError(f"Unsupported boolean op: {ast.dump(node.op)}")

    # Binary arithmetic operators
    if isinstance(node, ast.BinOp):
        left = _expr_to_formula(node.left)
        right = _expr_to_formula(node.right)
        op_str = _binop_to_str(node.op)
        return ArithOp(op=op_str, left=left, right=right)

    # Comparisons: <, <=, >, >=, ==, !=
    if isinstance(node, ast.Compare):
        # Python allows chained comparisons (a < b < c).
        # We translate `a < b < c` into (a < b) ∧ (b < c).
        left_expr = node.left
        ops = node.ops
        comparators = node.comparators

        if len(ops) != len(comparators):
            raise TheodoreParseError(
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

    # Calls: ForAll([...], body), Exists([...], body), Not(...), And(...),
    # Or(...), Implies(...), ToInt(...), RealVal(...), etc.
    if isinstance(node, ast.Call):
        # Simple function name, e.g. ForAll, Exists, ToInt, RealVal
        if isinstance(node.func, ast.Name):
            fname = node.func.id

            # Quantifiers
            if fname in ("ForAll", "Exists"):
                return _handle_quantifier_call(fname, node)

            # Logical Not as a function: Not(phi)
            if fname == "Not":
                if len(node.args) != 1:
                    raise TheodoreParseError(
                        f"Not(...) expects a single argument, got: {ast.dump(node)}"
                    )
                return Not(_expr_to_formula(node.args[0]))

            # Logical And / Or as functions: And(phi1, phi2, ...), Or(...)
            if fname in ("And", "Or"):
                if not node.args:
                    raise TheodoreParseError(
                        f"{fname}(...) expects at least one argument, got: {ast.dump(node)}"
                    )
                args = [_expr_to_formula(a) for a in node.args]
                if fname == "And":
                    return And(args)
                else:
                    return Or(args)

            # Implies(phi, psi)
            if fname == "Implies":
                if len(node.args) != 2:
                    raise TheodoreParseError(
                        f"Implies(...) expects 2 arguments, got: {ast.dump(node)}"
                    )
                left = _expr_to_formula(node.args[0])
                right = _expr_to_formula(node.args[1])
                return Implies(left=left, right=right)

            # Any other function-style call: build a generic FuncCall and recurse.
            # This covers ToInt(...), RealVal(...), etc.
            args = tuple(_expr_to_formula(a) for a in node.args)
            return FuncCall(func=fname, args=args)

        # Something like obj.method(...): use textual name as 'func'
        func_str = (
            ast.unparse(node.func)
            if hasattr(ast, "unparse")
            else _fallback_func_str(node.func)
        )
        args = tuple(_expr_to_formula(a) for a in node.args)
        return FuncCall(func=func_str, args=args)

    raise TheodoreParseError(f"Unsupported expression node: {ast.dump(node)}")


def _handle_quantifier_call(fname: str, call: ast.Call) -> Formula:
    if len(call.args) != 2:
        raise TheodoreParseError(
            f"{fname} expects two arguments (vars list, body), got: {ast.dump(call)}"
        )

    vars_arg, body_arg = call.args

    if not isinstance(vars_arg, (ast.List, ast.Tuple)):
        raise TheodoreParseError(
            f"{fname} first argument must be a list/tuple of variable names, got: {ast.dump(vars_arg)}"
        )

    var_names: List[str] = []
    for elt in vars_arg.elts:
        if isinstance(elt, ast.Name):
            var_names.append(elt.id)
        else:
            raise TheodoreParseError(
                f"{fname} variable list elements must be simple names, got: {ast.dump(elt)}"
            )

    body_formula = _expr_to_formula(body_arg)

    if fname == "ForAll":
        return ForAll(vars=var_names, body=body_formula)
    else:
        return Exists(vars=var_names, body=body_formula)


# --- String builders for "opaque" nodes ------------------------------------


def _subscript_to_str(node: ast.Subscript) -> str:
    """
    Render a subscript expression like `signal_5[s]` into a string.

    We use a conservative textual encoding for now.
    """
    base = _expr_to_formula(node.value)
    # handle Python 3.8 vs 3.9+ index AST changes
    slice_node = node.slice
    if isinstance(slice_node, ast.Index):  # type: ignore[attr-defined]
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
    raise TheodoreParseError(f"Unsupported binary operator: {op!r}")


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
    raise TheodoreParseError(f"Unsupported comparison operator: {op!r}")
