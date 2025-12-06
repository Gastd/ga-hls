# ga-hls/lang/internal_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


@dataclass
class PositionInfo:
    index: int             # the integer used in allowed_positions / numeric_bounds
    node: Formula          # the AST node
    path: str              # a stable textual path, e.g. "body.antecedent.args[2].right"
    role: str              # e.g. "NUMERIC_THRESHOLD", "RELATION_OP", "LOGICAL_CONNECTIVE"

@dataclass
class FeatureInfo:
    arff_name: str         # e.g. "NUM2"
    kind: str              # "NUMERIC", "NOMINAL"
    position_index: int    # which PositionInfo this came from

@dataclass
class FormulaLayout:
    positions: Dict[int, PositionInfo]        # index -> PositionInfo
    features: List[FeatureInfo]               # in ARFF column order

def _infer_role(node: Formula) -> str:
    """
    Heuristic role classification for layout / introspection.
    """
    if isinstance(node, ForAll) or isinstance(node, Exists):
        return "QUANTIFIER"
    if isinstance(node, And) or isinstance(node, Or):
        return "LOGICAL_CONNECTIVE"
    if isinstance(node, Not):
        return "NEGATION"
    if isinstance(node, Implies):
        return "IMPLIES"
    if isinstance(node, RelOp):
        return "RELATION_OP"
    if isinstance(node, ArithOp):
        return "ARITH_OP"
    if isinstance(node, (IntConst, RealConst)):
        return "NUMERIC_LITERAL"
    if isinstance(node, BoolConst):
        return "BOOLEAN_LITERAL"
    if isinstance(node, (Var, Subscript, FuncCall)):
        return "TERM"
    return "EXPR"


def _collect_positions(formula: Formula) -> Dict[int, PositionInfo]:
    """
    Traverse the formula in a deterministic order (preorder) and assign
    integer position indices with path strings and heuristic roles.
    """
    positions: Dict[int, PositionInfo] = {}
    next_index = 0

    def visit(node: Formula, path: str) -> None:
        nonlocal next_index
        idx = next_index
        next_index += 1

        positions[idx] = PositionInfo(
            index=idx,
            node=node,
            path=path or ".",
            role=_infer_role(node),
        )

        # Recurse according to node type
        if isinstance(node, ForAll) or isinstance(node, Exists):
            visit(node.body, f"{path}.body" if path else "body")

        elif isinstance(node, And) or isinstance(node, Or):
            for i, arg in enumerate(node.args):
                child_path = f"{path}.args[{i}]" if path else f"args[{i}]"
                visit(arg, child_path)

        elif isinstance(node, Not):
            child_path = f"{path}.arg" if path else "arg"
            visit(node.arg, child_path)

        elif isinstance(node, Implies):
            left_path = f"{path}.left" if path else "left"
            right_path = f"{path}.right" if path else "right"
            visit(node.left, left_path)
            visit(node.right, right_path)

        elif isinstance(node, RelOp):
            left_path = f"{path}.left" if path else "left"
            right_path = f"{path}.right" if path else "right"
            visit(node.left, left_path)
            visit(node.right, right_path)

        elif isinstance(node, ArithOp):
            left_path = f"{path}.left" if path else "left"
            right_path = f"{path}.right" if path else "right"
            visit(node.left, left_path)
            visit(node.right, right_path)

    visit(formula, "")
    return positions

def _build_features(positions: Dict[int, PositionInfo]) -> List[FeatureInfo]:
    """
    Map PositionInfo entries to ARFF feature slots.

    The policy here is:
      - every quantifier becomes QUANTIFIERS<i>
      - every TERM-like node becomes TERM<i>
      - every RelOp becomes RELATIONALS<i>
      - every numeric literal becomes NUM<i>
      - every logical connective (And/Or) becomes LOGICALS<i>
      - every Implies becomes IMP<i>

    This walks positions in ascending index order, so the mapping is deterministic.
    """
    features: List[FeatureInfo] = []

    quant_idx = 0
    term_idx = 0
    rel_idx = 0
    num_idx = 0
    log_idx = 0
    imp_idx = 0

    for pos in sorted(positions.keys()):
        pi = positions[pos]

        if pi.role == "QUANTIFIER":
            arff_name = f"QUANTIFIERS{quant_idx}"
            quant_idx += 1
            features.append(
                FeatureInfo(
                    arff_name=arff_name,
                    kind="NOMINAL",
                    position_index=pi.index,
                )
            )

        elif pi.role == "TERM":
            arff_name = f"TERM{term_idx}"
            term_idx += 1
            features.append(
                FeatureInfo(
                    arff_name=arff_name,
                    kind="NOMINAL",
                    position_index=pi.index,
                )
            )

        elif pi.role == "RELATION_OP":
            arff_name = f"RELATIONALS{rel_idx}"
            rel_idx += 1
            features.append(
                FeatureInfo(
                    arff_name=arff_name,
                    kind="NOMINAL",
                    position_index=pi.index,
                )
            )

        elif pi.role == "NUMERIC_LITERAL":
            arff_name = f"NUM{num_idx}"
            num_idx += 1
            features.append(
                FeatureInfo(
                    arff_name=arff_name,
                    kind="NUMERIC",
                    position_index=pi.index,
                )
            )

        elif pi.role == "LOGICAL_CONNECTIVE":
            arff_name = f"LOGICALS{log_idx}"
            log_idx += 1
            features.append(
                FeatureInfo(
                    arff_name=arff_name,
                    kind="NOMINAL",
                    position_index=pi.index,
                )
            )

        elif pi.role == "IMPLIES":
            arff_name = f"IMP{imp_idx}"
            imp_idx += 1
            features.append(
                FeatureInfo(
                    arff_name=arff_name,
                    kind="NOMINAL",
                    position_index=pi.index,
                )
            )

    return features

class InternalEncodeError(RuntimeError):
    """Raised when we cannot encode a Formula into the internal JSON format."""


def formula_to_internal_obj(formula: Formula) -> Any:
    """
    Encode a `Formula` AST node into the original internal list-of-lists format
    expected by `treenode.parse`.

    This is the inverse of `ga-hls.lang.internal_parser.parse_internal_obj` for
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

def encode_with_layout(formula: Formula) -> tuple[Any, FormulaLayout]:
    """
    Convenience helper that returns both the internal representation and
    a FormulaLayout describing mutation/position and feature metadata.
    """
    internal = formula_to_internal_obj(formula)
    positions = _collect_positions(formula)

    # Build ARFF feature mapping for all relevant positions.
    features = _build_features(positions)
    layout = FormulaLayout(positions=positions, features=features)
    
    return internal, layout
