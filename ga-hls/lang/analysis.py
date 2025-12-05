from __future__ import annotations

from typing import Set

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


def formula_size(f: Formula) -> int:
    """
    Total number of AST nodes in the formula.
    """

    def _size(node: Formula) -> int:
        if isinstance(
            node,
            (BoolConst, IntConst, RealConst, Var),
        ):
            return 1

        if isinstance(node, RelOp):
            return 1 + _size(node.left) + _size(node.right)

        if isinstance(node, ArithOp):
            return 1 + _size(node.left) + _size(node.right)

        if isinstance(node, Not):
            return 1 + _size(node.arg)

        if isinstance(node, Implies):
            return 1 + _size(node.left) + _size(node.right)

        if isinstance(node, And):
            return 1 + sum(_size(a) for a in node.args)

        if isinstance(node, Or):
            return 1 + sum(_size(a) for a in node.args)

        if isinstance(node, ForAll):
            return 1 + _size(node.body)

        if isinstance(node, Exists):
            return 1 + _size(node.body)

        # Fallback: count as a single node if we ever add new subclasses
        return 1

    return _size(f)


def formula_depth(f: Formula) -> int:
    """
    Maximum depth of the formula AST (leaf has depth 1).
    """

    def _depth(node: Formula) -> int:
        if isinstance(
            node,
            (BoolConst, IntConst, RealConst, Var),
        ):
            return 1

        if isinstance(node, RelOp):
            return 1 + max(_depth(node.left), _depth(node.right))

        if isinstance(node, ArithOp):
            return 1 + max(_depth(node.left), _depth(node.right))

        if isinstance(node, Not):
            return 1 + _depth(node.arg)

        if isinstance(node, Implies):
            return 1 + max(_depth(node.left), _depth(node.right))

        if isinstance(node, And):
            if not node.args:
                return 1
            return 1 + max(_depth(a) for a in node.args)

        if isinstance(node, Or):
            if not node.args:
                return 1
            return 1 + max(_depth(a) for a in node.args)

        if isinstance(node, ForAll):
            return 1 + _depth(node.body)

        if isinstance(node, Exists):
            return 1 + _depth(node.body)

        return 1

    return _depth(f)


def collect_vars(f: Formula) -> Set[str]:
    """
    Collect all variable-like names (including signal refs) that occur in the formula.
    """

    vars_: Set[str] = set()

    def _visit(node: Formula) -> None:
        if isinstance(node, Var):
            vars_.add(node.name)
            return

        if isinstance(node, (BoolConst, IntConst, RealConst)):
            return

        if isinstance(node, RelOp):
            _visit(node.left)
            _visit(node.right)
            return

        if isinstance(node, ArithOp):
            _visit(node.left)
            _visit(node.right)
            return

        if isinstance(node, Not):
            _visit(node.arg)
            return

        if isinstance(node, Implies):
            _visit(node.left)
            _visit(node.right)
            return

        if isinstance(node, And):
            for a in node.args:
                _visit(a)
            return

        if isinstance(node, Or):
            for a in node.args:
                _visit(a)
            return

        if isinstance(node, ForAll):
            _visit(node.body)
            return

        if isinstance(node, Exists):
            _visit(node.body)
            return

    _visit(f)
    return vars_
