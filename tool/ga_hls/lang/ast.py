from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol


class Formula:
    """
    Base type for all formula AST nodes.
    """
    pass


# --- Atomic terms -----------------------------------------------------------


@dataclass(frozen=True)
class BoolConst(Formula):
    value: bool

    def __str__(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True)
class IntConst(Formula):
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class RealConst(Formula):
    value: float

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Var(Formula):
    """
    Variables or signal-like identifiers (e.g. "signal_5[s]").
    """
    name: str

    def __str__(self) -> str:
        return self.name


# --- Logical and relational structure --------------------------------------


@dataclass(frozen=True)
class RelOp(Formula):
    """
    Relational operator node: <, <=, >, >=, ==, !=
    """
    op: str
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class And(Formula):
    args: List[Formula]

    def __str__(self) -> str:
        return "(" + " ∧ ".join(str(a) for a in self.args) + ")"


@dataclass(frozen=True)
class Or(Formula):
    args: List[Formula]

    def __str__(self) -> str:
        return "(" + " ∨ ".join(str(a) for a in self.args) + ")"


@dataclass(frozen=True)
class Not(Formula):
    arg: Formula

    def __str__(self) -> str:
        return f"¬({self.arg})"


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"


@dataclass(frozen=True)
class ForAll(Formula):
    vars: List[str]
    body: Formula

    def __str__(self) -> str:
        vars_str = ", ".join(self.vars)
        return f"∀ {vars_str}. {self.body}"


@dataclass(frozen=True)
class Exists(Formula):
    vars: List[str]
    body: Formula

    def __str__(self) -> str:
        vars_str = ", ".join(self.vars)
        return f"∃ {vars_str}. {self.body}"


@dataclass(frozen=True)
class ArithOp(Formula):
    """
    Arithmetic operator node: +, -, *, /, etc.
    """
    op: str
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"
