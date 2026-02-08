from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..lang.ast import (
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MutationConfig:
    """
    Configuration for AST-based formula mutation.
    """

    # Maximum number of node-level mutations per call to mutate_formula
    max_mutations: int = 1

    # Enable/disable classes of mutations
    enable_numeric_perturbation: bool = True
    enable_relop_flip: bool = True
    enable_logical_flip: bool = True
    enable_quantifier_flip: bool = True

    # restrict *which* node indices are allowed to mutate
    # (preorder indices from _walk(formula))
    allowed_positions: Optional[Set[int]] = None

    # Unified per-position constraints:
    # idx -> {"numeric": (lo, hi), "relational": [...], "equals": [...], "logical": [...], "arith": [...], "quantifier": [...]}
    allowed_changes: Dict[int, Dict[str, object]] = field(default_factory=dict)

    # How aggressively to perturb numeric constants. We reuse the idea of
    # "order of magnitude" steps from individual.get_new_term.
    int_magnitude_jitter: float = 1.0
    float_magnitude_jitter: float = 1.0

    # Operator families (mirroring individual.py)
    relational_ops: Tuple[str, ...] = ("<", ">", "<=", ">=")
    equals_ops: Tuple[str, ...] = ("==", "!=")
    logicals: Tuple[str, ...] = ("And", "Or", "Implies")
    quantifiers: Tuple[str, ...] = ("ForAll", "Exists")
    arithmetics: Tuple[str, ...] = ("+", "-", "*", "/")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def mutate_formula(
    formula: Formula,
    cfg: MutationConfig,
    rng: Optional[random.Random] = None,
) -> Formula:
    """
    Apply up to `cfg.max_mutations` random mutations to the given formula AST
    and return a NEW formula.

    This function is pure: it never mutates the input `formula` in place.

    Semantics of cfg.allowed_positions
    ----------------------------------
    - We conceptually number all nodes in the formula in a fixed preorder
      traversal: 0, 1, 2, ...
    - If cfg.allowed_positions is None:
        * Any node position may be selected for mutation.
    - If cfg.allowed_positions is a list of integers:
        * Only nodes whose preorder index is in that list are *eligible*
          for mutation (regardless of node kind: numeric literal, relop,
          logical connective, quantifier, etc.).

    Semantics of cfg.allowed_changes
    --------------------------------
    - cfg.allowed_changes is a per-position dictionary controlling what *values*
    a node may mutate to.
    - For numeric literals at position P:
        * if cfg.allowed_changes[P] has key "numeric", it must be [lo, hi] and the
        new numeric value is sampled inside that range.
        * if "numeric" is absent, numeric mutation falls back to a local ±10% window.
    - For operators at position P (logical/relational/equals/quantifier/arith):
        * if cfg.allowed_changes[P] defines the corresponding family, mutation is
        restricted to those tokens at that position.
    """
    if rng is None:
        rng = random

    if cfg.max_mutations <= 0:
        return formula

    # Flatten nodes to get a stable index space to pick from
    nodes: List[Formula] = list(_walk(formula))
    if not nodes:
        return formula

    all_indices = list(range(len(nodes)))

    # Restrict candidate positions if allowed_positions is set
    if cfg.allowed_positions is not None:
        allowed_set = set(cfg.allowed_positions)
        candidate_indices = [i for i in all_indices if i in allowed_set]
    else:
        candidate_indices = all_indices

    if not candidate_indices:
        print("[ga-hls][warning] mutate formula: no candidate indices, returning unchanged")
        return formula

    # Choose positions to mutate (without replacement)
    max_muts = min(cfg.max_mutations, len(candidate_indices))
    positions = rng.sample(candidate_indices, k=max_muts)

    # Rebuild once, mutating any node whose preorder index is in `positions`
    return _mutate_by_positions(formula, positions, cfg, rng)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _walk(f: Formula) -> Iterable[Formula]:
    """Preorder traversal of the AST."""
    stack: List[Formula] = [f]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, And) or isinstance(node, Or):
            # children stored in .args
            for child in reversed(node.args):
                stack.append(child)
        elif isinstance(node, Not):
            stack.append(node.arg)
        elif isinstance(node, Implies):
            stack.append(node.right)
            stack.append(node.left)
        elif isinstance(node, ForAll) or isinstance(node, Exists):
            stack.append(node.body)
        elif isinstance(node, RelOp) or isinstance(node, ArithOp):
            stack.append(node.right)
            stack.append(node.left)
        # BoolConst, IntConst, RealConst, Var: no children

def _mutate_by_positions(
    f: Formula,
    positions: List[int],
    cfg: MutationConfig,
    rng: random.Random,
) -> Formula:
    """
    Rebuild `f`, mutating nodes whose preorder index is in `positions`.

    `positions` is a list of integers corresponding to the preorder
    traversal index of each node. For each node, we:

      1. Recursively rebuild its children.
      2. If its index (here) is in `positions`, pass the rebuilt node
         to `_mutate_node(new_node, cfg, rng, here)` to obtain a possibly
         mutated node.

    `_mutate_node` is responsible for:
      - Deciding which specific mutation operator to apply (numeric
        perturbation, relop flip, logical flip, quantifier flip, etc.).
      - Looking at `cfg.allowed_changes.get(here)` to constrain numeric
        changes, if desired.
    """
    target_positions = set(positions)
    counter = 0  # preorder index

    def rebuild(node: Formula) -> Formula:
        nonlocal counter
        here = counter
        counter += 1

        # First rebuild children
        if isinstance(node, And):
            new_args = [rebuild(a) for a in node.args]
            new_node: Formula = And(args=new_args)
        elif isinstance(node, Or):
            new_args = [rebuild(a) for a in node.args]
            new_node = Or(args=new_args)
        elif isinstance(node, Not):
            new_node = Not(arg=rebuild(node.arg))
        elif isinstance(node, Implies):
            new_left = rebuild(node.left)
            new_right = rebuild(node.right)
            new_node = Implies(left=new_left, right=new_right)
        elif isinstance(node, ForAll):
            new_body = rebuild(node.body)
            new_node = ForAll(vars=list(node.vars), body=new_body)
        elif isinstance(node, Exists):
            new_body = rebuild(node.body)
            new_node = Exists(vars=list(node.vars), body=new_body)
        elif isinstance(node, RelOp):
            new_left = rebuild(node.left)
            new_right = rebuild(node.right)
            new_node = RelOp(op=node.op, left=new_left, right=new_right)
        elif isinstance(node, ArithOp):
            new_left = rebuild(node.left)
            new_right = rebuild(node.right)
            new_node = ArithOp(op=node.op, left=new_left, right=new_right)
        else:
            # Leaf / atomic: copy as-is
            new_node = node

        # Maybe mutate at this position
        if here in target_positions:
            new_node = _mutate_node(new_node, cfg, rng, here)

        return new_node

    return rebuild(f)

def _allowed_change(cfg: MutationConfig, idx: int, key: str) -> Optional[object]:
    """
    Return per-position restriction for a family, if any.
    Example keys: "numeric", "logical", "relational", "equals", "arith", "quantifier".
    """
    return cfg.allowed_changes.get(idx, {}).get(key)

def _filter_allowed(current: str, allowed: List[str]) -> List[str]:
    """Return allowed choices excluding current."""
    return [x for x in allowed if x != current]

def _mutate_node(node: Formula, cfg: MutationConfig, rng: random.Random, idx: int) -> Formula:
    """
    Mutate a single node, depending on its type and the config.
    """
    # --- Numeric perturbation ------------------------------------------------
    if cfg.enable_numeric_perturbation and isinstance(node, (IntConst, RealConst)):
        bounds_spec = _allowed_change(cfg, idx, "numeric")
        if isinstance(bounds_spec, (list, tuple)) and len(bounds_spec) == 2:
            lo, hi = float(bounds_spec[0]), float(bounds_spec[1])
        else:
            # Fallback: local ±10% window (at least size 1)
            v = float(node.value)
            span = max(abs(v) * 0.1, 1.0)
            lo, hi = v - span, v + span

        # Normalise bounds
        lo = float(lo)
        hi = float(hi)
        if lo > hi:
            lo, hi = hi, lo

        # Sample new value *inside* [lo, hi]
        if isinstance(node, IntConst):
            lo_i = int(round(lo))
            hi_i = int(round(hi))
            if lo_i > hi_i:
                lo_i, hi_i = hi_i, lo_i
            new_val = rng.randint(lo_i, hi_i)
            # print(
            #     f"[ga-hls][mutate] numeric at id={idx}, old={node.value}, "
            #     f"new={new_val}, bounds=({lo_i}, {hi_i})"
            # )
            return IntConst(value=new_val)

        else:  # RealConst
            new_val = rng.uniform(lo, hi)
            # print(
            #     f"[ga-hls][mutate] numeric at id={idx}, old={node.value}, "
            #     f"new={new_val:.6f}, bounds=({lo:.3f}, {hi:.3f})"
            # )
            return RealConst(value=new_val)

    # Operator flips
    if cfg.enable_relop_flip and isinstance(node, RelOp):
        return _mutate_relop(node, cfg, rng, idx)
    if cfg.enable_logical_flip and isinstance(node, (And, Or, Implies)):
        return _flip_logical(node, cfg, rng, idx)
    if cfg.enable_quantifier_flip and isinstance(node, (ForAll, Exists)):
        return _flip_quantifier(node, cfg, rng, idx)

    # Fallback: no change
    return node


# --- specific node mutations -------------------------------------------------


def _mutate_int_const(node: IntConst, cfg: MutationConfig, rng: random.Random) -> IntConst:
    """
    Roughly mirror individual.get_new_term(int) semantics:
    - For zero: +/- 1
    - For nonzero: perturb by one order-of-magnitude-ish step.
    """
    t = node.value
    if t == 0:
        return IntConst(1 if rng.random() > 0.5 else -1)

    magnitude = 10 ** int(math.log(abs(t), 10))
    jitter = max(1, int(magnitude * cfg.int_magnitude_jitter))
    if rng.random() > 0.5:
        return IntConst(t + rng.randint(jitter, 10 * jitter - 1))
    else:
        return IntConst(t - rng.randint(jitter, 10 * jitter - 1))


def _mutate_real_const(node: RealConst, cfg: MutationConfig, rng: random.Random) -> RealConst:
    """
    Rough approximation of the float behavior in get_new_term.
    """
    t = node.value
    if t == 0.0:
        # Just nudge a little bit
        delta = 10 ** -3 * cfg.float_magnitude_jitter
        if rng.random() > 0.5:
            return RealConst(t + rng.uniform(0, delta))
        else:
            return RealConst(t - rng.uniform(0, delta))

    mag = abs(t)
    if mag >= 1.0:
        base = 10 ** int(math.log(mag, 10))
        diff = rng.uniform(base, 10 * base) * cfg.float_magnitude_jitter
    else:
        # 0 < |t| < 1: small change
        diff = rng.uniform(0, 10 ** -3) * cfg.float_magnitude_jitter

    if rng.random() > 0.5:
        return RealConst(t + diff)
    else:
        return RealConst(t - diff)


def _choices_excluding(current: str, allowed: list[str] | tuple[str, ...]) -> list[str]:
    """
    Return allowed operator choices excluding the current operator.
    """
    return [x for x in allowed if x != current]

def _mutate_relop(node: RelOp, cfg: MutationConfig, rng: random.Random, idx: int) -> RelOp:
    op = node.op

    # Equality family
    if op in ("==", "!="):
        allowed = _allowed_change(cfg, idx, "equals")
        allowed_ops = list(cfg.equals_ops) if allowed is None else list(allowed)
        choices = _choices_excluding(op, allowed_ops)
        if not choices:
            return node
        return RelOp(op=rng.choice(choices), left=node.left, right=node.right)

    # Ordering family
    ordering_family = ("<", ">", "<=", ">=")
    if op in ordering_family:
        allowed = _allowed_change(cfg, idx, "relational")
        allowed_ops = list(cfg.relational_ops) if allowed is None else [o for o in list(allowed) if o in ordering_family]
        choices = _choices_excluding(op, allowed_ops)
        if not choices:
            return node
        return RelOp(op=rng.choice(choices), left=node.left, right=node.right)

    return node


def _flip_logical(node: And | Or | Implies, cfg: MutationConfig, rng: random.Random, idx: int) -> Formula:
    """
    Flip among And/Or/Implies, preserving children when possible.
    Note: Implies is binary. We only flip And/Or -> Implies when they have exactly 2 args.
    """
    allowed = _allowed_change(cfg, idx, "logical")
    allowed = list(cfg.logicals) if allowed is None else list(allowed)


    if isinstance(node, And):
        # And -> Or / Implies
        choices = [op for op in allowed if op != "And"]
        if not choices:
            return node
        new_op = rng.choice(choices)

        if new_op == "Or":
            return Or(args=node.args)

        if new_op == "Implies":
            if len(node.args) != 2:
                print("Tried to convert n-ary And/Or to binary Implies. Returning node without mutation.")
                return node  # cannot convert n-ary And to Implies without normalization
            return Implies(left=node.args[0], right=node.args[1])

        return node

    if isinstance(node, Or):
        # Or -> And / Implies
        choices = [op for op in allowed if op != "Or"]
        if not choices:
            return node
        new_op = rng.choice(choices)

        if new_op == "And":
            return And(args=node.args)

        if new_op == "Implies":
            if len(node.args) != 2:
                return node
            return Implies(left=node.args[0], right=node.args[1])

        return node

    # node is Implies
    choices = [op for op in allowed if op != "Implies"]
    if not choices:
        return node
    new_op = rng.choice(choices)

    if new_op == "And":
        return And(args=[node.left, node.right])
    if new_op == "Or":
        return Or(args=[node.left, node.right])

    return node


def _flip_quantifier(node: ForAll | Exists, cfg: MutationConfig, rng: random.Random, idx: int) -> Formula:
    """
    Flip between ForAll and Exists.
    """
    allowed = _allowed_change(cfg, idx, "quantifier")
    allowed_q = list(cfg.quantifiers) if allowed is None else list(allowed)

    if isinstance(node, ForAll) and "Exists" in allowed_q:
        return Exists(vars=list(node.vars), body=node.body)
    if isinstance(node, Exists) and "ForAll" in allowed_q:
        return ForAll(vars=list(node.vars), body=node.body)
    return node

