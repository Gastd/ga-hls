from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import ConfigError, load_config
from .lang.analysis import collect_vars, formula_depth, formula_size
from .lang.internal_parser import InternalFormatError, parse_internal_json
from .lang.internal_encoder import InternalEncodeError, formula_to_internal_obj
from .lang.theodore_parser import TheodoreParseError, load_formula_from_property
from .pipeline import run_diagnostics, build_layout_from_config

from .ga import GA


def _get_version() -> str:
    try:
        return version("ga-hls")
    except PackageNotFoundError:
        # Fallback when running from source without an installed dist
        return "0.1.0-dev"

def _cmd_explain_positions(args: argparse.Namespace) -> int:
    """
    Show a table of all positions in the requirement formula, with:
      - Position index
      - Node type
      - Pretty-printed node
      - ARFF-style feature names (QUANTIFIERSk, NUMk, RELATIONALSk, TERMk, LOGICALSk)
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    layout = build_layout_from_config(cfg)
    positions = layout.positions
    if not positions:
        print("No positions found in formula.", file=sys.stderr)
        return 1

    sorted_idxs = sorted(positions.keys())

    # Classify positions by role
    quant_idxs = [i for i in sorted_idxs if positions[i].role == "QUANTIFIER"]
    logical_idxs = [i for i in sorted_idxs if positions[i].role == "LOGICAL_CONNECTIVE"]
    relop_idxs = [i for i in sorted_idxs if positions[i].role == "RELATION_OP"]

    # For each relational operator, find its first numeric child and first term child
    from ga_hls.lang.ast import IntConst, RealConst, Var, Subscript, FuncCall  # local import

    relop_slots: list[dict[str, int | None]] = []
    for r_idx in relop_idxs:
        relop_pos = positions[r_idx]
        prefix = relop_pos.path + "." if relop_pos.path else ""
        num_pos: int | None = None
        term_pos: int | None = None
        started = False

        for j in sorted_idxs:
            if j <= r_idx:
                continue
            p = positions[j]
            if prefix and p.path.startswith(prefix):
                started = True
                child = p.node
                if num_pos is None and isinstance(child, (IntConst, RealConst)):
                    num_pos = j
                if term_pos is None and isinstance(child, (Var, Subscript, FuncCall)):
                    term_pos = j
            else:
                if started:
                    break

        relop_slots.append({"relop": r_idx, "num": num_pos, "term": term_pos})

    # Build mapping: position index -> list of ARFF-style feature names
    features_by_pos: dict[int, list[str]] = {}

    def _add_feature(pos_index: int | None, name: str) -> None:
        if pos_index is None:
            return
        features_by_pos.setdefault(pos_index, []).append(name)

    # QUANTIFIERSk
    for k, q_idx in enumerate(quant_idxs):
        _add_feature(q_idx, f"QUANTIFIERS{k}")

    # LOGICALSk
    for k, l_idx in enumerate(logical_idxs):
        _add_feature(l_idx, f"LOGICALS{k}")

    # RELATIONALSk, NUMk, TERMk per relational slot
    for k, slot in enumerate(relop_slots):
        _add_feature(slot["relop"], f"RELATIONALS{k}")
        _add_feature(slot["num"], f"NUM{k}")
        _add_feature(slot["term"], f"TERM{k}")

    # Print table
    print(f"{'Pos':>3}  {'NodeType':<18}  {'Node':<60}  Features")
    print("-" * 100)

    for idx in sorted_idxs:
        pos = positions[idx]
        node = pos.node
        node_type = type(node).__name__

        node_repr = str(node)
        if len(node_repr) > 60:
            node_repr = node_repr[:57] + "..."

        feat_names = features_by_pos.get(idx, [])
        features_str = ", ".join(feat_names) if feat_names else "-"

        print(f"{idx:3d}  {node_type:<18}  {node_repr:<60}  {features_str}")

    return 0

def _cmd_explain_position(args: argparse.Namespace) -> int:
    """
    Show detailed info about a single mutation position:
      - AST node and a 'Pretty' form
      - ARFF-style features (QUANTIFIERSk, NUMk, RELATIONALSk, TERMk, LOGICALSk)
      - Current numeric_bounds for this position from the config, if present
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    layout = build_layout_from_config(cfg)
    positions = layout.positions
    if not positions:
        print("No positions found in formula.", file=sys.stderr)
        return 1

    idx = args.position
    if idx not in positions:
        print(f"No position {idx} found in formula.", file=sys.stderr)
        return 1

    pos = positions[idx]
    node = pos.node

    sorted_idxs = sorted(positions.keys())

    # Classify positions by role
    quant_idxs = [i for i in sorted_idxs if positions[i].role == "QUANTIFIER"]
    logical_idxs = [i for i in sorted_idxs if positions[i].role == "LOGICAL_CONNECTIVE"]
    relop_idxs = [i for i in sorted_idxs if positions[i].role == "RELATION_OP"]

    from ga_hls.lang.ast import IntConst, RealConst, Var, Subscript, FuncCall, RelOp as RelOpNode

    # Relational slots as in explain-positions
    relop_slots: list[dict[str, int | None]] = []
    for r_idx in relop_idxs:
        relop_pos = positions[r_idx]
        prefix = relop_pos.path + "." if relop_pos.path else ""
        num_pos: int | None = None
        term_pos: int | None = None
        started = False

        for j in sorted_idxs:
            if j <= r_idx:
                continue
            p = positions[j]
            if prefix and p.path.startswith(prefix):
                started = True
                child = p.node
                if num_pos is None and isinstance(child, (IntConst, RealConst)):
                    num_pos = j
                if term_pos is None and isinstance(child, (Var, Subscript, FuncCall)):
                    term_pos = j
            else:
                if started:
                    break

        relop_slots.append({"relop": r_idx, "num": num_pos, "term": term_pos})

    # Build mapping: position index -> list of ARFF-style feature names
    features_by_pos: dict[int, list[str]] = {}

    def _add_feature(pos_index: int | None, name: str) -> None:
        if pos_index is None:
            return
        features_by_pos.setdefault(pos_index, []).append(name)

    for k, q_idx in enumerate(quant_idxs):
        _add_feature(q_idx, f"QUANTIFIERS{k}")
    for k, l_idx in enumerate(logical_idxs):
        _add_feature(l_idx, f"LOGICALS{k}")
    for k, slot in enumerate(relop_slots):
        _add_feature(slot["relop"], f"RELATIONALS{k}")
        _add_feature(slot["num"], f"NUM{k}")
        _add_feature(slot["term"], f"TERM{k}")

    feat_names = sorted(features_by_pos.get(idx, []))

    # AST node description
    ast_node_desc = str(node)
    pretty_repr = str(node)

    # If this is a RelOp, make it look like RelOp("<", left=TERMk, right=NUMk)
    relop_k: int | None = None
    if isinstance(node, RelOpNode):
        for k, slot in enumerate(relop_slots):
            if slot["relop"] == idx:
                relop_k = k
                break

    if isinstance(node, RelOpNode) and relop_k is not None:
        ast_node_desc = f'RelOp("{node.op}", left=TERM{relop_k}, right=NUM{relop_k})'
        term_pos = relop_slots[relop_k]["term"]
        if term_pos is not None:
            left_term_repr = str(positions[term_pos].node)
        else:
            left_term_repr = str(node.left)
        pretty_repr = f"{left_term_repr} {node.op} NUM{relop_k}"

    # --- Print header ---
    print(f"Position {idx}")
    print("----------")
    print(f"AST node:      {ast_node_desc}")
    print(f"Pretty:        {pretty_repr}")
    print()

    # --- Affects ARFF ---
    print("Affects ARFF:")
    if not feat_names:
        print("  (no direct ARFF features)")
    else:
        for name in feat_names:
            if name.startswith("RELATIONALS"):
                print(f"  - {name}: {{<, >, <=, >=}}")
            elif name.startswith("QUANTIFIERS"):
                print(f"  - {name}: {{ForAll, Exists}}")
            elif name.startswith("LOGICALS"):
                print(f"  - {name}: {{And, Or}}")
            elif name.startswith("NUM"):
                print(f"  - {name}: NUMERIC")
            elif name.startswith("TERM"):
                # Try to show something like {v_speed[...], t}
                # Use the term child for the corresponding relational slot, if available.
                k = None
                for prefix in ("TERM",):
                    if name.startswith(prefix):
                        suffix = name[len(prefix):]
                        if suffix.isdigit():
                            k = int(suffix)
                        break
                term_node_repr = "t"
                if k is not None and k < len(relop_slots):
                    term_pos = relop_slots[k]["term"]
                    if term_pos is not None:
                        term_node_repr = str(positions[term_pos].node)
                print(f"  - {name}: {{{term_node_repr}, t}}")
            else:
                print(f"  - {name}: (unknown domain)")
    print()

    # --- Current bounds (from config) ---
    print("Current bounds (from config):")

    mut_cfg = None
    if hasattr(cfg, "mutation"):
        mut_cfg = cfg.mutation
    elif hasattr(cfg, "ga") and hasattr(cfg.ga, "mutation"):
        mut_cfg = cfg.ga.mutation

    numeric_bounds = None
    if mut_cfg is not None:
        if hasattr(mut_cfg, "numeric_bounds"):
            numeric_bounds = getattr(mut_cfg, "numeric_bounds")
        elif isinstance(mut_cfg, dict) and "numeric_bounds" in mut_cfg:
            numeric_bounds = mut_cfg["numeric_bounds"]

    value = None
    if numeric_bounds and isinstance(numeric_bounds, dict):
        if idx in numeric_bounds:
            value = numeric_bounds[idx]
        elif str(idx) in numeric_bounds:
            value = numeric_bounds[str(idx)]

    if value is not None:
        print(f'  - numeric_bounds["{idx}"]: {value}')
    else:
        print("  - none")

    return 0

def _cmd_run(args) -> int:
    """
    Main GA/diagnostics entrypoint.

    Canonical input: a ThEodorE-generated Python property file
    (e.g., property_01_err_twenty.py) configured in Config.input.requirement_file.

    Pipeline:
      ThEodorE property Python -> Formula AST
                              -> internal JSON -> GA
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Error loading config: {e}")
        return 1

    requirement_path = cfg.input.requirement_file
    print(f"Loading ThEodorE Python requirement from: {requirement_path}")

    # 1) Parse requirement into Formula AST (ThEodorE Python is canonical)
    try:
        formula = load_formula_from_property(requirement_path)
    except TheodoreParseError as e:
        print(f"Failed to parse ThEodorE Python requirement file:\n  {e}")
        return 1

    print("\nParsed requirement as AST:")
    print(formula)

    # 2) Encode AST into legacy internal JSON format for GA
    try:
        internal_obj = formula_to_internal_obj(formula)
    except InternalEncodeError as e:
        print(f"Failed to encode AST into internal GA format:\n  {e}")
        return 1

    print("\nStarting GA with encoded initial formula...")
    ga = GA(init_form=internal_obj)
    datasets = ga.evolve()
    for qty, arff_path in datasets.items():
        run_j48(arff_path, qty, ga.path)

    return 0

def _cmd_inspect_internal(args: argparse.Namespace) -> int:
    """
    Inspect an internal JSON list-of-lists formula and print basic AST info.

    Usage:
        ga-hls inspect-internal --file path/to/formula.json
    """
    path = Path(args.file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading {path!s}: {exc}", file=sys.stderr)
        return 1

    try:
        formula = parse_internal_json(text)
    except InternalFormatError as exc:
        print(f"Error parsing internal formula from {path!s}:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1

    print("Parsed formula:")
    print(f"  {formula}")
    print()

    size = formula_size(formula)
    depth = formula_depth(formula)
    vars_ = sorted(collect_vars(formula))

    print("AST statistics:")
    print(f"  Size (nodes): {size}")
    print(f"  Depth      : {depth}")
    print(f"  Variables  : {vars_ if vars_ else 'none'}")

    return 0

def _cmd_inspect_theodore(args) -> int:
    """
    Inspect a ThEodorE-generated Python property file specified in the config.

    Canonical input: property_X.py generated in ThEodorE (src-gen/*.py).
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Error loading config: {e}")
        return 1

    req_path = cfg.input.requirement_file
    print(f"Inspecting ThEodorE Python requirement file: {req_path}")

    try:
        formula = load_formula_from_property(req_path)
    except TheodoreParseError as e:
        print(f"Failed to parse ThEodorE Python requirement file:\n  {e}")
        return 1

    print("\nParsed formula (AST pretty-print):")
    print(formula)
    print("\nFormula stats:")
    print(f"  size  = {formula_size(formula)}")
    print(f"  depth = {formula_depth(formula)}")
    print(f"  vars  = {sorted(collect_vars(formula))}")

    return 0

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ga-hls",
        description="ga-hls: GA-based mutation and diagnostics for HLS/ThEodorE requirements.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed ga-hls version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run ga-hls using a JSON config file (currently: load & summarize).",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON configuration file.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-internal",
        help="Inspect an internal JSON list-of-lists formula.",
    )
    inspect_parser.add_argument(
        "--file",
        required=True,
        help="Path to a file containing the internal JSON formula.",
    )

    inspect_theodore_parser = subparsers.add_parser(
        "inspect-theodore",
        help="Inspect a ThEodorE/HLS Python property file defined in the config.",
    )
    inspect_theodore_parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file.",
    )
    inspect_theodore_parser.set_defaults(cmd="inspect_theodore")

    explain_pos_p = subparsers.add_parser(
        "explain-positions",
        help="Describe GA mutation positions for the configured requirement.",
    )
    explain_pos_p.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file.",
    )
    explain_pos_p.set_defaults(func=_cmd_explain_positions)

    explain_pos_single_p = subparsers.add_parser(
        "explain-position",
        help="Show detailed info about a single mutation position.",
    )
    explain_pos_single_p.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file.",
    )
    explain_pos_single_p.add_argument(
        "--position",
        type=int,
        required=True,
        help="Position index as used in mutation.allowed_positions.",
    )
    explain_pos_single_p.set_defaults(func=_cmd_explain_position)


    args = parser.parse_args(argv)

    # Global --version only (no subcommand)
    if args.version and args.command is None:
        print(_get_version())
        return 0

    if args.command == "run":
        try:
            cfg = load_config(args.config)
        except ConfigError as e:
            print(f"[ga-hls] Error loading config: {e}")
            return 1

        run_diagnostics(cfg)
        return 0
        
    if args.command == "inspect-internal":
        return _cmd_inspect_internal(args)

    if args.command == "inspect-theodore":
        return _cmd_inspect_theodore(args)

    if args.command == "explain-positions":
        return _cmd_explain_positions(args)

    if args.command == "explain-position":
        return _cmd_explain_position(args)

    # No subcommand: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
