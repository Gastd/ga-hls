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

from .ga import GA


def _get_version() -> str:
    try:
        return version("ga-hls")
    except PackageNotFoundError:
        # Fallback when running from source without an installed dist
        return "0.1.0-dev"

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
    ga.evolve()

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

    # `run` subcommand (config-based pipeline entrypoint, still stubbed)
    run_parser = subparsers.add_parser(
        "run",
        help="Run ga-hls using a JSON config file (currently: load & summarize).",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON configuration file.",
    )

    # `inspect-internal` subcommand
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


    args = parser.parse_args(argv)

    # Global --version only (no subcommand)
    if args.version and args.command is None:
        print(_get_version())
        return 0

    if args.command == "run":
        return _cmd_run(args)

    if args.command == "inspect-internal":
        return _cmd_inspect_internal(args)

    if args.command == "inspect-theodore":
        return _cmd_inspect_theodore(args)

    # No subcommand: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
