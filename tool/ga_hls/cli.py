from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import ConfigError, load_config
from .lang.analysis import collect_vars, formula_depth, formula_size
from .lang.internal_parser import InternalFormatError, parse_internal_json
from .lang.hls_parser import HLSParseError, load_formula_from_hls 


def _get_version() -> str:
    try:
        return version("ga-hls")
    except PackageNotFoundError:
        # Fallback when running from source without an installed dist
        return "0.1.0-dev"


def _cmd_run(args: argparse.Namespace) -> int:
    """
    Load and validate a configuration file.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Loaded configuration:")
    print(f"  Requirement file : {cfg.input.requirement_file}")
    print(f"  Traces file      : {cfg.input.traces_file}")
    print(f"  Output directory : {cfg.input.output_dir}")
    print()
    print("  GA configuration:")
    print(f"    population_size: {cfg.ga.population_size}")
    print(f"    generations    : {cfg.ga.generations}")
    print(f"    crossover_rate : {cfg.ga.crossover_rate}")
    print(f"    mutation_rate  : {cfg.ga.mutation_rate}")
    print(f"    elitism        : {cfg.ga.elitism}")
    print(f"    seed           : {cfg.ga.seed}")
    print()
    print("  Diagnostics configuration:")
    print(f"    ARFF filename  : {cfg.diagnostics.arff_filename}")
    print(f"    Weka JAR       : {cfg.diagnostics.weka_jar}")
    print(f"    J48 options    : {cfg.diagnostics.j48_options}")

    # Later we will call the actual GA/diagnostics pipeline from here.
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
    Inspect an HLS/.hls requirements file specified in the config.

    This uses ga_hls.lang.hls_parser to translate the Specification into
    the internal Formula AST and prints a human-readable representation.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Error loading config: {e}")
        return 1

    prop_path = cfg.input.requirement_file
    print(f"Inspecting ThEodorE HLS requirement file: {prop_path}")

    try:
        formula = load_formula_from_hls(prop_path, property_name=None)
    except HLSParseError as e:
        print(f"Failed to parse HLS requirement file:\n  {e}")
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
