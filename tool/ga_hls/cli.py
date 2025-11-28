from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import ConfigError, load_config


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
    config_path = args.config

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"Configuration error while loading {config_path!r}:")
        print(f"  {exc}")
        return 1

    print("Loaded configuration:")
    print(f"  requirement_file : {cfg.input.requirement_file}")
    print(f"  traces_file      : {cfg.input.traces_file}")
    print(f"  output_dir       : {cfg.input.output_dir}")
    print()
    print("  GA parameters:")
    print(f"    population_size: {cfg.ga.population_size}")
    print(f"    generations    : {cfg.ga.generations}")
    print(f"    crossover_rate : {cfg.ga.crossover_rate}")
    print(f"    mutation_rate  : {cfg.ga.mutation_rate}")
    print(f"    elitism        : {cfg.ga.elitism}")
    print(f"    seed           : {cfg.ga.seed}")
    print()
    print("  Diagnostics:")
    print(f"    arff_filename  : {cfg.diagnostics.arff_filename}")
    print(f"    weka_jar       : {cfg.diagnostics.weka_jar}")
    print(f"    j48_options    : {cfg.diagnostics.j48_options}")

    # Basic path checks
    problems = False

    req_path = Path(cfg.input.requirement_file)
    traces_path = Path(cfg.input.traces_file)

    if not req_path.is_file():
        print(f"\nWARNING: requirement file does not exist: {req_path}")
        problems = True
    if not traces_path.is_file():
        print(f"WARNING: traces file does not exist: {traces_path}")
        problems = True

    out_dir = Path(cfg.input.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"\nERROR: could not create output directory {out_dir}: {exc}")
        return 1

    if problems:
        print(
            "\nConfig loaded, but there were missing input files "
            "(see warnings above)."
        )
        # Treat missing inputs as a non-successful validation for now.
        return 1

    print("\nConfig validation OK: inputs exist and output directory is ready.")
    return 0


def main(argv=None) -> int:
    """
    Entry point for the ga-hls command-line interface.
    """
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
        help="Load and validate a configuration file.",
    )
    run_parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to a JSON configuration file.",
    )

    args = parser.parse_args(argv)

    if args.version:
        print(_get_version())
        return 0

    if args.command == "run":
        return _cmd_run(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
