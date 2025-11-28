import argparse
from importlib.metadata import version, PackageNotFoundError


def _get_version() -> str:
    try:
        return version("ga-hls")
    except PackageNotFoundError:
        # Fallback when running from source without an installed dist
        return "0.1.0-dev"


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

    args = parser.parse_args(argv)

    if args.version:
        print(_get_version())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
