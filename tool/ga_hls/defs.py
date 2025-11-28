from __future__ import annotations

from pathlib import Path

# Root of the installed ga_hls package
_PACKAGE_ROOT = Path(__file__).resolve().parent

# Default ThEodorE Python property script used by legacy GA/Z3 integration.
# This is still a legacy path; the HLS file is now the canonical requirement
# source, but GA viability checks rely on this Python script.
FILEPATH = str(_PACKAGE_ROOT / "benchmark" / "property_01_err_twenty.py")
FILEPATH2 = FILEPATH
