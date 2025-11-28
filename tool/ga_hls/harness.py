from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class Verdict(Enum):
    SAT = auto()
    UNSAT = auto()
    ERROR = auto()


@dataclass
class HarnessResult:
    """Result of executing a property script with Z3."""
    verdict: Verdict
    stdout: str
    stderr: str
    returncode: int
    error: Optional[BaseException] = None


SAT_MARK = "SATISFIED"
UNSAT_MARK = "VIOLATED"


def run_property_script(
    filepath: str | Path,
    timeout: int = 50,
) -> HarnessResult:
    """
    Execute the given property Python file in a subprocess and interpret its outcome.

    This function is robust to Python/Z3 exceptions: they are captured and represented
    as Verdict.ERROR instead of raising back into the GA loop.
    """
    path = Path(filepath)

    # Build a command that uses the same Python interpreter as the current process.
    cmd = [sys.executable, str(path)]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = proc.returncode

    except subprocess.TimeoutExpired as exc:
        # Treat timeouts as ERROR with a helpful message
        return HarnessResult(
            verdict=Verdict.ERROR,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + "\n[TIMEOUT]",
            returncode=-1,
            error=exc,
        )
    except Exception as exc:
        # Any unexpected failure starting the process
        return HarnessResult(
            verdict=Verdict.ERROR,
            stdout="",
            stderr=str(exc),
            returncode=-1,
            error=exc,
        )

    # Decide verdict based on stdout contents first
    if SAT_MARK in stdout:
        verdict = Verdict.SAT
    elif UNSAT_MARK in stdout:
        verdict = Verdict.UNSAT
    else:
        # No clear SAT/UNSAT markers; treat as ERROR but keep stdout/stderr for analysis.
        verdict = Verdict.ERROR

    # If Python traceback / NameError shows up, we *force* ERROR even if markers exist.
    if "Traceback (most recent call last):" in stderr or "NameError:" in stderr:
        verdict = Verdict.ERROR

    return HarnessResult(
        verdict=verdict,
        stdout=stdout,
        stderr=stderr,
        returncode=rc,
        error=None,
    )
