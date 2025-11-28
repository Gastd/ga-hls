from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def run_j48(
    arff_file: str,
    output_dir: str,
    qty: float = 1.0,
    weka_jar: Optional[str] = None,
    extra_options: Optional[Iterable[str]] = None,
) -> Path:
    """
    Run Weka's J48 classifier on the given ARFF file.

    Parameters
    ----------
    arff_file:
        Path to the ARFF dataset file.
    output_dir:
        Directory where model/output files should be written.
    qty:
        Fraction of the dataset this run corresponds to (0.0 - 1.0), only
        used for naming the output files.
    weka_jar:
        Optional path to the Weka JAR. If None, assumes `weka` is available
        on the CLASSPATH.
    extra_options:
        Additional command-line options for J48, e.g. ["-C", "0.25", "-M", "2"].

    Returns
    -------
    Path to the `.out` file with J48's stdout/stderr.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arff_path = Path(arff_file)
    if not arff_path.is_file():
        raise FileNotFoundError(f"ARFF file not found: {arff_path}")

    qty_percent = int(qty * 100)
    model_path = out_dir / f"J48-data-{qty_percent}.model"
    out_path = out_dir / f"J48-data-{qty_percent}.out"

    # Base command: either "java -Xmx1024m -cp weka.jar weka.classifiers.trees.J48 ..."
    # or assume Weka is on the CLASSPATH.
    base = ["java", "-Xmx1024m"]
    if weka_jar is not None:
        base.extend(["-cp", weka_jar, "weka.classifiers.trees.J48"])
    else:
        base.append("weka.classifiers.trees.J48")

    cmd = [
        *base,
        "-t",
        str(arff_path),
        "-k",
        "-d",
        str(model_path),
    ]

    if extra_options:
        cmd.extend(list(extra_options))

    # Run J48.
    print("Running J48:", " ".join(shlex.quote(str(c)) for c in cmd))
    try:
        with out_path.open("w", encoding="utf-8") as out_fp:
            subprocess.run(
                cmd,
                stdout=out_fp,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
    except OSError as exc:
        raise RuntimeError(f"Failed to run J48: {exc}") from exc

    return out_path
