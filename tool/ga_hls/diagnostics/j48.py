# tool/ga_hls/diagnostics/j48.py

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _resolve_weka_classpath(weka_jar: Optional[str] = None) -> str:
    """
    Decide the full Java classpath to use for Weka:

    1. Start from:
       - explicit weka_jar argument, OR
       - WEKA_JAR env var, OR
       - /opt/weka/weka.jar

    2. If /opt/weka/bounce.jar exists and is not already in the classpath,
       append it (Weka depends on org.bounce.net.DefaultAuthenticator).
    """
    base = weka_jar or os.environ.get("WEKA_JAR") or "/opt/weka/weka.jar"
    paths = [p for p in base.split(":") if p]

    bounce_default = "/opt/weka/bounce.jar"
    if bounce_default not in paths and Path(bounce_default).exists():
        paths.append(bounce_default)

    return ":".join(paths)


def _count_instances_in_arff(arff_file: str | Path) -> int:
    """
    Count data instances in a Weka ARFF file (lines after @data that are
    non-empty and not comments).
    """
    arff_path = Path(arff_file)
    if not arff_path.exists():
        return 0

    in_data = False
    count = 0
    with arff_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("@data"):
                in_data = True
                continue
            if not in_data:
                continue
            if stripped.startswith("%"):
                continue
            # Treat every non-empty, non-comment line after @data as an instance
            count += 1
    return count


def run_j48(
    arff_file: str,
    qty: float,
    out_dir: str,
    timeout: int = 100,
    weka_jar: Optional[str] = None,
) -> str:
    """
    Run Weka J48 on the given ARFF file, saving the model and .out file
    into out_dir. Returns the path to the .out file.

    weka_jar:
        Optional explicit base classpath / jar. If not provided, WEKA_JAR
        env var is used; if that is also absent, defaults to /opt/weka/weka.jar.
        In all cases, /opt/weka/bounce.jar is appended if present.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    arff_path = Path(arff_file)
    model_path = out_dir_path / f"J48-data-{int(qty * 100)}.model"

    # Determine a safe number of CV folds
    num_instances = _count_instances_in_arff(arff_path)
    if num_instances == 0:
        print(f"[ga-hls] J48: ARFF {arff_path} has 0 instances; skipping.")
        # Optionally write an empty .out explaining this
        out_path = out_dir_path / f"J48-data-{int(qty * 100)}.out"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"ARFF {arff_path} has 0 instances; J48 was not run.\n")
        return str(out_path)

    folds = min(10, num_instances)  # ensures folds <= instances

    classpath = _resolve_weka_classpath(weka_jar)

    cmd = [
        "java",
        "-Xmx1024m",
        "-cp",
        classpath,
        "weka.classifiers.trees.J48",
        "-t",
        str(arff_path),
        "-k",
        "-x",
        str(folds),
        "-d",
        str(model_path),
    ]

    print(" ".join(cmd))

    try:
        run_process = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        print(run_process.stdout)
        print(run_process.stderr)

        out_path = out_dir_path / f"J48-data-{int(qty * 100)}.out"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(run_process.stdout)
            f.write(run_process.stderr)

        return str(out_path)
    except Exception as exc:
        print(f"[ga-hls] Weka J48 failed: {exc}")
        return ""
