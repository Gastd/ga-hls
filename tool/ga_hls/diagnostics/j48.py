# tool/ga_hls/diagnostics/j48.py

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run_j48(arff_file: str, qty: float, out_dir: str, timeout: int = 100) -> str:
    """
    Run Weka J48 on the given ARFF file, saving the model and .out file
    into out_dir. Returns the path to the .out file.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"java -Xmx1024m weka.classifiers.trees.J48 "
        f"-t {arff_file} -k "
        f"-d {out_dir_path}/J48-data-{int(qty * 100)}.model"
    )
    print(cmd)
    weka_tk = shlex.split(cmd)

    try:
        run_process = subprocess.run(
            weka_tk,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
        print(run_process.stdout)
        print(run_process.stderr)
        out_path = out_dir_path / f"J48-data-{int(qty * 100)}.out"
        with out_path.open("w") as f:
            f.write(run_process.stdout)
            f.write(run_process.stderr)
        return str(out_path)
    except Exception as exc:
        print(f"[ga-hls] Weka J48 failed: {exc}")
        return ""
