"""Run iFeature and merge feature types. Uses scripts/run_ifeature.py logic."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_ifeature_and_merge(
    input_fasta: str,
    feature_types: list[str],
    output_csv: str,
    ifeature_script: str,
    python_exe: str | None = None,
) -> str:
    """Call run_ifeature script and return path to merged CSV."""
    if python_exe is None:
        python_exe = sys.executable
    # From src/acp_learn/data/ -> project root is parents[3]
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
    script = root / "scripts" / "run_ifeature.py"
    cmd = [
        python_exe,
        str(script),
        "--input", str(input_fasta),
        "--out", str(output_csv),
        "--ifeature-script", str(ifeature_script),
    ]
    cmd += ["--types"] + feature_types
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"run_ifeature failed: {result.stderr}")
    return output_csv
