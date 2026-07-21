#!/usr/bin/env python
"""
Run iFeature for selected descriptor types and merge outputs into one CSV.
Can be driven by YAML config or CLI. Used by the training pipeline and for preprocessing.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def run_ifeature_type(
    ifeature_script: str,
    input_fasta: str,
    feature_type: str,
    output_path: str,
    python_exe: str | None = None,
) -> None:
    if python_exe is None:
        python_exe = sys.executable
    cmd = [
        python_exe,
        ifeature_script,
        "--file", input_fasta,
        "--type", feature_type,
        "--out", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(ifeature_script).parent))
    if result.returncode != 0:
        raise RuntimeError(
            f"iFeature failed for type={feature_type}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def load_ifeature_output(path: str, feature_type: str) -> pd.DataFrame:
    text = Path(path).read_text().strip()
    if text == "Descriptor calculation failed.":
        raise RuntimeError(
            f"iFeature descriptor {feature_type} failed. "
            "The output file only contains 'Descriptor calculation failed.'."
        )
    df = pd.read_csv(path, sep="\t")
    failed_cols = [c for c in df.columns if "descriptor calculation failed" in str(c).lower()]
    if failed_cols:
        raise RuntimeError(
            f"iFeature descriptor {feature_type} failed and produced invalid columns: {failed_cols}"
        )
    if df.empty:
        raise RuntimeError(f"iFeature descriptor {feature_type} produced no feature rows.")
    if "#" not in df.columns:
        raise RuntimeError(f"iFeature descriptor {feature_type} output is missing the sequence id column '#'.")
    if df["#"].duplicated().any():
        dupes = df.loc[df["#"].duplicated(), "#"].head().tolist()
        raise RuntimeError(f"iFeature descriptor {feature_type} produced duplicate sequence ids: {dupes}")
    df = df.set_index("#")
    return df


def run_and_merge(
    input_fasta: str,
    feature_types: list[str],
    output_csv: str,
    ifeature_script: str | None = None,
    ifeature_dir: str | None = None,
    python_exe: str | None = None,
    keep_intermediate: bool = False,
) -> str:
    """
    Run iFeature for each type, merge on row order, save to output_csv.
    Returns path to output_csv.
    """
    if ifeature_script is None:
        ifeature_dir = ifeature_dir or os.environ.get("ACP_IFEATURE_DIR", "iFeature")
        ifeature_script = str(Path(ifeature_dir) / "iFeature.py")
    script_dir = Path(ifeature_script).parent
    input_fasta = Path(input_fasta).resolve()
    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_csv.parent / "_ifeature_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for ft in feature_types:
        out_path = out_dir / f"{ft}.tsv"
        run_ifeature_type(
            ifeature_script=str(Path(ifeature_script).resolve()),
            input_fasta=str(input_fasta),
            feature_type=ft,
            output_path=str(out_path),
            python_exe=python_exe,
        )
        dfs.append(load_ifeature_output(str(out_path), ft))
        if not keep_intermediate:
            out_path.unlink(missing_ok=True)

    row_counts = {ft: len(df) for ft, df in zip(feature_types, dfs)}
    if len(set(row_counts.values())) != 1:
        raise RuntimeError(f"iFeature descriptors produced different row counts: {row_counts}")
    id_sets = {ft: set(df.index) for ft, df in zip(feature_types, dfs)}
    first_type = feature_types[0]
    first_ids = id_sets[first_type]
    mismatched_ids = {
        ft: len(first_ids.symmetric_difference(ids))
        for ft, ids in id_sets.items()
        if ids != first_ids
    }
    if mismatched_ids:
        raise RuntimeError(f"iFeature descriptors produced different sequence id sets: {mismatched_ids}")

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.join(d)
    merged = merged.reset_index().rename(columns={"#": "ID"})
    merged.to_csv(output_csv, index=False)
    if not keep_intermediate and out_dir.exists():
        out_dir.rmdir()
    return str(output_csv)


def main():
    ap = argparse.ArgumentParser(description="Run iFeature and merge feature types")
    ap.add_argument("--config", type=str, help="Path to YAML config (input_fasta, feature_types, output_csv, etc.)")
    ap.add_argument("--input", type=str, help="Input FASTA file")
    ap.add_argument("--types", nargs="+", default=["CTDC", "CKSAAGP", "CTDD"], help="Feature types")
    ap.add_argument("--out", type=str, help="Output merged CSV path")
    ap.add_argument("--ifeature-script", type=str, default=None, help="Path to iFeature.py")
    ap.add_argument("--ifeature-dir", type=str, default=None, help="Directory containing iFeature.py")
    ap.add_argument("--python", type=str, default=None, help="Python executable for iFeature")
    ap.add_argument("--keep-intermediate", action="store_true", help="Keep per-type TSV files")
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        input_fasta = cfg["input_fasta"]
        feature_types = cfg.get("feature_types", ["CTDC", "CKSAAGP", "CTDD"])
        output_csv = cfg["output_csv"]
        ifeature_script = cfg.get("ifeature_script") or args.ifeature_script
        ifeature_dir = cfg.get("ifeature_dir") or args.ifeature_dir
        python_exe = cfg.get("python_exe") or args.python
    else:
        input_fasta = args.input
        feature_types = args.types
        output_csv = args.out
        ifeature_script = args.ifeature_script
        ifeature_dir = args.ifeature_dir
        python_exe = args.python
        if not input_fasta or not output_csv:
            ap.error("Without --config, --input and --out are required")

    path = run_and_merge(
        input_fasta=input_fasta,
        feature_types=feature_types,
        output_csv=output_csv,
        ifeature_script=ifeature_script,
        ifeature_dir=ifeature_dir,
        python_exe=python_exe,
        keep_intermediate=args.keep_intermediate,
    )
    print(f"Saved merged features to {path}")


if __name__ == "__main__":
    main()
