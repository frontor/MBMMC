from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_toy_data(tmp_path: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "examples/make_toy_data.py",
            "--outdir",
            str(tmp_path),
            "--samples-per-class",
            "4",
            "--features",
            "40",
        ]
    )
    reference = pd.read_csv(tmp_path / "legacy_reference.csv")
    meta = pd.read_csv(tmp_path / "meta.tsv", sep="\t")
    matrix = pd.read_csv(tmp_path / "sample_matrix.csv")
    labels = pd.read_csv(tmp_path / "labels.csv")
    assert reference.shape[0] == 40
    assert len(meta) == 12
    assert len(matrix) == len(labels) == 12
