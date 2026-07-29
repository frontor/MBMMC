from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_make_simulation_reference(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "examples/make_simulation_reference.py",
            "--outdir",
            str(tmp_path),
            "--features",
            "30",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

    beta = pd.read_csv(tmp_path / "reference_beta.csv")
    meta = pd.read_csv(tmp_path / "metadata.tsv", sep="	")
    assert beta.shape[0] == 30
    assert {"Sample", "Types", "SourceSplit", "BackgroundRole"}.issubset(meta.columns)
    assert (meta["Types"] == "CONTROL").any()
    assert (meta["BackgroundRole"] == "tumor_source").any()
