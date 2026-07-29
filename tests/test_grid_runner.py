from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_grid_dry_run(tmp_path: Path) -> None:
    config = tmp_path / "grid.yaml"
    config.write_text(
        """
output_root: {output}
models:
  rf:
    module: mbmmc.train_rf
    fixed_args:
      ref_csv: ${{REF_CSV}}
      meta: ${{META}}
      model: rf
    grid:
      beta_threshold: [0.55, 0.60]
""".format(output=(tmp_path / "outputs").as_posix()),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["REF_CSV"] = "/tmp/reference.csv"
    env["META"] = "/tmp/meta.tsv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.run_grid",
            "--config",
            str(config),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[DRY-RUN]") == 2
    assert (tmp_path / "outputs" / "run_manifest.jsonl").exists()
