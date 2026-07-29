from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    [sys.executable, "train_rf.py", "--help"],
    [sys.executable, "train_crossnn.py", "--help"],
    [sys.executable, "train_mpcnet.py", "--help"],
    [sys.executable, "scripts/simulation/generate_cross_platform_in_silico_beta.py", "--help"],
    [sys.executable, "-m", "tools.run_grid", "--help"],
    [sys.executable, "-m", "tools.check_release", "--help"],
    [sys.executable, "scripts/simulation/generate_cross_platform_in_silico_beta.py", "--help"],
]


def test_cli_help() -> None:
    for command in COMMANDS:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
