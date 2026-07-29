from __future__ import annotations

import re
from pathlib import Path


def test_simulator_has_stable_public_name_and_main() -> None:
    path = Path("scripts/simulation/generate_cross_platform_in_silico_beta.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "def build_arg_parser()" in text
    assert "def main()" in text
    assert 'if __name__ == "__main__":' in text
    assert not re.search(r"\bv6(?:[._]\d+)*\b", text, flags=re.IGNORECASE)
    assert "LOCKED" not in text
