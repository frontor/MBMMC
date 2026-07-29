from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Scan only files maintained by this repository. Local virtual environments,
# installed dependencies, build products, caches, and generated outputs are not
# part of the MBMMC source tree and must not influence this naming test.
CONTROLLED_DIRECTORIES = (
    ".github",
    "configs",
    "data",
    "docs",
    "examples",
    "mbmmc",
    "scripts",
    "tests",
    "tools",
)

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "outputs",
    "runs",
    "site-packages",
    "venv",
}

FORBIDDEN_FILE_NAME = re.compile(r"(?:^|_)v[0-9]+(?:_|\.|$)", re.IGNORECASE)
FORBIDDEN_MODEL_TEXT = re.compile(
    r"\b(?:RF|crossNN|MPCNet)[_ .-]*v[0-9]+(?:[._][0-9]+)*\b",
    re.IGNORECASE,
)


def _is_excluded(path: Path) -> bool:
    try:
        relative_parts = path.relative_to(PROJECT_ROOT).parts
    except ValueError:
        return True
    return any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_parts)


def _iter_controlled_files() -> Iterator[Path]:
    # Repository-root files such as train_rf.py, README.md, and pyproject.toml.
    for path in PROJECT_ROOT.iterdir():
        if path.is_file():
            yield path

    # Source, configuration, documentation, examples, and tests maintained here.
    for directory_name in CONTROLLED_DIRECTORIES:
        directory = PROJECT_ROOT / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and not _is_excluded(path):
                yield path


def test_no_historical_model_version_names() -> None:
    for path in _iter_controlled_files():
        relative_path = path.relative_to(PROJECT_ROOT)

        assert not FORBIDDEN_FILE_NAME.search(path.name), relative_path

        if path.suffix.lower() != ".py":
            continue

        text = path.read_text(encoding="utf-8")
        assert not FORBIDDEN_MODEL_TEXT.search(text), relative_path
