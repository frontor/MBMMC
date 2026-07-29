from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PLACEHOLDER = re.compile(r"REPLACE_WITH_[A-Z0-9_]+")
FORBIDDEN_SUFFIXES = {
    ".bam", ".cram", ".fastq", ".gz", ".joblib", ".pkl", ".pickle",
    ".pt", ".pth", ".ckpt", ".onnx", ".pyc"
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "outputs", "runs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the repository for publication placeholders, large files, caches, and model/data artifacts."
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--max-mb", type=float, default=10.0, help="Warn for files larger than this size.")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Do not fail when REPLACE_WITH_* placeholders remain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    issues: list[str] = []

    license_path = root / "LICENSE"
    if not license_path.is_file():
        issues.append("missing LICENSE")
    else:
        license_text = license_path.read_text(encoding="utf-8")
        if "PolyForm Noncommercial License 1.0.0" not in license_text:
            issues.append("LICENSE is not PolyForm Noncommercial License 1.0.0")
        found = sorted(set(PLACEHOLDER.findall(license_text)))
        if found and not args.allow_placeholders:
            issues.append(f"publication placeholder(s) {found}: LICENSE")

    required_provenance = [
        root / "THIRD_PARTY_NOTICES.md",
        root / "docs" / "LICENSE_POLICY.md",
        root / "docs" / "CROSSNN_METHOD_PROVENANCE.md",
    ]
    for required in required_provenance:
        if not required.is_file():
            issues.append(f"missing required provenance file: {required.relative_to(root)}")

    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > args.max_mb:
            issues.append(f"large file ({size_mb:.1f} MB): {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"data/model/cache artifact: {rel}")
        if path.suffix.lower() in {".md", ".cff", ".toml", ".json", ".yaml", ".yml", ".txt"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            found = sorted(set(PLACEHOLDER.findall(text)))
            if found and not args.allow_placeholders:
                issues.append(f"publication placeholder(s) {found}: {rel}")

    if issues:
        print("Release check failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    print("Release check passed.")


if __name__ == "__main__":
    main()
