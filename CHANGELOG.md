# Changelog

## 1.0.0 — 2026-07-28

- Established the stable public name MBMMC.
- Standardized entry points as `train_rf.py`, `train_crossnn.py`, and `train_mpcnet.py`.
- Removed historical model-version identifiers and compatibility filenames.
- Retained only the three training workflows and direct dependencies.
- Added configurations, examples, tests, scientific documentation, citation metadata,
  and continuous integration.
- Added the standalone cross-platform in silico beta simulator under `scripts/simulation/` with a stable public filename, documentation, and CLI entry.


- Added an independent cross-platform in silico beta simulation workflow under
  `scripts/simulation/`.
- Added simulation input/output documentation, synthetic reference generation,
  a runnable example, and CLI/example tests.


- Replaced the BSD-3-Clause template with PolyForm Noncommercial License 1.0.0.
- Added a license-change policy explaining treatment of future and historical releases.
- Added crossNN scholarly attribution, third-party notices, and a code-provenance review.
- Added explicit noncommercial and research-use-only statements to repository metadata.

- Fixed `tests/test_stable_names.py` so it scans only repository-controlled files and ignores local virtual environments, installed dependencies, build products, caches, and generated outputs.
