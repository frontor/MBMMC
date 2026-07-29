# Packaging validation

Validation date: 2026-07-29

## Completed checks

- Python syntax compilation for model code, tools, simulation workflow, examples,
  tests, and root entry points: passed.
- RF command-line help: passed.
- crossNN command-line help: passed.
- MPCNet command-line help: passed.
- Cross-platform simulator command-line help: passed.
- PolyForm Noncommercial license text and SPDX metadata consistency tests: passed.
- crossNN attribution and provenance-file tests: passed.
- Stable public-name test: passed.
- Simulation public-interface test: passed.
- macOS/Linux publishing-script shell syntax: passed.
- `pyproject.toml` parsing and console-entry consistency: passed.
- Archive-content and forbidden-artifact audit: passed during final packaging.

## crossNN provenance conclusion

The technical review supports describing MBMMC crossNN as an independent
implementation informed by the 2025 crossNN methodology. Shared elements are
method-level concepts such as the bias-free linear layer, ternary encoding, and
random masking. The reviewed MBMMC implementation has different custom identifiers,
data interfaces, masking implementation, validation, weighting, tuning, evaluation,
and output organization.

This is a technical repository review, not a legal opinion. See:

```text
THIRD_PARTY_NOTICES.md
docs/CROSSNN_METHOD_PROVENANCE.md
```

## Required checks before the public release


```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
python -m compileall -q mbmmc tools scripts examples tests
pytest -q
python examples/make_toy_data.py
bash examples/run_all_smoke.sh
python examples/make_simulation_reference.py
bash examples/run_simulation.sh
python -m tools.check_release
```

The full production training workflows must be repeated in the exact environment
reported by the manuscript. Synthetic examples verify software wiring only and are
not biological benchmarks or independent validation.


## Local virtual-environment regression check

`tests/test_stable_names.py` was tested with the following third-party-style path
present inside a temporary local virtual environment:

```text
.venv/lib/python3.11/site-packages/nvidia/cu13/include/cublas_v2.h
```

The test correctly ignored this dependency file. The complete test suite then passed:

```text
8 passed
```
