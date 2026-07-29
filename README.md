# MBMMC

MBMMC provides publication-oriented training code for three methylation-based tumor
classification models:

1. **Random Forest (RF)** — `train_rf.py`
2. **crossNN** — `train_crossnn.py`
3. **MPCNet** — `train_mpcnet.py`

The repository contains the code required to train these models, their direct
dependencies, reproducible parameter configurations, examples, tests, and scientific
documentation. A separate cross-platform in silico beta simulator is provided under
`scripts/simulation/`; it is not part of model training.

## Installation

Conda:

```bash
conda env create -f environment.yml
conda activate mbmmc
pip install -e .
```

Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Command-line help

Direct entry points:

```bash
python train_rf.py --help
python train_crossnn.py --help
python train_mpcnet.py --help
python scripts/simulation/generate_cross_platform_in_silico_beta.py --help
```

After installation:

```bash
mbmmc-train-rf --help
mbmmc-train-crossnn --help
mbmmc-train-mpcnet --help
mbmmc-run-grid --help
mbmmc-generate-simulation --help
```

## Synthetic model-training examples

```bash
python examples/make_toy_data.py
bash examples/run_rf.sh
bash examples/run_crossnn.sh
bash examples/run_mpcnet.sh
```

Synthetic examples verify software wiring only and must not be reported as scientific
performance.

## Cross-platform simulation

Generate the small simulator reference data and run the smoke example:

```bash
python examples/make_simulation_reference.py
bash examples/run_simulation.sh
```

The simulator is a standalone, source-audited data-generation workflow. See
`docs/SIMULATION.md` for input/output contracts, source-governance requirements, and
formal-use recommendations.

## Documentation

- `docs/INPUT_OUTPUT.md`
- `docs/PARAMETERS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/SIMULATION.md`
- `docs/LICENSE_POLICY.md`
- `docs/CROSSNN_METHOD_PROVENANCE.md`
- `THIRD_PARTY_NOTICES.md`
- `README_zh.md`

## Parameter candidates

Set the real-data paths and preview commands without training:

```bash
export REF_CSV=/absolute/path/to/reference.csv
export META=/absolute/path/to/meta.tsv
export DEVICE=cpu

python -m tools.run_grid \
  --config configs/publication_candidates.yaml \
  --dry-run \
  --limit 2
```

Run one model family:

```bash
python -m tools.run_grid \
  --config configs/publication_candidates.yaml \
  --model rf \
  --resume
```

Each candidate gets a separate output directory containing its command, resolved
arguments, and training log. The grid root contains `run_manifest.jsonl`.

## Testing

```bash
pip install -r requirements-dev.txt
python -m compileall -q mbmmc tools scripts examples tests
pytest -q
```

## Data governance

Patient-level data, generated outputs, and model binaries are ignored by default.
Do not upload protected health information, linkable identifiers, credentials, or
controlled-access molecular data.

## Citation

Complete `CITATION.cff` before public release. The paper should identify the repository,
release number, commit hash, archived DOI, exact configuration, and data version.

The MBMMC crossNN implementation is independently developed with reference to the
method reported by Yuan et al. (2025). See `THIRD_PARTY_NOTICES.md` and
`docs/CROSSNN_METHOD_PROVENANCE.md`.

## License

MBMMC is **source-available for noncommercial use** under the
PolyForm Noncommercial License 1.0.0
(`PolyForm-Noncommercial-1.0.0`).

Commercial use is not granted by this repository. See `LICENSE` and
`docs/LICENSE_POLICY.md`.
