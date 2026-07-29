# Cross-platform simulation workflow

## Scope

`generate_cross_platform_in_silico_beta.py` generates in silico methylation beta matrices from
a reference feature-by-sample beta matrix and sample metadata. It is separate from the
three model-training programs and is placed under `scripts/simulation/`.

The simulator supports:

- source-split and donor/replicate leakage checks;
- tumor-anchor or Dirichlet source mixing;
- hierarchical control/background composition;
- paired or independent latent profiles across platforms;
- TAPS, WGBS, ONT, and array-like observation presets;
- Gaussian, binomial, or beta-binomial measurement;
- explicit missingness and platform/batch effects;
- source maps, parameter tables, manifests, validation reports, and source-usage audits.

## Command-line help

```bash
python scripts/simulation/generate_cross_platform_in_silico_beta.py --help
# or, after pip install -e .:
mbmmc-generate-simulation --help
```

## Required inputs

### Reference beta matrix

CSV containing three identifier columns followed by reference sample columns:

```csv
probe_id,chr,pos,TUMOR_A_1,TUMOR_A_2,CONTROL_1
cg00000001,chr1,100000,0.81,0.75,0.12
```

### Metadata

At minimum:

```text
Sample  Types
TUMOR_A_1  TumorA
TUMOR_A_2  TumorA
CONTROL_1  CONTROL
```

For formal source governance, the following fields are strongly recommended:

- `SourceSplit`
- `DonorID`
- `ReplicateGroup`
- `SourceDataset`
- `BackgroundRole`
- `ControlFamily`
- `ControlSubtype`
- `BiologicalState`
- `IncludeForSimulation`
- `QCStatus`

Tumor sources should use `BackgroundRole=tumor_source`. Eligible cfDNA background
sources commonly use `plasma_anchor` or `background_component`.

## Main outputs

| Argument | Output |
|---|---|
| `--out_beta_csv` | Synthetic feature-by-sample beta matrix |
| `--out_meta` | Training-oriented sample metadata |
| `--out_params` | Complete per-sample simulation parameters |
| `--out_manifest` | JSON manifest containing inputs, outputs, presets, arguments, and audit summaries |
| `--out_source_map` | Exact latent-profile source samples and weights |
| `--out_source_usage_audit` | Per-source usage audit |
| `--out_validation_report` | Metadata-governance and source-pool validation report |

## Minimal synthetic example

Generate a small reference dataset:

```bash
python examples/make_simulation_reference.py
```

Run the simulator:

```bash
bash examples/run_simulation.sh
```

The example is intended only to verify the workflow. Its generated values are not a
validated biological benchmark.

## Formal-use recommendations

- Prepare donor-disjoint `SourceSplit` values before simulation.
- Generate development, lock-test, and final-augmentation pools separately.
- Preserve `out_source_map`, `out_manifest`, `out_validation_report`, and input
  SHA-256 checksums.
- Do not use simulated data as a substitute for independent real-platform validation.
- State clearly in the manuscript which cohorts are real and which are simulated.
