# Input and output specification

## Legacy feature-by-sample matrix

```csv
probe_id,chr,pos,S001,S002,S003
cg00000001,chr1,100000,0.82,0.14,NA
cg00000002,chr1,100010,0.18,0.77,0.55
```

Requirements:

- columns 1–3 contain feature identifier, chromosome, and genomic position;
- columns 4 onward contain unique sample identifiers;
- beta values should be in `[0, 1]`;
- missing values may be `NA`, `NaN`, `N/A`, `null`, or empty.

## Metadata

Required columns:

| Column | Meaning |
|---|---|
| `Sample` | Sample identifier matching the matrix |
| `Types` | Tumor class label |

Optional columns may include `Platform`, `Material`, `Patient`, `Study`, `Batch`,
`SplitGroup`, `LatentID`, `RecommendedTrainingWeight`, and `TumorFraction`.

## MPCNet sample-by-feature input

Matrix:

```csv
sample_id,chr1_100000,chr1_100010
S001,0.82,NA
S002,0.14,0.77
```

Labels:

```csv
sample_id,label
S001,TumorA
S002,TumorB
```

## Main outputs

### RF

- `final_model_bundle.joblib`
- `final_best_params.json`
- `config.json`
- `features.txt`
- CV metrics, summaries, plots, and optional sample-weight audits

### crossNN

- `final_crossnn_bundle.pt`
- `final_params.json`
- `preprocess.json`
- final CV metrics and plots
- `final_train_loss.csv`
- `features.txt`

### MPCNet

- `mpcnet_model.pt`
- `mpcnet_bundle.json`
- `model_features_mpcnet.txt`
- `mpcnet_internal_metrics.json`
- internal validation predictions
- calibration summaries
- training and optional final-refit histories

Preserve each model artifact with its configuration, feature list, class mapping,
code commit, environment, and data checksum.


## Independent simulation outputs

The cross-platform simulator has a separate input/output contract documented in
`docs/SIMULATION.md`. Its primary products are a synthetic beta matrix, sample
metadata, full parameter table, source map, source-usage audit, validation report,
and JSON manifest.
