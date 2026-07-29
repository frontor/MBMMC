# crossNN method and code provenance review

## Scope

This review compares the MBMMC crossNN implementation with the public crossNN
method paper and the public official repository files `training.py` and
`NN_model.py`, as reviewed on 2026-07-29.

MBMMC files reviewed:

```text
mbmmc/train_crossnn.py
mbmmc/utils/crossnn.py
```

Official project:

```text
https://gitlab.com/euskirchen-lab/crossNN
```

Paper:

```text
Yuan D, Jugas R, Pokorna P, et al.
crossNN is an explainable framework for cross-platform DNA
methylation-based classification of tumors.
Nature Cancer. 2025;6:1283–1294.
doi:10.1038/s43018-025-00976-5
```

## Shared method-level concepts

The following similarities are expected because they are central to the published
crossNN method:

- a single fully connected linear output layer without bias;
- beta-value binarization around the published threshold;
- methylated, unmethylated, and missing/masked ternary representation;
- random masking during training;
- cross-entropy optimization with Adam;
- multiclass probability output.

These are scientific-method concepts and common machine-learning operations. Their
presence alone does not establish source-code copying.

## Code-level differences

### Public official implementation

The reviewed official training script:

- reads a fixed HDF5 data contract;
- defines `DNN`, `read_data`, `preprocessing`, `mask_input`,
  `model_training`, and `parse_arguments`;
- removes constant probes using the fixed reference sample count;
- masks a sampled set of feature columns across a batch;
- performs one direct training run with a PyTorch `DataLoader`;
- saves a state dictionary and a pickle bundle.

The reviewed inference module defines `NN_Model` and `NN_classifier` and contains
bedMethyl-specific inference handling.

### MBMMC implementation

The MBMMC implementation:

- reads a feature-by-sample CSV and a separate metadata table;
- defines `CrossNNLinear`, `make_mask`, `train_one_fold`, `cv_crossnn`,
  `train_final_model`, `random_search_tune`, and additional helpers;
- supports fold-local feature selection to reduce leakage;
- supports stratified, grouped, and leave-one-group-out validation;
- uses a Bernoulli mask generated per tensor element rather than the official
  shared feature-index masking implementation;
- supports class weighting and optional per-sample weighting;
- supports early stopping, parameter search, multiple evaluation metrics,
  plots, audit files, and structured model bundles;
- does not include the official HDF5 loader, official pickle bundle layout,
  or bedMethyl inference class.

## Identifier and structure review

No distinctive custom class or function names from the reviewed official files are
used by the MBMMC implementation. Shared names are limited to ordinary Python or
PyTorch conventions such as `__init__`, `forward`, `CrossEntropyLoss`, `Adam`, and
`LabelEncoder`.

A normalized token-sequence comparison found only short common runs associated with
the minimal bias-free linear-layer declaration and its `forward` method. No long,
distinctive source-code sequence was identified in this review.

## Conclusion

The available evidence supports describing MBMMC crossNN as an **independent
implementation informed by the published crossNN methodology**, rather than a
redistribution or modification of the official crossNN source files.

Accordingly:

- the crossNN paper and official project are cited for scholarly attribution;
- the original crossNN authors are not listed as MBMMC copyright holders solely
  because the method was referenced;
- the MBMMC repository license applies only to MBMMC code and content owned or
  licensable by the MBMMC copyright holder;
- the original crossNN project remains governed by whatever rights and terms apply
  to that project.

## Limit of this review

This is a repository-level technical provenance review, not a legal opinion. If any
earlier development version copied or translated code that is not present in the
reviewed MBMMC files, that history should be disclosed and assessed separately.
