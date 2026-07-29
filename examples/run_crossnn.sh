#!/usr/bin/env bash
set -euo pipefail
python -m mbmmc.train_crossnn \
  --ref_csv examples/data/legacy_reference.csv \
  --meta examples/data/meta.tsv \
  --outdir outputs/toy_crossnn \
  --preset paper \
  --beta_threshold 0.60 \
  --equal_mode negative \
  --feature_select variance_topk \
  --topk 40 \
  --mask_keep_fraction 0.50 \
  --epochs 2 \
  --cv_folds 2 \
  --random_state 17 \
  --device cpu
