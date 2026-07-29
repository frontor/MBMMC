#!/usr/bin/env bash
set -euo pipefail
python -m mbmmc.train_rf \
  --ref_csv examples/data/legacy_reference.csv \
  --meta examples/data/meta.tsv \
  --outdir outputs/toy_rf \
  --model rf \
  --beta_threshold 0.60 \
  --equal_mode negative \
  --feature_select variance_topk \
  --topk 40 \
  --corr_filter 0 \
  --mask_alg mcar \
  --mask_rate 0.10 \
  --mask_on train \
  --cv_mode none \
  --inner_folds 2 \
  --search_iters 1 \
  --random_state 17 \
  --n_jobs 1
