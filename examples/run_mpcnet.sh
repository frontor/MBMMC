#!/usr/bin/env bash
set -euo pipefail
python -m mbmmc.train_mpcnet \
  --matrix examples/data/sample_matrix.csv \
  --labels examples/data/labels.csv \
  --outdir outputs/toy_mpcnet \
  --feature_select variance_topk \
  --topk 40 \
  --input_compression dense \
  --value_mode raw \
  --epochs 2 \
  --patience 1 \
  --hidden 16 \
  --depth 1 \
  --batch_size 8 \
  --temperature_scaling off \
  --final_refit none \
  --seed 17 \
  --device cpu
