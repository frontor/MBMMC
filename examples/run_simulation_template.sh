#!/usr/bin/env bash
set -euo pipefail

# Replace the two input paths before running.
python scripts/simulation/generate_cross_platform_in_silico_beta.py \
  --beta_csv /absolute/path/reference_beta.csv \
  --meta /absolute/path/source_metadata.tsv \
  --out_beta_csv outputs/simulation/simulated_beta.csv \
  --out_meta outputs/simulation/simulated_meta.tsv \
  --out_params outputs/simulation/simulated_params.tsv \
  --out_manifest outputs/simulation/simulation_manifest.json \
  --out_source_map outputs/simulation/source_map.tsv \
  --out_source_usage_audit outputs/simulation/source_usage_audit.tsv \
  --out_validation_report outputs/simulation/validation_report.json \
  --platforms TAPS,WGBS,ONT \
  --platform_design paired_latent \
  --source_split_value sim_train \
  --generator_split sim_train \
  --seed 42
