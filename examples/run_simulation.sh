#!/usr/bin/env bash
set -euo pipefail

python scripts/simulation/generate_cross_platform_in_silico_beta.py \
  --beta_csv examples/data/simulation/reference_beta.csv \
  --meta examples/data/simulation/metadata.tsv \
  --out_beta_csv outputs/toy_simulation/simulated_beta.csv \
  --out_meta outputs/toy_simulation/simulated_meta.tsv \
  --out_params outputs/toy_simulation/simulation_params.tsv \
  --out_manifest outputs/toy_simulation/simulation_manifest.json \
  --out_source_map outputs/toy_simulation/source_map.tsv \
  --out_source_usage_audit outputs/toy_simulation/source_usage_audit.tsv \
  --out_validation_report outputs/toy_simulation/validation_report.json \
  --source_split_value sim_train \
  --generator_split sim_train \
  --platforms TAPS \
  --platform_design paired_latent \
  --n_per_type_per_platform 3 \
  --n_boundary_per_type_per_platform 1 \
  --n_control_per_platform 4 \
  --min_background_sources 1 \
  --min_anchor_sources 1 \
  --min_tumor_sources_per_type 1 \
  --background_min_families_per_latent 1 \
  --background_max_families_per_latent 2 \
  --background_max_sources_per_latent 3 \
  --tumor_max_sources_per_latent 2 \
  --measurement_model gaussian \
  --missing_mode none \
  --chunk_rows 30 \
  --progress_every 0 \
  --seed 23
