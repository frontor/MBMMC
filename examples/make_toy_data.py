from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic methylation data in both supported training formats."
    )
    parser.add_argument("--outdir", default="examples/data", help="Output directory.")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--features", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 4:
        raise ValueError("--samples-per-class must be at least 4.")
    if args.features < 30:
        raise ValueError("--features must be at least 30.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    classes = ["TumorA", "TumorB", "TumorC"]
    labels = np.repeat(classes, args.samples_per_class)
    n_samples = len(labels)
    n_features = args.features
    samples = [f"S{i:03d}" for i in range(n_samples)]

    x = rng.beta(2.0, 5.0, size=(n_samples, n_features))
    block = max(8, n_features // 10)
    for class_index, class_name in enumerate(classes):
        start = class_index * block
        stop = min(start + block, n_features)
        x[labels == class_name, start:stop] += 0.45
    x = np.clip(x, 0.0, 1.0)
    x[rng.random(x.shape) < 0.08] = np.nan

    chromosomes = np.array([f"chr{1 + (i % 3)}" for i in range(n_features)])
    positions = np.arange(100_000, 100_000 + 10 * n_features, 10)
    feature_names = [f"{chrom}_{pos}" for chrom, pos in zip(chromosomes, positions)]

    # Legacy feature-by-sample reference format.
    legacy = pd.DataFrame(
        {
            "probe_id": [f"cg{i:08d}" for i in range(n_features)],
            "chr": chromosomes,
            "pos": positions,
        }
    )
    for sample_index, sample in enumerate(samples):
        legacy[sample] = x[sample_index]
    legacy.to_csv(outdir / "legacy_reference.csv", index=False, na_rep="NA")

    meta = pd.DataFrame(
        {
            "Sample": samples,
            "Types": labels,
            "Platform": ["ARRAY"] * n_samples,
            "Material": ["Frozen"] * n_samples,
            "Patient": [f"P{i:03d}" for i in range(n_samples)],
        }
    )
    meta.to_csv(outdir / "meta.tsv", sep="\t", index=False)

    # Sample-by-feature matrix + labels format used by MPCNet.
    matrix = pd.DataFrame(x, columns=feature_names)
    matrix.insert(0, "sample_id", samples)
    matrix.to_csv(outdir / "sample_matrix.csv", index=False, na_rep="NA")
    pd.DataFrame({"sample_id": samples, "label": labels}).to_csv(
        outdir / "labels.csv", index=False
    )

    print(f"Wrote toy data to {outdir.resolve()}")


if __name__ == "__main__":
    main()
