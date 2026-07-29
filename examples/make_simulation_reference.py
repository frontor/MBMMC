from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small source-governed reference dataset for the simulator example."
    )
    parser.add_argument("--outdir", default="examples/data/simulation")
    parser.add_argument("--features", type=int, default=60)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.features < 20:
        raise ValueError("--features must be at least 20.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tumor_classes = ["TumorA", "TumorB"]
    rows = []
    samples = []

    for tumor_index, tumor_type in enumerate(tumor_classes):
        for donor_index in range(3):
            sample = f"{tumor_type}_D{donor_index + 1}"
            samples.append(sample)
            rows.append(
                {
                    "Sample": sample,
                    "Types": tumor_type,
                    "SourceSplit": "sim_train",
                    "DonorID": f"{tumor_type}_DONOR_{donor_index + 1}",
                    "ReplicateGroup": f"{tumor_type}_REP_{donor_index + 1}",
                    "SourceDataset": "TOY_TUMOR",
                    "BackgroundRole": "tumor_source",
                    "ControlFamily": "",
                    "ControlSubtype": "",
                    "BiologicalState": "tumor",
                    "IncludeForSimulation": "YES",
                    "QCStatus": "PASS",
                }
            )

    control_specs = [
        ("CONTROL_PLASMA_1", "CFDNA_HEALTHY", "plasma_anchor"),
        ("CONTROL_PLASMA_2", "CFDNA_HEALTHY", "plasma_anchor"),
        ("CONTROL_IMMUNE_1", "BLOOD_IMMUNE", "background_component"),
        ("CONTROL_IMMUNE_2", "BLOOD_IMMUNE", "background_component"),
    ]
    for index, (sample, family, role) in enumerate(control_specs, start=1):
        samples.append(sample)
        rows.append(
            {
                "Sample": sample,
                "Types": "CONTROL",
                "SourceSplit": "sim_train",
                "DonorID": f"CONTROL_DONOR_{index}",
                "ReplicateGroup": f"CONTROL_REP_{index}",
                "SourceDataset": "TOY_CONTROL",
                "BackgroundRole": role,
                "ControlFamily": family,
                "ControlSubtype": family,
                "BiologicalState": "healthy",
                "IncludeForSimulation": "YES",
                "QCStatus": "PASS",
            }
        )

    n_features = args.features
    chromosome = np.array([f"chr{1 + i % 3}" for i in range(n_features)])
    position = np.arange(200_000, 200_000 + 10 * n_features, 10)
    beta = rng.beta(2.0, 6.0, size=(n_features, len(samples)))

    block = max(6, n_features // 8)
    for tumor_index, tumor_type in enumerate(tumor_classes):
        tumor_columns = [
            i for i, sample in enumerate(samples) if sample.startswith(tumor_type)
        ]
        start = tumor_index * block
        stop = min(start + block, n_features)
        beta[start:stop, tumor_columns] += 0.50

    beta = np.clip(beta, 0.0, 1.0)
    beta[rng.random(beta.shape) < 0.03] = np.nan

    reference = pd.DataFrame(
        {
            "probe_id": [f"cg{i:08d}" for i in range(n_features)],
            "chr": chromosome,
            "pos": position,
        }
    )
    for column_index, sample in enumerate(samples):
        reference[sample] = beta[:, column_index]

    reference.to_csv(outdir / "reference_beta.csv", index=False, na_rep="NA")
    pd.DataFrame(rows).to_csv(outdir / "metadata.tsv", sep="\t", index=False)

    print(f"Wrote simulation example inputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
