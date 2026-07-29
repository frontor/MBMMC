# Model card: MBMMC

## Model details

- Models: Random Forest, crossNN, MPCNet
- Developers: Guangjian Liu, Chinese Institute for Brain Research, Beijing.
- Repository: https://github.com/frontor/MBMMC
- Associated paper: 
- Software archive DOI: 

## Intended use

research-use-only.

## Inputs and outputs

Inputs are DNA methylation beta values keyed by genomic location. Outputs are
classifiers for tumor classification.

## Training and evaluation data

methylation array and simulated data.

## Metrics


## Limitations

Document domain shift across assay, platform, material, laboratory, ancestry, tumor
fraction, specimen quality, and missingness. State unsupported tumor types.

## Safety and ethics

Do not use the software as a standalone diagnostic system without appropriate
validation, governance, quality controls, and expert review.


## License and access conditions

MBMMC source code is available for noncommercial purposes under the PolyForm
Noncommercial License 1.0.0. The public repository does not grant commercial
use rights.

The software is for research use only. It is not intended to serve as a standalone
diagnostic device or as the sole basis for clinical decision-making.

## Method provenance

The MBMMC crossNN implementation was independently developed with reference to
the crossNN methodology reported by Yuan et al. (Nature Cancer, 2025;
doi:10.1038/s43018-025-00976-5). It is not an official release of the original
crossNN authors. See `THIRD_PARTY_NOTICES.md` and
`docs/CROSSNN_METHOD_PROVENANCE.md`.
