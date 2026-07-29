# Third-party methods and notices

## crossNN methodology

The MBMMC crossNN implementation was independently developed with reference to
the methodology described in:

Dongsheng Yuan, Robin Jugas, Petra Pokorna, et al.  
“crossNN is an explainable framework for cross-platform DNA methylation-based
classification of tumors.”  
*Nature Cancer* 6, 1283–1294 (2025).  
DOI: 10.1038/s43018-025-00976-5

Official project repository:

```text
https://gitlab.com/euskirchen-lab/crossNN
```

The reviewed MBMMC files do not include copies of the official repository's
`training.py` or `NN_model.py`. The implementations share method-level concepts
reported in the paper, including a bias-free single linear layer, ternary
methylation encoding, and random feature masking. MBMMC uses its own data-loading,
feature-selection, grouped cross-validation, weighting, tuning, evaluation,
plotting, and artifact-output implementation.

This notice provides scholarly attribution. It does not state that the original
crossNN authors endorse MBMMC, and it does not apply the MBMMC license to the
original crossNN project.

See `docs/CROSSNN_METHOD_PROVENANCE.md` for the code-level review.
