# Code provenance and cleaning record

The public package was derived from the supplied working archive.

Retained:

- the audited Random Forest implementation with top-k feature selection before masking;
- the leakage-controlled crossNN implementation;
- the sparse/hybrid MPCNet implementation;
- only utilities imported by these trainers.

Stable public entry points:

- `train_rf.py`
- `train_crossnn.py`
- `train_mpcnet.py`

Historical file suffixes, historical model labels, compatibility aliases, backup files,
bytecode caches, patch archives, and unrelated workflows were excluded. Algorithmic
training procedures were retained while public names, comments, CLI text, metadata,
and output names were normalized.


## Independent simulator

The source-audited cross-platform in silico beta generator supplied separately was
added under the stable public path:

```text
scripts/simulation/generate_cross_platform_in_silico_beta.py
```

Only public packaging metadata, historical version wording, the filename, the CLI
version label, and the manifest software identifier were normalized. The simulation
logic was otherwise retained.

Original uploaded source SHA-256:

```text
0810496952113d2f4399697053b2e15b434b7a7e69c37ff507126036042b49ef
```
