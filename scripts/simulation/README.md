# Cross-platform in silico beta simulation

The simulator is an independent data-generation workflow. It is not imported by the
RF, crossNN, or MPCNet training implementations.

Public entry point:

```bash
python scripts/simulation/generate_cross_platform_in_silico_beta.py --help
```

Its main purpose is to generate source-audited synthetic methylation beta matrices for
cross-platform development and stress testing across TAPS, WGBS, ONT, and array-like
observations.

Scientific details, inputs, outputs, and a small example are documented in
`docs/SIMULATION.md`.
