# Minimal examples

Generate synthetic data:

```bash
python examples/make_toy_data.py
```

Run one model or all models:

```bash
bash examples/run_rf.sh
bash examples/run_crossnn.sh
bash examples/run_mpcnet.sh
bash examples/run_all_smoke.sh
```

The generated data are synthetic and verify software wiring only. They are not
scientifically meaningful benchmarks.


## Simulation command template

`run_simulation_template.sh` documents the recommended output set for the standalone
cross-platform simulator. Replace both input paths before execution. The template is
not included in `run_all_smoke.sh` because a formal simulator input requires explicit
source-governance metadata.


## Simulation workflow

```bash
python examples/make_simulation_reference.py
bash examples/run_simulation.sh
```

This workflow is independent of the three model-training smoke examples.
