# Parameter guide

Complete options:

```bash
python train_rf.py --help
python train_crossnn.py --help
python train_mpcnet.py --help
```

## RF

Key groups include beta thresholding, fold-local feature selection, correlation
filtering, masking, nested cross-validation, grouping, and sample weighting.

## crossNN

Key groups include fixed or tuned presets, observation keep fraction, optimization,
early stopping, group-aware CV, and feature-selection scope.

## MPCNet

Key groups include feature policy, input compression, value representation, sparse-mask
augmentation, architecture, optimization, calibration, no-call thresholds, and
group-aware validation.

## Model-selection policy

Define before accessing the independent test set:

1. candidate set;
2. primary endpoint;
3. tie-breakers;
4. development cohorts;
5. locked-model rule;
6. independent-test access rule.
