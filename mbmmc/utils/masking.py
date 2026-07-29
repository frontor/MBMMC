from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Masker(BaseEstimator, TransformerMixin):
    """Simulate missingness by setting entries to 0.

    This transformer expects input to already be binarized into {-1, 0, 1},
    where 0 represents missing.

    Parameters
    ----------
    mask_alg:
      - mcar: uniform random masking of entries (MCAR)
      - mcar_beta_rate: per-sample MCAR where each sample's mask_rate is drawn from a Beta distribution
      - weighted_mcar: per-sample masking where masked features are sampled with feature weights
                      (by default inferred from training-set missingness rate per feature)
      - chrom_block: mask contiguous blocks per chromosome (requires chr/pos arrays)
      - predefined: mask a predefined set of features (mask_feature_indices)

    mask_rate:
      Fraction of entries/features masked to 0 during training (and optionally validation/pred).

    mask_on:
      - train: apply mask only during fit_transform (recommended)
      - both: apply mask during fit_transform and transform (stress-test; makes prediction stochastic)
      - none: no masking

    Notes
    -----
    - For chrom_block, blocks are defined by sorted positions within each chromosome.
    - For weighted_mcar, masking is implemented by sampling ~mask_rate * n_features features per sample.
    """

    def __init__(
        self,
        mask_alg: str = "mcar",
        mask_rate: float = 0.9,
        mask_on: str = "train",
        random_state: int = 42,
        chr_arr: np.ndarray | None = None,
        pos_arr: np.ndarray | None = None,
        mask_feature_indices: np.ndarray | None = None,
        block_size_bp: int = 500_000,
        # mcar_beta_rate params
        beta_kappa: float = 20.0,
        beta_alpha: float = -1.0,
        beta_beta: float = -1.0,
        # weighted_mcar params
        feature_weights: np.ndarray | None = None,
        weight_eps: float = 1e-3,
        weight_power: float = 1.0,
    ):
        self.mask_alg = mask_alg
        self.mask_rate = float(mask_rate)
        self.mask_on = mask_on
        self.random_state = int(random_state)
        self.chr_arr = chr_arr
        self.pos_arr = pos_arr
        self.mask_feature_indices = mask_feature_indices
        self.block_size_bp = int(block_size_bp)

        self.beta_kappa = float(beta_kappa)
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)

        self.feature_weights = feature_weights
        self.weight_eps = float(weight_eps)
        self.weight_power = float(weight_power)

        self._rng = None
        self._fitted_feature_weights = None

    def fit(self, X, y=None):
        self._rng = np.random.default_rng(self.random_state)
        X = np.asarray(X)

        if self.mask_alg == "weighted_mcar":
            if self.feature_weights is not None:
                w = np.asarray(self.feature_weights, dtype=np.float64)
                if w.ndim != 1 or w.shape[0] != X.shape[1]:
                    raise ValueError("feature_weights must be 1D and match n_features")
            else:
                # infer weights from training missingness rate per feature
                miss_rate = (X == 0).astype(np.float32).mean(axis=0)
                w = np.asarray(miss_rate, dtype=np.float64)

            w = np.maximum(w, 0.0) + float(self.weight_eps)
            if self.weight_power != 1.0:
                w = np.power(w, float(self.weight_power))
            s = float(w.sum())
            if s <= 0:
                w = np.ones_like(w) / float(w.size)
            else:
                w = w / s
            self._fitted_feature_weights = w.astype(np.float64)
        else:
            self._fitted_feature_weights = None

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y) ##bug修复增加
        X = np.asarray(X)
        if self.mask_on in ("none", None) or self.mask_rate <= 0:
            return X
        return self._apply_mask(X, training=True)

    def transform(self, X):
        X = np.asarray(X)
        if self.mask_on in ("none", None) or self.mask_rate <= 0:
            return X
        if self.mask_on == "train":
            return X
        return self._apply_mask(X, training=False)

    def _apply_mask(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        alg = self.mask_alg
        rng = self._rng if self._rng is not None else np.random.default_rng(self.random_state)
        X2 = X.copy()

        if alg == "mcar":
            m = rng.random(X2.shape) < self.mask_rate
            X2[m] = 0
            return X2

        if alg == "mcar_beta_rate":
            mu = float(self.mask_rate)
            mu = min(max(mu, 1e-6), 1.0 - 1e-6)

            if self.beta_alpha > 0 and self.beta_beta > 0:
                a, b = float(self.beta_alpha), float(self.beta_beta)
            else:
                kappa = max(float(self.beta_kappa), 1e-3)
                a, b = mu * kappa, (1.0 - mu) * kappa

            rates = rng.beta(a, b, size=X2.shape[0]).astype(np.float32)
            for i in range(X2.shape[0]):
                m = rng.random(X2.shape[1]) < float(rates[i])
                X2[i, m] = 0
            return X2

        if alg == "weighted_mcar":
            w = self._fitted_feature_weights
            if w is None:
                # fit() not called; infer uniform
                w = np.ones(X2.shape[1], dtype=np.float64) / float(X2.shape[1])

            n_feat = X2.shape[1]
            k = int(round(self.mask_rate * n_feat))
            k = min(max(k, 0), n_feat)

            if k == 0:
                return X2

            for i in range(X2.shape[0]):
                cols = rng.choice(n_feat, size=k, replace=False, p=w)
                X2[i, cols] = 0
            return X2

        if alg == "chrom_block":
            if self.chr_arr is None or self.pos_arr is None:
                raise ValueError("chrom_block requires chr_arr and pos_arr")
            chr_arr = np.asarray(self.chr_arr)
            pos_arr = np.asarray(self.pos_arr)
            # group indices by chromosome
            for chrom in np.unique(chr_arr):
                cols = np.where(chr_arr == chrom)[0]
                if cols.size == 0:
                    continue
                # sort by position
                order = cols[np.argsort(pos_arr[cols])]
                n = order.size
                # compute how many bp to mask approximately based on mask_rate
                # approximate by masking a fraction of features within the chromosome
                k_chrom = int(round(self.mask_rate * n))
                k_chrom = min(max(k_chrom, 0), n)
                if k_chrom == 0:
                    continue
                # pick random start indices, then expand by block_size_bp window
                # repeat until reaching k_chrom unique features
                selected = set()
                # map from ordered index to position
                ord_pos = pos_arr[order]
                while len(selected) < k_chrom:
                    j = int(rng.integers(0, n))
                    start_pos = int(ord_pos[j])
                    end_pos = start_pos + int(self.block_size_bp)
                    # include all within window
                    in_block = np.where((ord_pos >= start_pos) & (ord_pos <= end_pos))[0]
                    for ii in in_block.tolist():
                        selected.add(int(order[ii]))
                        if len(selected) >= k_chrom:
                            break
                sel = np.fromiter(selected, dtype=np.int32)
                X2[:, sel] = 0
            return X2

        if alg == "predefined":
            if self.mask_feature_indices is None or len(self.mask_feature_indices) == 0:
                return X2
            cols = np.asarray(self.mask_feature_indices, dtype=np.int32)
            X2[:, cols] = 0
            return X2

        raise ValueError(f"Unknown mask_alg: {alg}")


def load_mask_feature_list(path: str, feature_names: List[str]) -> np.ndarray:
    """Load a list of chr_pos features (one per line) and convert to column indices."""
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s in feat_to_idx:
                cols.append(feat_to_idx[s])
    return np.array(sorted(set(cols)), dtype=np.int32)
