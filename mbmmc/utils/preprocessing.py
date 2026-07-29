from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif

BinarizeEqualMode = Literal["positive", "negative"]

class BetaBinarizer(BaseEstimator, TransformerMixin):
    """Convert beta values to {-1,0,1} by threshold.
    - beta > t -> 1
    - beta < t -> -1
    - missing (NaN) -> 0
    If beta == t, controlled by equal_mode.
    """

    def __init__(self, threshold: float = 0.6, equal_mode: BinarizeEqualMode = "positive"):
        self.threshold = float(threshold)
        self.equal_mode = equal_mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros_like(X, dtype=np.int8)
        nan_mask = np.isnan(X)
        gt = X > self.threshold
        lt = X < self.threshold
        eq = (~nan_mask) & (~gt) & (~lt)

        out[gt] = 1
        out[lt] = -1
        if self.equal_mode == "positive":
            out[eq] = 1
        else:
            out[eq] = -1
        out[nan_mask] = 0
        return out
        # 过滤所有样本数值一样的特征（常数列）
        # 计算每列的唯一值数量，忽略NaN（已经处理为0）
#        unique_counts = np.array([len(np.unique(out[:, i])) for i in range(out.shape[1])])
        # 保留唯一值数量大于1的列（即有变化的特征）
#        valid_columns = unique_counts > 1
        # 如果所有特征都被过滤掉了，可以返回空数组或做其他处理
#        if not np.any(valid_columns):
#            raise ValueError("所有特征在所有样本中取值完全一致，无法进行有效分析！！！\n")
#            return np.empty((out.shape[0], 0), dtype=np.int8)
#        return out[:, valid_columns]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select top-k features by a method fit on training data only.

    Methods:
      - none: keep all features
      - variance_topk: top-k by variance on int8 features
      - f_classif_topk: top-k by ANOVA F score (multi-class)
      - pairwise_score_topk: top-k by sum of pairwise absolute differences of class means
        (recommended when avoiding batch correction; computed in binarized space)

    Optional:
      - corr_filter: if >0, apply greedy de-correlation to the candidate set using absolute Pearson correlation.
        This runs *after* TopK selection and can reduce redundancy / platform-driven correlated blocks.
    """

    def __init__(
        self,
        method: str = "none",
        topk: int = 50000,
        random_state: int = 42,
        corr_filter: float = 0.0,
    ):
        self.method = method
        self.topk = int(topk)
        self.random_state = int(random_state)
        self.corr_filter = float(corr_filter)
        self.indices_: Optional[np.ndarray] = None

    def _pairwise_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        classes = np.unique(y)
        means = []
        for c in classes:
            means.append(X[y == c].astype(np.float32).mean(axis=0))
        means = np.vstack(means)  # (C, F)
        # sum over all pairs |mean_i - mean_j|
        score = np.zeros(means.shape[1], dtype=np.float32)
        for i in range(means.shape[0]):
            for j in range(i + 1, means.shape[0]):
                score += np.abs(means[i] - means[j])
        return score

    def _corr_filter_indices(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        thr = float(self.corr_filter)
        if thr <= 0:
            return idx
        if thr >= 1.0:
            # correlation threshold >=1 means no filtering
            return idx

        rng = np.random.default_rng(self.random_state)

        # To keep runtime predictable, only apply correlation filtering to at most this many candidates.
       # max_candidates = 20000
       # if idx.shape[0] > max_candidates:
       #     idx = idx[:max_candidates]
        Xc = X[:, idx].astype(np.float32, copy=False)
        # Standardize each column
        mean = Xc.mean(axis=0)
        Xc = Xc - mean
        std = Xc.std(axis=0)
        valid = std > 1e-6
        idx = idx[valid]
        Xc = Xc[:, valid]
        std = std[valid]
        Xc = Xc / std

        n = Xc.shape[0]
        m = Xc.shape[1]
        if m <= 1:
            return idx.astype(np.int32)

        kept_idx = []
        kept_mat = np.empty((n, m), dtype=np.float32)
        kept_count = 0

        # when kept is large, compare only to a random subset to reduce cost
        max_full_check = 5000
        sample_check = 1024

        for j in range(m):
            v = Xc[:, j]
            if kept_count == 0:
                kept_mat[:, 0] = v
                kept_count = 1
                kept_idx.append(int(idx[j]))
                continue

            if kept_count > max_full_check:
                sel = rng.choice(kept_count, size=min(sample_check, kept_count), replace=False)
                K = kept_mat[:, sel]
            else:
                K = kept_mat[:, :kept_count]

            corr = np.abs((v @ K) / (n - 1))
            if float(np.max(corr)) < thr:
                kept_mat[:, kept_count] = v
                kept_count += 1
                kept_idx.append(int(idx[j]))

        return np.array(kept_idx, dtype=np.int32)

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_features = X.shape[1]
        if self.method == "none":
            self.indices_ = np.arange(n_features, dtype=np.int32)
            return self

        k = min(self.topk, n_features)
        if k <= 0:
            self.indices_ = np.arange(0, dtype=np.int32)
            return self

        if self.method == "variance_topk":
            var = X.astype(np.float32).var(axis=0)
            order = np.argsort(var)[::-1]
            idx = order[:k].astype(np.int32)
            self.indices_ = self._corr_filter_indices(X.astype(np.float32), idx)
            return self

        if self.method == "f_classif_topk":
            if y is None:
                raise ValueError("f_classif_topk requires y")
            scores, _ = f_classif(X.astype(np.float32), y)
            scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
            order = np.argsort(scores)[::-1]
            idx = order[:k].astype(np.int32)
            self.indices_ = self._corr_filter_indices(X.astype(np.float32), idx)
            return self

        if self.method == "pairwise_score_topk":
            if y is None:
                raise ValueError("pairwise_score_topk requires y")
            score = self._pairwise_score(X, y)
            order = np.argsort(score)[::-1]
            idx = order[:k].astype(np.int32)
            self.indices_ = self._corr_filter_indices(X.astype(np.float32), idx)
            return self

        raise ValueError(f"Unknown feature selection method: {self.method}")

    def transform(self, X):
        if self.indices_ is None:
            raise RuntimeError("FeatureSelector must be fit before transform")
        X = np.asarray(X)
        return X[:, self.indices_]
