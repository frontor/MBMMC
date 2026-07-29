from __future__ import annotations
import hashlib, json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


def _stable_hash_int(text: str, seed: int = 17) -> int:
    h = hashlib.blake2b((str(seed) + "::" + text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def observed_variance_ignore_nan(x: np.ndarray) -> float:
    obs = x[~np.isnan(x)]
    if obs.size <= 1:
        return 0.0
    return float(np.var(obs))


def technical_qc_features(
    X: pd.DataFrame,
    min_observed_per_feature: int = 3,
    min_observed_fraction: float = 0.0,
    min_variance_observed: float = 0.0,
    max_missing_fraction: float = 1.0,
) -> List[str]:
    """Label-free feature QC.

    Missing CpGs are not encoded as zero for QC. Variance is calculated only on
    observed methylation values.
    """
    n = X.shape[0]
    keep = []
    arr = X.to_numpy(dtype=float, copy=False)
    cols = list(map(str, X.columns))
    obs_counts = np.sum(~np.isnan(arr), axis=0)
    for j, c in enumerate(cols):
        obs = int(obs_counts[j])
        obs_frac = obs / max(n, 1)
        miss_frac = 1.0 - obs_frac
        if obs < min_observed_per_feature:
            continue
        if obs_frac < min_observed_fraction:
            continue
        if miss_frac > max_missing_fraction:
            continue
        if min_variance_observed > 0 and observed_variance_ignore_nan(arr[:, j]) < min_variance_observed:
            continue
        keep.append(c)
    return keep


def supervised_select_features(
    X: pd.DataFrame,
    y,
    method: str = "f_classif_topk",
    topk: int = 50000,
) -> List[str]:
    method = method or "none"
    if method in {"none", "technical_only", "technical_only_qc"}:
        return list(map(str, X.columns))
    topk = min(int(topk), X.shape[1])
    if method == "variance_topk":
        vars_ = np.array([observed_variance_ignore_nan(X.iloc[:, j].to_numpy(dtype=float)) for j in range(X.shape[1])])
        idx = np.argsort(vars_)[::-1][:topk]
        return list(map(str, X.columns[idx]))
    if method == "f_classif_topk":
        Xv = X.to_numpy(dtype=float)
        # f_classif cannot handle NaN; fill with column means based on training set only.
        means = np.nanmean(Xv, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
        inds = np.where(np.isnan(Xv))
        Xv = Xv.copy()
        Xv[inds] = means[inds[1]]
        scores, _ = f_classif(Xv, y)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        idx = np.argsort(scores)[::-1][:topk]
        return list(map(str, X.columns[idx]))
    if method == "pairwise_score_topk":
        # Lightweight robust alternative: rank by between-class mean variance.
        Xv = X.to_numpy(dtype=float)
        classes = pd.Series(y).astype(str).unique()
        means = []
        for cls in classes:
            m = np.nanmean(Xv[pd.Series(y).astype(str).to_numpy() == cls, :], axis=0)
            means.append(m)
        score = np.nanvar(np.vstack(means), axis=0)
        score = np.nan_to_num(score, nan=-np.inf)
        idx = np.argsort(score)[::-1][:topk]
        return list(map(str, X.columns[idx]))
    raise ValueError(f"Unknown feature selection method: {method}")


def make_value_mask(X: pd.DataFrame, feature_list: List[str], fill_value: float = 0.0):
    X2 = X.reindex(columns=feature_list)
    values = X2.to_numpy(dtype=np.float32)
    mask = (~np.isnan(values)).astype(np.float32)
    values = np.nan_to_num(values, nan=float(fill_value)).astype(np.float32)
    coverage = mask.mean(axis=1).astype(np.float32)
    return values, mask, coverage


def binarize_values(
    values: np.ndarray,
    mask: np.ndarray,
    beta_threshold: float = 0.6,
    positive_value: float = 1.0,
    negative_value: float = -1.0,
    equal_mode: str = "negative",
) -> np.ndarray:
    """Convert observed beta values to a robust ternary representation.

    Missing values remain 0 through multiplication by mask.

    equal_mode controls the boundary case exactly equal to beta_threshold:
      - "negative": beta > threshold is positive; beta <= threshold is negative
      - "positive": beta >= threshold is positive; beta < threshold is negative

    The default is "negative" to match RF/crossNN binarization semantics.
    """
    thr = float(beta_threshold)
    if equal_mode == "positive":
        ternary = np.where(values >= thr, float(positive_value), float(negative_value)).astype(np.float32)
    elif equal_mode == "negative":
        ternary = np.where(values > thr, float(positive_value), float(negative_value)).astype(np.float32)
    else:
        raise ValueError("equal_mode must be one of: positive, negative")
    ternary = ternary * mask.astype(np.float32)
    return ternary.astype(np.float32)


def build_hash_map(feature_list: List[str], n_bins: int = 32768, seed: int = 17) -> Dict[str, object]:
    bins = []
    signs = []
    for f in feature_list:
        h = _stable_hash_int(str(f), seed=seed)
        bins.append(h % int(n_bins))
        signs.append(1.0 if ((h >> 17) & 1) == 0 else -1.0)
    counts = np.bincount(np.array(bins, dtype=int), minlength=int(n_bins)).astype(float)
    counts[counts == 0] = 1.0
    return {
        "n_bins": int(n_bins),
        "seed": int(seed),
        "feature_list": list(map(str, feature_list)),
        "bins": bins,
        "signs": signs,
        "bin_feature_counts": counts.tolist(),
    }


def apply_hash(values: np.ndarray, mask: np.ndarray, hash_map: Dict[str, object]):
    n_bins = int(hash_map["n_bins"])
    bins = np.asarray(hash_map["bins"], dtype=np.int64)
    signs = np.asarray(hash_map["signs"], dtype=np.float32)
    counts = np.asarray(hash_map["bin_feature_counts"], dtype=np.float32)
    hv = np.zeros((values.shape[0], n_bins), dtype=np.float32)
    hm = np.zeros((values.shape[0], n_bins), dtype=np.float32)
    # Loop over source features; this is memory-safe and deterministic.
    for j in range(values.shape[1]):
        b = bins[j]
        s = signs[j]
        obs = mask[:, j]
        hv[:, b] += values[:, j] * obs * s
        hm[:, b] += obs
    hv = hv / np.sqrt(counts)[None, :]
    hm = hm / counts[None, :]
    return hv.astype(np.float32), hm.astype(np.float32), hm.mean(axis=1).astype(np.float32)


def fit_feature_bundle(
    X: pd.DataFrame,
    y=None,
    feature_select: str = "none",
    topk: int = 50000,
    technical_qc: Optional[Dict[str, object]] = None,
    input_compression: str = "auto",
    hash_bins: int = 32768,
    max_dense_features: int = 30000,
    hash_seed: int = 17,
    value_mode: str = "raw",
    beta_threshold: float = 0.6,
    beta_threshold_equal_mode: str = "negative",
) -> Dict[str, object]:
    technical_qc = technical_qc or {}
    value_mode = str(value_mode or "raw").lower()
    if value_mode not in {"raw", "ternary", "hybrid"}:
        raise ValueError("value_mode must be one of raw, ternary, hybrid")
    beta_threshold_equal_mode = str(beta_threshold_equal_mode or "negative").lower()
    if beta_threshold_equal_mode not in {"negative", "positive"}:
        raise ValueError("beta_threshold_equal_mode must be one of: negative, positive")
    qc_features = technical_qc_features(X, **technical_qc)
    Xqc = X.reindex(columns=qc_features)
    selected = supervised_select_features(Xqc, y, method=feature_select, topk=topk)
    selected = list(map(str, selected))
    compression = input_compression
    if compression == "auto":
        compression = "hash" if len(selected) > int(max_dense_features) else "dense"
    n_value_channels = 2 if value_mode == "hybrid" else 1
    bundle = {
        "feature_select": feature_select,
        "feature_policy": "technical_only_qc" if feature_select in {"none", "technical_only", "technical_only_qc"} else "supervised_or_variance_selection",
        "topk": int(topk),
        "technical_qc": technical_qc,
        "original_feature_list": selected,
        "n_original_features": len(selected),
        "input_compression": compression,
        "max_dense_features": int(max_dense_features),
        "value_mode": value_mode,
        "beta_threshold": float(beta_threshold),
        "beta_threshold_equal_mode": beta_threshold_equal_mode,
        "n_value_channels": int(n_value_channels),
    }
    if compression == "hash":
        bundle["hash_map"] = build_hash_map(selected, n_bins=hash_bins, seed=hash_seed)
        bundle["n_model_features"] = int(hash_bins) * n_value_channels
    elif compression == "dense":
        bundle["hash_map"] = None
        bundle["n_model_features"] = len(selected) * n_value_channels
    else:
        raise ValueError("input_compression must be one of auto, dense, hash")
    return bundle


def _apply_value_mode(
    values: np.ndarray,
    mask: np.ndarray,
    value_mode: str,
    beta_threshold: float,
    beta_threshold_equal_mode: str = "negative",
):
    mode = str(value_mode or "raw").lower()
    if mode == "raw":
        return values.astype(np.float32), mask.astype(np.float32)
    if mode == "ternary":
        return binarize_values(
            values,
            mask,
            beta_threshold=beta_threshold,
            equal_mode=beta_threshold_equal_mode,
        ), mask.astype(np.float32)
    if mode == "hybrid":
        ternary = binarize_values(
            values,
            mask,
            beta_threshold=beta_threshold,
            equal_mode=beta_threshold_equal_mode,
        )
        return np.concatenate([values.astype(np.float32), ternary.astype(np.float32)], axis=1), np.concatenate([mask.astype(np.float32), mask.astype(np.float32)], axis=1)
    raise ValueError("value_mode must be one of raw, ternary, hybrid")


def transform_by_feature_bundle(X: pd.DataFrame, bundle: Dict[str, object], fill_value: float = 0.0):
    values, mask, coverage = make_value_mask(X, bundle["original_feature_list"], fill_value=fill_value)
    value_mode = str(bundle.get("value_mode", "raw")).lower()
    beta_threshold = float(bundle.get("beta_threshold", 0.6))
    beta_threshold_equal_mode = str(bundle.get("beta_threshold_equal_mode", "negative")).lower()
    if bundle.get("input_compression") == "hash":
        if value_mode == "hybrid":
            # Hash raw beta and robust ternary channels separately, then concatenate.
            raw_values, raw_mask = values.astype(np.float32), mask.astype(np.float32)
            ternary_values = binarize_values(
                values,
                mask,
                beta_threshold=beta_threshold,
                equal_mode=beta_threshold_equal_mode,
            )
            hv_raw, hm_raw, _ = apply_hash(raw_values, raw_mask, bundle["hash_map"])
            hv_ter, hm_ter, _ = apply_hash(ternary_values, raw_mask, bundle["hash_map"])
            return np.concatenate([hv_raw, hv_ter], axis=1).astype(np.float32), np.concatenate([hm_raw, hm_ter], axis=1).astype(np.float32), coverage
        values2, mask2 = _apply_value_mode(
            values,
            mask,
            value_mode,
            beta_threshold,
            beta_threshold_equal_mode=beta_threshold_equal_mode,
        )
        values2, mask2, coverage2 = apply_hash(values2, mask2, bundle["hash_map"])
        # retain original observed coverage as primary clinical low-coverage measure
        return values2, mask2, coverage
    values2, mask2 = _apply_value_mode(
        values,
        mask,
        value_mode,
        beta_threshold,
        beta_threshold_equal_mode=beta_threshold_equal_mode,
    )
    return values2, mask2, coverage
