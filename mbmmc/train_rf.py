from __future__ import annotations

import os
import json
import argparse
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
    RandomizedSearchCV,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .utils.data_io import load_reference
from .utils.sample_weights import read_meta_for_weights, resolve_sample_weight_vector, write_sample_weight_audit
from .utils.preprocessing import BetaBinarizer, FeatureSelector
from .utils.masking import Masker
from .utils.model_zoo import build_model, param_distributions
from .utils.metrics import compute_metrics, confusion_matrix_norm
from .utils.plots import plot_metrics_bar, plot_confusion_matrix, plot_macro_roc


def _ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)


def _load_mask_feature_list(path: str, features: List[str]) -> np.ndarray:
    feat_set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            feat_set.add(line)
    idx = [i for i, feat in enumerate(features) if feat in feat_set]
    return np.array(idx, dtype=np.int32)


def _load_feature_weights(path: str, features: List[str]) -> np.ndarray:
    # Expect TSV/CSV with columns: chr_pos, weight
    dfw = None
    try:
        dfw = pd.read_csv(path, sep="\t")
    except Exception:
        dfw = pd.read_csv(path)
    if "chr_pos" not in dfw.columns or "weight" not in dfw.columns:
        raise ValueError("--mask_feature_weights must contain columns: chr_pos, weight")
    wmap = dict(zip(dfw["chr_pos"].astype(str).tolist(), dfw["weight"].astype(float).tolist()))
    w = np.array([wmap.get(f, 0.0) for f in features], dtype=np.float64)
    return w


def _prefilter_features_by_presence(
    X_beta: np.ndarray,
    features: List[str],
    chr_arr: Optional[np.ndarray],
    pos_arr: Optional[np.ndarray],
    min_present_rate: float,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    if min_present_rate <= 0:
        return X_beta, features, chr_arr, pos_arr
    present = ~np.isnan(X_beta)
    present_rate = present.mean(axis=0)
    keep = present_rate >= float(min_present_rate)
    if keep.all():
        return X_beta, features, chr_arr, pos_arr
    X2 = X_beta[:, keep]
    f2 = [f for f, k in zip(features, keep.tolist()) if k]
    c2 = chr_arr[keep] if chr_arr is not None else None
    p2 = pos_arr[keep] if pos_arr is not None else None
    return X2, f2, c2, p2


def _prefilter_features_by_platform_missing_diff(
    X_beta: np.ndarray,
    features: List[str],
    chr_arr: Optional[np.ndarray],
    pos_arr: Optional[np.ndarray],
    platform: Optional[np.ndarray],
    max_abs_missing_diff: float,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    if platform is None:
        return X_beta, features, chr_arr, pos_arr
    max_abs_missing_diff = float(max_abs_missing_diff)
    if max_abs_missing_diff <= 0:
        return X_beta, features, chr_arr, pos_arr
    plat = np.asarray(platform, dtype=object)
    uniq = np.unique(plat)
    if len(uniq) < 2:
        return X_beta, features, chr_arr, pos_arr

    miss = np.isnan(X_beta)
    miss_rates = []
    for u in uniq:
        idx = np.where(plat == u)[0]
        if len(idx) == 0:
            continue
        miss_rates.append(miss[idx].mean(axis=0))
    if len(miss_rates) < 2:
        return X_beta, features, chr_arr, pos_arr
    miss_rates = np.vstack(miss_rates)
    diff = miss_rates.max(axis=0) - miss_rates.min(axis=0)
    keep = diff <= max_abs_missing_diff
    if keep.all():
        return X_beta, features, chr_arr, pos_arr

    X2 = X_beta[:, keep]
    f2 = [f for f, k in zip(features, keep.tolist()) if k]
    c2 = chr_arr[keep] if chr_arr is not None else None
    p2 = pos_arr[keep] if pos_arr is not None else None
    return X2, f2, c2, p2


def _make_groups(group_by: str, platform: Optional[np.ndarray], material: Optional[np.ndarray]) -> np.ndarray:
    if group_by == "platform":
        if platform is None:
            raise ValueError("meta.txt must contain Platform column for group_by=platform")
        return np.asarray(platform, dtype=object)
    if group_by == "material":
        if material is None:
            raise ValueError("meta.txt must contain Material column for group_by=material")
        return np.asarray(material, dtype=object)
    if group_by == "platform_material":
        if platform is None or material is None:
            raise ValueError("meta.txt must contain Platform and Material columns for group_by=platform_material")
        return np.asarray([f"{p}__{m}" for p, m in zip(platform.tolist(), material.tolist())], dtype=object)
    raise ValueError(f"Unknown group_by: {group_by}")


def _groups_from_meta_column(meta: pd.DataFrame, samples: List[str], group_col: str) -> np.ndarray:
    """Return group labels aligned to samples from an arbitrary metadata column.

    Missing/empty values are replaced by the Sample ID so that real array samples
    without a simulated LatentID/SplitGroup do not collapse into one giant group.
    This is intended for leakage-safe internal CV, e.g. --cv_group_col SplitGroup
    where simulated paired-latent TAPS/WGBS/ONT records share one group.
    """
    if not group_col:
        raise ValueError("group_col must be non-empty")
    col_map = {str(c).strip().lower(): c for c in meta.columns}
    key = str(group_col).strip().lower()
    if key not in col_map:
        raise ValueError(f"--cv_group_col {group_col!r} not found in meta columns: {list(meta.columns)}")
    col = col_map[key]
    vals = meta[col].copy()
    vals = vals.where(~pd.isna(vals), other="")
    vals = vals.astype(str).str.strip()
    samples_s = pd.Series(list(map(str, samples)), index=vals.index)
    missing_like = vals.eq("") | vals.str.lower().isin(["na", "nan", "none", "null"])
    vals.loc[missing_like] = samples_s.loc[missing_like]
    return vals.to_numpy(dtype=object)


def _resolve_cv_groups(
    cv_group_mode: str,
    cv_group_col: str,
    meta_aligned: pd.DataFrame,
    samples: List[str],
    group_by: str,
    platform: Optional[np.ndarray],
    material: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if cv_group_mode == "none":
        return None
    if cv_group_col:
        return _groups_from_meta_column(meta_aligned, samples, cv_group_col)
    return _make_groups(group_by, platform, material)


def _outer_splitter(cv_group_mode: str, outer_folds: int, random_state: int):
    if cv_group_mode == "none":
        return StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    if cv_group_mode == "batch":
        return StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    if cv_group_mode == "logo":
        return LeaveOneGroupOut()
    raise ValueError(f"Unknown cv_group_mode: {cv_group_mode}")
def _platform_marker_scores_maxmin(X_bin: np.ndarray, groups: np.ndarray) -> Optional[np.ndarray]:
    """Score each feature by cross-group mean range: max(mean_g) - min(mean_g).
    X_bin should be {-1,0,1} (int8 ok). groups are platform/material strings.
    """
    if groups is None:
        return None
    g = np.asarray(groups, dtype=object)
    uniq = np.unique(g)
    if uniq.size < 2:
        return None
    Xf = X_bin.astype(np.float32, copy=False)
    means = []
    for u in uniq.tolist():
        m = Xf[g == u].mean(axis=0)
        means.append(m)
    means = np.vstack(means)  # (G, F)
    scores = means.max(axis=0) - means.min(axis=0)
    return scores
def _apply_drop_platform_markers(
    X_bin: np.ndarray,
    features: List[str],
    chr_arr: Optional[np.ndarray],
    pos_arr: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    drop_n: int,
    outdir: str,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Drop top-N platform-associated features globally (based on groups provided)."""
    if drop_n is None or int(drop_n) <= 0 or groups is None:
        return X_bin, features, chr_arr, pos_arr
    scores = _platform_marker_scores_maxmin(X_bin, groups)
    if scores is None:
        print("[drop_platform_markers] groups < 2; skip dropping.")
        return X_bin, features, chr_arr, pos_arr
    p = X_bin.shape[1]
    # 至少保留 1 个特征，避免后续模型报 0 feature
    n_drop = int(min(drop_n, max(0, p - 1)))
    if n_drop <= 0:
        return X_bin, features, chr_arr, pos_arr
    order = np.argsort(scores)[::-1]
    drop_idx = order[:n_drop].astype(int)
    keep = np.ones(p, dtype=bool)
    keep[drop_idx] = False
    # 保存被删的 CpG 列表，便于复现实验
    try:
        os.makedirs(outdir, exist_ok=True)
        df_drop = pd.DataFrame({
            "feature": np.asarray(features, dtype=object)[drop_idx],
            "score": scores[drop_idx],
        }).sort_values("score", ascending=False)
        df_drop.to_csv(os.path.join(outdir, "platform_markers_dropped.csv"), index=False)
        print(f"[drop_platform_markers] Dropped {n_drop} / {p} features. "
            f"Saved to {os.path.join(outdir, 'platform_markers_dropped.csv')}")
    except Exception as e:
        print(f"[drop_platform_markers] Dropped {n_drop} / {p} features. (Save failed: {e})")
    X2 = X_bin[:, keep]
    f2 = [f for f, k in zip(features, keep.tolist()) if k]
    c2 = chr_arr[keep] if chr_arr is not None else None
    p2 = pos_arr[keep] if pos_arr is not None else None
    return X2, f2, c2, p2

def _inner_splitter(cv_group_mode: str, groups_train: Optional[np.ndarray], inner_folds: int, seed: int):
    if cv_group_mode in ("batch", "logo") and groups_train is not None:
        uniq = np.unique(groups_train)
        if len(uniq) >= inner_folds and inner_folds >= 2:
            return StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

def _align_proba_to_global_classes(
    proba: Optional[np.ndarray],
    est: Pipeline,
    n_classes: int,
) -> Optional[np.ndarray]:
    """Pad/align predict_proba output to fixed (n_samples, n_classes).
    When some classes are missing in a training fold (common with group CV),
    sklearn estimators return proba with fewer columns. This makes stacking
    across folds impossible unless we align columns using clf.classes_.
    """
    if proba is None:
        return None
    if proba.ndim != 2:
        return None
    if proba.shape[1] == n_classes:
        return proba
    # Get classes seen by classifier in this fold
    classes = getattr(est, "classes_", None)
    if classes is None and hasattr(est, "named_steps") and "clf" in est.named_steps:
        classes = getattr(est.named_steps["clf"], "classes_", None)
    if classes is None:
        # Can't align safely
        return None
    classes = np.asarray(classes, dtype=int)
    full = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    # Guard against unexpected class ids
    ok = (classes >= 0) & (classes < n_classes)
    if not np.all(ok):
        classes = classes[ok]
        proba = proba[:, ok]
    full[:, classes] = proba
    return full

class FeatureSelectorThenMasker(BaseEstimator, TransformerMixin):
    """Apply FeatureSelector first, then Masker, without leaking test-fold data.

    The old RF/ML pipeline was mask -> select -> clf.  This transformer keeps
    the selector inside the sklearn Pipeline, so each CV training fold learns its
    own TopK features, but it remaps chromosome positions, predefined mask
    indices, and external feature weights to the selected feature space before
    applying the mask.

    It also applies a fold-local non-constant filter before supervised TopK.
    A feature may be non-constant globally but constant in a specific CV training
    fold; sklearn.f_classif warns and emits NaNs for those fold-local constants.
    Filtering them locally keeps CV leakage-safe behavior while avoiding noisy
    warnings.
    """

    def __init__(
        self,
        feature_select: str = "variance_topk",
        topk: int = 50000,
        corr_filter: float = 0.0,
        random_state: int = 42,
        mask_alg: str = "mcar",
        mask_rate: float = 0.90,
        mask_on: str = "train",
        chr_arr: Optional[np.ndarray] = None,
        pos_arr: Optional[np.ndarray] = None,
        mask_feature_indices: Optional[np.ndarray] = None,
        block_size_bp: int = 500000,
        mask_beta_kappa: float = 20.0,
        mask_beta_alpha: float = -1.0,
        mask_beta_beta: float = -1.0,
        mask_weight_power: float = 1.0,
        mask_weight_eps: float = 1e-3,
        mask_feature_weights: Optional[np.ndarray] = None,
        drop_fold_constant_features: bool = True,
    ):
        self.feature_select = feature_select
        self.topk = int(topk)
        self.corr_filter = float(corr_filter)
        self.random_state = int(random_state)
        self.mask_alg = mask_alg
        self.mask_rate = float(mask_rate)
        self.mask_on = mask_on
        self.chr_arr = chr_arr
        self.pos_arr = pos_arr
        self.mask_feature_indices = mask_feature_indices
        self.block_size_bp = int(block_size_bp)
        self.mask_beta_kappa = float(mask_beta_kappa)
        self.mask_beta_alpha = float(mask_beta_alpha)
        self.mask_beta_beta = float(mask_beta_beta)
        self.mask_weight_power = float(mask_weight_power)
        self.mask_weight_eps = float(mask_weight_eps)
        self.mask_feature_weights = mask_feature_weights
        self.drop_fold_constant_features = bool(drop_fold_constant_features)

    def _fold_feature_indices(self, X: np.ndarray) -> np.ndarray:
        """Return feature indices kept before TopK inside the current fit fold.

        Global constant columns are already removed before the Pipeline is built.
        However, a column can still be constant within a specific CV training
        split.  sklearn.f_classif warns and emits NaNs for those columns.  We
        remove them locally before TopK and map selected indices back to the
        original post-global-filter feature space.
        """
        n_features = int(X.shape[1])
        if n_features <= 0:
            raise ValueError("No features available for FeatureSelectorThenMasker")

        # Do not change feature_select=none behavior.  In this mode there is no
        # univariate f_classif call to warn, and dropping features would silently
        # change the meaning of "none".
        if (not self.drop_fold_constant_features) or self.feature_select == "none":
            return np.arange(n_features, dtype=np.int32)

        # X is already int8 {-1, 0, 1}; min/max is fast and avoids high-memory unique().
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        keep = np.asarray(col_min != col_max, dtype=bool)

        if not np.any(keep):
            raise ValueError(
                "All features are constant within this training fold after binarization. "
                "Try fewer CV folds, larger training folds, or a less fragmented group split."
            )

        return np.flatnonzero(keep).astype(np.int32)

    def _selected_chr_pos(self, sel_idx: np.ndarray):
        chr_sel = None if self.chr_arr is None else np.asarray(self.chr_arr)[sel_idx]
        pos_sel = None if self.pos_arr is None else np.asarray(self.pos_arr)[sel_idx]
        return chr_sel, pos_sel

    def _selected_mask_indices(self, sel_idx: np.ndarray) -> Optional[np.ndarray]:
        if self.mask_feature_indices is None:
            return None
        old = np.asarray(self.mask_feature_indices, dtype=int)
        if old.size == 0:
            return np.asarray([], dtype=np.int32)
        old_to_new = {int(old_i): int(new_i) for new_i, old_i in enumerate(sel_idx.tolist())}
        remapped = [old_to_new[int(i)] for i in old.tolist() if int(i) in old_to_new]
        return np.asarray(remapped, dtype=np.int32)

    def _selected_feature_weights(self, sel_idx: np.ndarray, n_full_features: int) -> Optional[np.ndarray]:
        if self.mask_feature_weights is None:
            return None
        w = np.asarray(self.mask_feature_weights, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError("mask_feature_weights must be 1D")
        if w.shape[0] == n_full_features:
            return w[sel_idx]
        if w.shape[0] == sel_idx.shape[0]:
            return w
        raise ValueError(
            "mask_feature_weights length must match either full features before TopK "
            "or selected features after TopK"
        )

    def _make_masker(self, sel_idx: np.ndarray, n_full_features: int) -> Masker:
        chr_sel, pos_sel = self._selected_chr_pos(sel_idx)
        return Masker(
            mask_alg=self.mask_alg,
            mask_rate=self.mask_rate,
            mask_on=self.mask_on,
            random_state=self.random_state,
            chr_arr=chr_sel,
            pos_arr=pos_sel,
            mask_feature_indices=self._selected_mask_indices(sel_idx),
            block_size_bp=self.block_size_bp,
            beta_kappa=self.mask_beta_kappa,
            beta_alpha=self.mask_beta_alpha,
            beta_beta=self.mask_beta_beta,
            feature_weights=self._selected_feature_weights(sel_idx, n_full_features),
            weight_power=self.mask_weight_power,
            weight_eps=self.mask_weight_eps,
        )

    def _fit_selector(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fold_feature_indices_ = self._fold_feature_indices(X)
        X_for_selector = X[:, self.fold_feature_indices_]

        self.selector_ = FeatureSelector(
            method=self.feature_select,
            topk=self.topk,
            random_state=self.random_state,
            corr_filter=self.corr_filter,
        )

        # sklearn.f_classif emits warnings for features that are constant within
        # one or more classes. In binarized methylation data, such features can
        # be valid strong markers; FeatureSelector already converts NaN scores
        # to -inf and keeps +inf scores. Suppress only these known numerical
        # warnings so training logs remain clean without changing ranking logic.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Features .* are constant\.",
                category=UserWarning,
                module=r"sklearn\.feature_selection\._univariate_selection",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"invalid value encountered in divide",
                category=RuntimeWarning,
                module=r"sklearn\.feature_selection\._univariate_selection",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"divide by zero encountered in divide",
                category=RuntimeWarning,
                module=r"sklearn\.feature_selection\._univariate_selection",
            )
            self.selector_.fit(X_for_selector, y)

        local_idx = np.asarray(self.selector_.indices_, dtype=np.int32)
        if local_idx.size == 0:
            raise ValueError("FeatureSelector selected zero features")

        # selector_.indices_ are in X_for_selector coordinates; map them back to
        # the original post-global-filter feature coordinates used by chr_arr,
        # pos_arr, predefined mask indices, and full-length feature weights.
        self.indices_ = self.fold_feature_indices_[local_idx].astype(np.int32)
        return X[:, self.indices_]

    def fit(self, X, y=None):
        X = np.asarray(X)
        X_sel = self._fit_selector(X, y)
        self.masker_ = self._make_masker(self.indices_, X.shape[1])
        self.masker_.fit(X_sel, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X = np.asarray(X)
        X_sel = self._fit_selector(X, y)
        self.masker_ = self._make_masker(self.indices_, X.shape[1])
        return self.masker_.fit_transform(X_sel, y)

    def transform(self, X):
        if not hasattr(self, "selector_") or not hasattr(self, "masker_") or not hasattr(self, "indices_"):
            raise RuntimeError("FeatureSelectorThenMasker must be fit before transform")
        X_sel = np.asarray(X)[:, self.indices_]
        return self.masker_.transform(X_sel)


def build_search_pipeline(
    model_name: str,
    mask_alg: str,
    mask_rate: float,
    mask_on: str,
    chr_arr: Optional[np.ndarray],
    pos_arr: Optional[np.ndarray],
    mask_feature_indices: Optional[np.ndarray],
    block_size_bp: int,
    # new mask params
    mask_beta_kappa: float,
    mask_beta_alpha: float,
    mask_beta_beta: float,
    mask_weight_power: float,
    mask_weight_eps: float,
    mask_feature_weights: Optional[np.ndarray],
    # feature selection
    feature_select: str,
    topk: int,
    corr_filter: float,
    random_state: int,
) -> Pipeline:
    select_mask = FeatureSelectorThenMasker(
        feature_select=feature_select,
        topk=topk,
        corr_filter=corr_filter,
        random_state=random_state,
        mask_alg=mask_alg,
        mask_rate=mask_rate,
        mask_on=mask_on,
        chr_arr=chr_arr,
        pos_arr=pos_arr,
        mask_feature_indices=mask_feature_indices,
        block_size_bp=block_size_bp,
        mask_beta_kappa=mask_beta_kappa,
        mask_beta_alpha=mask_beta_alpha,
        mask_beta_beta=mask_beta_beta,
        mask_weight_power=mask_weight_power,
        mask_weight_eps=mask_weight_eps,
        mask_feature_weights=mask_feature_weights,
    )
    clf = build_model(model_name, random_state=random_state)
    return Pipeline([
        ("select_mask", select_mask),
        ("clf", clf),
    ])


def _fit_and_eval_one_fold(
    X_bin: np.ndarray,
    y_enc: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    pipe: Pipeline,
    param_dist: Dict[str, Any],
    inner_folds: int,
    search_iters: int,
    random_state: int,
    scoring: str,
    n_jobs: int,
    cv_group_mode: str,
    groups: Optional[np.ndarray],
    fold: int,
    use_sample_weight: str,
    base_sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    X_tr, X_te = X_bin[tr_idx], X_bin[te_idx]
    y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]
    g_tr = None if groups is None else groups[tr_idx]
    print(f"[fold {fold}] train classes={np.unique(y_tr)}, test classes={np.unique(y_te)}")
    inner = _inner_splitter(cv_group_mode, g_tr, inner_folds, random_state + fold + 11)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=search_iters,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=inner,
        refit=True,
        verbose=0,
        random_state=random_state + fold + 101,
    )

    fit_params: Dict[str, Any] = {}
    sw = None
    if base_sample_weight is not None:
        sw = np.asarray(base_sample_weight, dtype=np.float64)[tr_idx].copy()
    if use_sample_weight == "balanced":
        sw_bal = compute_sample_weight(class_weight="balanced", y=y_tr).astype(np.float64)
        sw = sw_bal if sw is None else sw * sw_bal
    if sw is not None:
        fit_params["clf__sample_weight"] = sw

    if isinstance(inner, StratifiedGroupKFold) and g_tr is not None:
        search.fit(X_tr, y_tr, groups=g_tr, **fit_params)
    else:
        search.fit(X_tr, y_tr, **fit_params)

    best_est = search.best_estimator_
    y_hat = best_est.predict(X_te)
    n_classes = int(len(np.unique(y_enc)))  # Global class count used to align fold-level probability columns.

    proba = None
    if hasattr(best_est, "predict_proba"):
        try:
            proba = best_est.predict_proba(X_te)
            proba = _align_proba_to_global_classes(proba, best_est, n_classes)
        except Exception:
            proba = None
    row = compute_metrics(y_te, y_hat, proba, labels=list(range(n_classes)))
    row.update({
        "fold": fold,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "best_params": json.dumps(search.best_params_, ensure_ascii=False),
    })
    return row, y_hat, proba


def run_cv(
    X_bin: np.ndarray,
    y_enc: np.ndarray,
    class_names: List[str],
    pipe: Pipeline,
    param_dist: Dict[str, Any],
    outer_folds: int,
    inner_folds: int,
    search_iters: int,
    random_state: int,
    scoring: str,
    n_jobs: int,
    outdir: str,
    groups: Optional[np.ndarray] = None,
    cv_group_mode: str = "none",
    use_sample_weight: str = "none",
    base_sample_weight: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    splitter = _outer_splitter(cv_group_mode, outer_folds, random_state)

    rows: List[Dict[str, Any]] = []
    all_true, all_pred, all_proba = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(splitter.split(X_bin, y_enc, groups=groups)):
        row, y_hat, proba = _fit_and_eval_one_fold(
            X_bin=X_bin,
            y_enc=y_enc,
            tr_idx=tr_idx,
            te_idx=te_idx,
            pipe=pipe,
            param_dist=param_dist,
            inner_folds=inner_folds,
            search_iters=search_iters,
            random_state=random_state,
            scoring=scoring,
            n_jobs=n_jobs,
            cv_group_mode=cv_group_mode,
            groups=groups,
            fold=fold,
            use_sample_weight=use_sample_weight,
            base_sample_weight=base_sample_weight,
        )
        rows.append(row)
        all_true.extend(y_enc[te_idx].tolist())
        all_pred.extend(y_hat.tolist())
        if proba is not None:
            all_proba.append(proba)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "cv_metrics.csv"), index=False)
    plot_metrics_bar(df, os.path.join(outdir, "plots", "metrics_bar.png"))

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)
    cmn = confusion_matrix_norm(all_true, all_pred, labels=list(range(len(class_names))))
    np.savetxt(os.path.join(outdir, "plots", "confusion_matrix.txt"),cmn,header="\t".join(class_names),delimiter="\t", fmt="%.2f")
    plot_confusion_matrix(cmn, class_names, os.path.join(outdir, "plots", "confusion_matrix.png"))

    proba_agg = np.vstack(all_proba) if len(all_proba) > 0 else None
    if proba_agg is not None:
        plot_macro_roc(all_true, proba_agg, list(range(len(class_names))),
                       os.path.join(outdir, "plots", "roc_macro.png"),
                       title="OvR ROC curves", class_names=class_names)

    summary = {
        "cv_group_mode": cv_group_mode,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "scoring": scoring,
        "mean_accuracy": float(df["accuracy"].mean()),
        "mean_balanced_accuracy": float(df["balanced_accuracy"].mean()),
        "mean_macro_f1": float(df["macro_f1"].mean()),
    }
    if "macro_roc_auc_ovr" in df.columns:
        summary["mean_macro_roc_auc_ovr"] = float(df["macro_roc_auc_ovr"].mean())
    if "log_loss" in df.columns:
        summary["mean_log_loss"] = float(df["log_loss"].mean())

    with open(os.path.join(outdir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return df, summary


def fit_final_and_export_bundle(
    X_bin: np.ndarray,
    y_enc: np.ndarray,
    class_names: List[str],
    features_full: List[str],
    chr_arr: Optional[np.ndarray],
    pos_arr: Optional[np.ndarray],
    model_name: str,
    pipe: Pipeline,
    param_dist: Dict[str, Any],
    inner_folds: int,
    search_iters: int,
    random_state: int,
    scoring: str,
    n_jobs: int,
    outdir: str,
    groups: Optional[np.ndarray] = None,
    cv_group_mode: str = "none",
    beta_threshold: float = 0.6,
    use_sample_weight: str = "none",
    base_sample_weight: Optional[np.ndarray] = None,
    sample_weight_report: Optional[Dict[str, Any]] = None,
):
    inner = _inner_splitter(cv_group_mode, groups, inner_folds, random_state + 999)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=search_iters,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=inner,
        refit=True,
        verbose=0,
        random_state=random_state + 1001,
    )

    fit_params: Dict[str, Any] = {}
    sw_all = None
    if base_sample_weight is not None:
        sw_all = np.asarray(base_sample_weight, dtype=np.float64).copy()
    if use_sample_weight == "balanced":
        sw_bal_all = compute_sample_weight(class_weight="balanced", y=y_enc).astype(np.float64)
        sw_all = sw_bal_all if sw_all is None else sw_all * sw_bal_all
    if sw_all is not None:
        fit_params["clf__sample_weight"] = sw_all

    if isinstance(inner, StratifiedGroupKFold) and groups is not None:
        search.fit(X_bin, y_enc, groups=groups, **fit_params)
    else:
        search.fit(X_bin, y_enc, **fit_params)

    best_pipe: Pipeline = search.best_estimator_
    best_params = search.best_params_

    # Extract selected features.  The CV/search pipeline applies TopK before masking,
    # but the exported inference bundle is mask -> clf on selected features only.
    select_mask = best_pipe.named_steps["select_mask"]
    sel_idx = np.asarray(select_mask.indices_, dtype=int)
    features_sel = [features_full[i] for i in sel_idx.tolist()]
    X_sel = X_bin[:, sel_idx]

    # Build final pipeline WITHOUT selector, trained on selected features only.
    # select_mask.masker_ was already constructed in the selected-feature space,
    # so predefined mask indices and external weighted_mcar weights must be copied
    # directly rather than mapped a second time against full-feature indices.
    final_masker: Masker = select_mask.masker_
    chr_sel = chr_arr[sel_idx] if chr_arr is not None else None
    pos_sel = pos_arr[sel_idx] if pos_arr is not None else None

    mfi = None
    if final_masker.mask_feature_indices is not None:
        mfi = np.asarray(final_masker.mask_feature_indices, dtype=np.int32).copy()
        if mfi.size and (mfi.min() < 0 or mfi.max() >= len(features_sel)):
            raise ValueError("Internal error: selected-space predefined mask indices are out of bounds.")

    w_sel = None
    if getattr(final_masker, "feature_weights", None) is not None:
        w = np.asarray(final_masker.feature_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != len(features_sel):
            raise ValueError("Internal error: selected-space mask_feature_weights length does not match selected features.")
        w_sel = w.copy()

    masker2 = Masker(
        mask_alg=final_masker.mask_alg,
        mask_rate=final_masker.mask_rate,
        mask_on=final_masker.mask_on,
        random_state=final_masker.random_state,
        chr_arr=chr_sel,
        pos_arr=pos_sel,
        mask_feature_indices=mfi,
        block_size_bp=final_masker.block_size_bp,
        beta_kappa=getattr(final_masker, "beta_kappa", 20.0),
        beta_alpha=getattr(final_masker, "beta_alpha", -1.0),
        beta_beta=getattr(final_masker, "beta_beta", -1.0),
        feature_weights=w_sel,
        weight_power=getattr(final_masker, "weight_power", 1.0),
        weight_eps=getattr(final_masker, "weight_eps", 1e-3),
    )
    clf2 = build_model(model_name, random_state=random_state)
    final_pipe = Pipeline([("mask", masker2), ("clf", clf2)])

    # Transfer best clf params
    clf_params = {k: v for k, v in best_params.items() if k.startswith("clf__")}
    if clf_params:
        final_pipe.set_params(**clf_params)

    final_fit_params: Dict[str, Any] = {}
    sw_all2 = None
    if base_sample_weight is not None:
        sw_all2 = np.asarray(base_sample_weight, dtype=np.float64).copy()
    if use_sample_weight == "balanced":
        sw_bal_all2 = compute_sample_weight(class_weight="balanced", y=y_enc).astype(np.float64)
        sw_all2 = sw_bal_all2 if sw_all2 is None else sw_all2 * sw_bal_all2
    if sw_all2 is not None:
        final_fit_params["clf__sample_weight"] = sw_all2
    final_pipe.fit(X_sel, y_enc, **final_fit_params)

    bundle = {
        "model": final_pipe,
        "label_encoder_classes": class_names,
        "feature_names": features_sel,
        "beta_threshold": float(beta_threshold),
        "best_params": best_params,
        "sample_weight_report": sample_weight_report,
    }
    joblib.dump(bundle, os.path.join(outdir, "final_model_bundle.joblib"))
    with open(os.path.join(outdir, "final_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    return bundle


def main():
    ap = argparse.ArgumentParser(description="Train the MBMMC Random Forest methylation classifier with fold-local feature selection and masking.")
    ap.add_argument("--version", action="version", version="MBMMC 1.0.0")
    ap.add_argument("--ref_csv", required=True)
    ap.add_argument("--meta", required=True, help="meta.txt with Sample and Types (optional Platform/Material)")
    ap.add_argument("--exclude_types", type=str, default="", help="Comma-separated Types labels to exclude (e.g., Control)")

    ap.add_argument("--model", type=str, default="rf", choices=["rf"], help="Publication model; only Random Forest is retained in this repository.")
    ap.add_argument("--beta_threshold", type=float, default=0.6)
    ap.add_argument("--equal_mode", type=str, default="negative", choices=["positive","negative"])

    ap.add_argument("--feature_select", type=str, default="variance_topk",
                    choices=["none","variance_topk","f_classif_topk","pairwise_score_topk"])
    ap.add_argument("--topk", type=int, default=50000)
    ap.add_argument("--corr_filter", type=float, default=0.0,
                    help="Absolute Pearson correlation threshold for redundancy filtering after TopK (0 disables). Recommended 0.90-0.98.")

    ap.add_argument("--mask_alg", type=str, default="mcar",
                    choices=["mcar","mcar_beta_rate","weighted_mcar","chrom_block","predefined"])
    ap.add_argument("--mask_rate", type=float, default=0.90, help="Fraction of entries/features masked to 0 during training")
    ap.add_argument("--mask_on", type=str, default="train", choices=["train","both","none"])

    ap.add_argument("--mask_beta_kappa", type=float, default=20.0,
                    help="For mcar_beta_rate: concentration (larger => rates closer to mask_rate).") ##kappa跟mask_rate(均值μ)生成α和β, k越小方差越大，缺失分布越分散，反之缺失率越集中;如果设置了α和β，则不需要设置kappa;
    ap.add_argument("--mask_beta_alpha", type=float, default=-1.0,
                    help="For mcar_beta_rate: override alpha (>0). If set, also set --mask_beta_beta.")
    ap.add_argument("--mask_beta_beta", type=float, default=-1.0,
                    help="For mcar_beta_rate: override beta (>0).")
    ap.add_argument("--mask_weight_power", type=float, default=1.0,
                    help="For weighted_mcar: weight exponent applied to inferred weights.")
    ap.add_argument("--mask_weight_eps", type=float, default=1e-3,
                    help="For weighted_mcar: epsilon added to weights.")
    ap.add_argument("--mask_feature_weights", type=str, default="",
                    help="Optional: TSV with columns chr_pos and weight for weighted_mcar.")

    ap.add_argument("--mask_feature_list", type=str, default="", help="Required if mask_alg=predefined. One chr_pos per line.")
    ap.add_argument("--block_size_bp", type=int, default=500_000, help="For chrom_block")

    ap.add_argument("--prefilter_min_present_rate", type=float, default=0.0,
                    help="Remove features present in < this fraction of samples")
    ap.add_argument("--prefilter_platform_missing_diff", type=float, default=0.0,
                    help="Remove features with abs(missing_rate_platform_diff) > this")

    ap.add_argument("--cv_mode", type=str, default="nested", choices=["nested","cv","none"])
    ap.add_argument("--outer_folds", type=int, default=3)
    ap.add_argument("--inner_folds", type=int, default=2)
    ap.add_argument("--search_iters", type=int, default=40)
    ap.add_argument("--scoring", type=str, default="f1_macro", choices=["f1_macro","balanced_accuracy","accuracy"])
    ap.add_argument("--use_sample_weight", type=str, default="none", choices=["none","balanced"],
                    help="If balanced, multiply sample weights by class-balanced weights to mitigate class imbalance.")
    ap.add_argument("--sample_weight_col", type=str, default="",
                    help="Optional metadata column containing per-sample training weights. Recommended for real+in-silico merged training: RecommendedTrainingWeight.")
    ap.add_argument("--auto_sample_weight", action="store_true",
                    help="Infer real/synthetic/tumor-fraction weights from metadata when sample_weight_col is absent.")
    ap.add_argument("--synthetic_prefix", type=str, default="SIM_", help="Sample-name prefix used to infer synthetic samples.")
    ap.add_argument("--real_weight", type=float, default=1.0)
    ap.add_argument("--synthetic_weight", type=float, default=0.35)
    ap.add_argument("--low_tf_weight", type=float, default=0.10,
                    help="Weight for synthetic tumor samples below decision_tf_weight_boundary; use mainly for no-call calibration.")
    ap.add_argument("--boundary_tf_weight", type=float, default=0.45,
                    help="Weight for synthetic samples near the clinical decision boundary, default 10-20%% tumor fraction.")
    ap.add_argument("--high_tf_weight", type=float, default=0.35,
                    help="Weight for synthetic tumor samples >=20%% tumor fraction.")
    ap.add_argument("--decision_tf_weight_boundary", type=float, default=0.10,
                    help="Tumor-fraction boundary below which supervised positive labels should be downweighted.")
    ap.add_argument("--platform_weight_map", type=str, default="",
                    help="Optional multiplicative platform factors, e.g. TAPS:0.8,WGBS:0.6,ONT:0.5")
    ap.add_argument("--sample_regex_weight_map", type=str, default="",
                    help="Optional multiplicative regex factors, e.g. '.*lowtf.*:0.2,.*artifact.*:0.1'")

    ap.add_argument("--cv_group_mode", type=str, default="none", choices=["none","batch","logo"],
                    help="Internal CV split mode: none=StratifiedKFold; batch=StratifiedGroupKFold; logo=LeaveOneGroupOut. Use --cv_group_col for leakage-safe custom groups.")
    ap.add_argument("--cv_group_col", type=str, default="",
                    help="Optional metadata column used as CV group labels when cv_group_mode != none, e.g. SplitGroup or LatentID. Missing values fall back to Sample.")
    ap.add_argument("--group_by", type=str, default="platform_material", choices=["platform","material","platform_material"],
                    help="Legacy group constructor used only when --cv_group_col is empty, and for drop_platform_markers grouping.")
    ap.add_argument("--drop_platform_markers",type=int,default=0,help="Drop top-N CpGs most associated with platform/group (unsupervised wrt tumor). Score = max(mean_by_group) - min(mean_by_group) on X_bin. 0 disables")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--outdir", type=str, default="outputs/train_ml")

    args = ap.parse_args()
    _ensure_outdir(args.outdir)

    exclude_types = [x.strip() for x in args.exclude_types.split(",") if x.strip()]
    ref = load_reference(args.ref_csv, args.meta, exclude_types=exclude_types)

    # prefilter (unsupervised; uses missingness only)
    X_beta, features, chr_arr, pos_arr = _prefilter_features_by_presence(
        ref.X_beta, ref.features, ref.chr_arr, ref.pos_arr, args.prefilter_min_present_rate
    )
    X_beta, features, chr_arr, pos_arr = _prefilter_features_by_platform_missing_diff(
        X_beta, features, chr_arr, pos_arr, ref.platform, args.prefilter_platform_missing_diff
    )

    # binarize reference beta to {-1,0,1}
    print(f"二值化前数据集X_beta:", ref.X_beta.shape, "len(features):", len(ref.features))
    binarizer = BetaBinarizer(threshold=args.beta_threshold, equal_mode=args.equal_mode)
    X_bin = binarizer.transform(X_beta).astype(np.int8)
    print("二值化后数据集X_bin:", X_bin.shape)

    # Remove constant features (optional but helps)
    # constant means all values same across samples (after binarization)
    col_min = X_bin.min(axis=0)
    col_max = X_bin.max(axis=0)
    non_const = (col_min != col_max)
    X_bin = X_bin[:, non_const]
    features = [f for f, keep in zip(features, non_const.tolist()) if keep]
    chr_arr = chr_arr[non_const] if chr_arr is not None else None
    pos_arr = pos_arr[non_const] if pos_arr is not None else None
    print(f"去除常值后的X_bin:", X_bin.shape, "len(features):", len(features))
    if chr_arr is not None:                                    
        assert chr_arr.shape[0] == X_bin.shape[1]
    if pos_arr is not None:                                   
        assert pos_arr.shape[0] == X_bin.shape[1]
    assert len(features) == X_bin.shape[1]                    

    le = LabelEncoder()
    y_enc = le.fit_transform(ref.y)
    class_names = le.classes_.tolist()

    meta_weights = read_meta_for_weights(args.meta, ref.samples)
    base_sample_weight, sample_weight_report = resolve_sample_weight_vector(
        meta=meta_weights,
        samples=ref.samples,
        y=y_enc,
        sample_weight_col=args.sample_weight_col,
        platform_weight_map=args.platform_weight_map,
        sample_regex_weight_map=args.sample_regex_weight_map,
        synthetic_prefix=args.synthetic_prefix,
        synthetic_weight=args.synthetic_weight,
        real_weight=args.real_weight,
        low_tf_weight=args.low_tf_weight,
        boundary_tf_weight=args.boundary_tf_weight,
        high_tf_weight=args.high_tf_weight,
        decision_tf=args.decision_tf_weight_boundary,
        auto_weight_synthetic=args.auto_sample_weight,
        use_class_balance=False,
    )
    if args.sample_weight_col or args.auto_sample_weight or args.platform_weight_map or args.sample_regex_weight_map:
        write_sample_weight_audit(args.outdir, ref.samples, base_sample_weight, sample_weight_report, meta_weights)
        print(f"[sample_weights] enabled: min={base_sample_weight.min():.4g}, median={np.median(base_sample_weight):.4g}, max={base_sample_weight.max():.4g}, synthetic={sample_weight_report.get('n_synthetic_inferred')}")
    else:
        base_sample_weight = None
        sample_weight_report = None

    # groups for internal CV (leakage control) and platform-marker dropping (technical marker control).
    # Do not reuse SplitGroup/LatentID groups for drop_platform_markers; that filter is intended
    # to remove platform/material-associated features, not sample/latent-specific features.
    groups = _resolve_cv_groups(
        cv_group_mode=args.cv_group_mode,
        cv_group_col=args.cv_group_col,
        meta_aligned=meta_weights,
        samples=ref.samples,
        group_by=args.group_by,
        platform=ref.platform,
        material=ref.material,
    )
    marker_groups = None
    if args.drop_platform_markers > 0:
        marker_groups = _make_groups(args.group_by, ref.platform, ref.material)
        X_bin, features, chr_arr, pos_arr = _apply_drop_platform_markers(
            X_bin=X_bin,
            features=features,
            chr_arr=chr_arr,
            pos_arr=pos_arr,
            groups=marker_groups,
            drop_n=args.drop_platform_markers,
            outdir=args.outdir,
        )
    if groups is not None:
        uniq_g = np.unique(groups)
        label = args.cv_group_col if args.cv_group_col else args.group_by
        print(f"[CV Groups] source={label}, n_unique_groups={len(uniq_g)}")
        per_cls = {}
        for c in np.unique(y_enc):
            per_cls[int(c)] = len(np.unique(groups[y_enc == c]))
        print("[CV Groups] n_groups_per_class:", per_cls)
        min_g = min(per_cls.values())
        print(f"[CV Groups] min_groups_per_class={min_g} (outer_folds should be <= this for StratifiedGroupKFold)")

    mask_feature_indices = None
    if args.mask_alg == "predefined":
        if not args.mask_feature_list:
            raise ValueError("--mask_feature_list is required for mask_alg=predefined")
        mask_feature_indices = _load_mask_feature_list(args.mask_feature_list, features)

    mask_feature_weights = None
    if args.mask_alg == "weighted_mcar" and args.mask_feature_weights:
        mask_feature_weights = _load_feature_weights(args.mask_feature_weights, features)

    # search pipeline: select + mask + clf (binarization already done outside)
    # The selector stays inside the sklearn pipeline to avoid CV leakage.
    pipe = build_search_pipeline(
        model_name=args.model,
        mask_alg=args.mask_alg,
        mask_rate=args.mask_rate,
        mask_on=args.mask_on,
        chr_arr=chr_arr,
        pos_arr=pos_arr,
        mask_feature_indices=mask_feature_indices,
        block_size_bp=args.block_size_bp,
        mask_beta_kappa=args.mask_beta_kappa,
        mask_beta_alpha=args.mask_beta_alpha,
        mask_beta_beta=args.mask_beta_beta,
        mask_weight_power=args.mask_weight_power,
        mask_weight_eps=args.mask_weight_eps,
        mask_feature_weights=mask_feature_weights,
        feature_select=args.feature_select,
        topk=args.topk,
        corr_filter=args.corr_filter,
        random_state=args.random_state,
    )
    param_dist = param_distributions(args.model)

    # CV
    if args.cv_mode in ("nested","cv"):
        run_cv(
            X_bin=X_bin,
            y_enc=y_enc,
            class_names=class_names,
            pipe=pipe,
            param_dist=param_dist,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            search_iters=args.search_iters,
            random_state=args.random_state,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            outdir=args.outdir,
            groups=groups,
            cv_group_mode=args.cv_group_mode,
            use_sample_weight=args.use_sample_weight,
            base_sample_weight=base_sample_weight,
        )

    bundle = fit_final_and_export_bundle(
        X_bin=X_bin,
        y_enc=y_enc,
        class_names=class_names,
        features_full=features,
        chr_arr=chr_arr,
        pos_arr=pos_arr,
        model_name=args.model,
        pipe=pipe,
        param_dist=param_dist,
        inner_folds=args.inner_folds,
        search_iters=args.search_iters,
        random_state=args.random_state,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        outdir=args.outdir,
        groups=groups,
        cv_group_mode=args.cv_group_mode,
        beta_threshold=args.beta_threshold,
        use_sample_weight=args.use_sample_weight,
        base_sample_weight=base_sample_weight,
        sample_weight_report=sample_weight_report,
    )

    # save features.txt for convenience
    with open(os.path.join(args.outdir, "features.txt"), "w", encoding="utf-8") as f:
        for feat in bundle["feature_names"]:
            f.write(feat + "\n")

    # config
    config = vars(args)
    config["n_samples"] = int(X_bin.shape[0])
    config["n_features_used"] = int(len(bundle["feature_names"]))
    config["class_names"] = class_names
    config["cv_group_col_effective"] = str(args.cv_group_col or "")
    config["rf_pipeline_order"] = "binarize -> remove_constant -> feature_select_topk -> mask -> clf"
    config["fold_local_constant_filter_before_topk"] = True
    config["suppress_expected_f_classif_constant_warnings"] = True
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Saved final model bundle:", os.path.join(args.outdir, "final_model_bundle.joblib"))


if __name__ == "__main__":
    main()
