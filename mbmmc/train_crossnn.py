"""MBMMC independent implementation informed by the crossNN methodology.

Method reference:
Yuan D, Jugas R, Pokorna P, et al. Nature Cancer. 2025;6:1283–1294.
doi:10.1038/s43018-025-00976-5

This file is not an official release of the original crossNN authors.
See THIRD_PARTY_NOTICES.md and docs/CROSSNN_METHOD_PROVENANCE.md.
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from .utils.data_io import load_reference
from .utils.sample_weights import read_meta_for_weights, resolve_sample_weight_vector, write_sample_weight_audit
from .utils.preprocessing import BetaBinarizer, FeatureSelector
from .utils.crossnn import CrossNNLinear, make_mask, class_weights_from_labels, predict_proba, set_seed
from .utils.metrics import compute_metrics, confusion_matrix_norm
from .utils.plots import plot_metrics_bar, plot_confusion_matrix, plot_macro_roc


def _parse_csv_floats(s: str) -> List[float]:
    items: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        items.append(float(x))
    return items


def _parse_csv_ints(s: str) -> List[int]:
    items: List[int] = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        items.append(int(float(x)))
    return items




def _make_groups(group_by: str, platform, material) -> np.ndarray:
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

    Missing/empty values are replaced by Sample so real array samples without
    simulated LatentID/SplitGroup values remain independent groups.
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
    platform,
    material,
) -> Optional[np.ndarray]:
    if cv_group_mode == "none":
        return None
    if cv_group_col:
        return _groups_from_meta_column(meta_aligned, samples, cv_group_col)
    return _make_groups(group_by, platform, material)


def _make_splitter(cv_group_mode: str, cv_folds: int, random_state: int):
    """
    cv_group_mode:
      - "none": StratifiedKFold
      - "batch": StratifiedGroupKFold (uses groups)
      - "logo": LeaveOneGroupOut (uses groups)
    """
    if cv_group_mode == "none":
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    if cv_group_mode == "batch":
        return StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    if cv_group_mode == "logo":
        return LeaveOneGroupOut()
    raise ValueError(f"Unknown cv_group_mode: {cv_group_mode}")


def make_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)


def train_one_fold(
    X_int8: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    keep_fraction: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    seed: int,
    patience: Optional[int] = None,
    max_epochs: Optional[int] = None,
    eval_keep_fraction: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
):
    set_seed(seed)
    if max_epochs is None:
        max_epochs = epochs
    if eval_keep_fraction is None:
        eval_keep_fraction = keep_fraction

    X_tr = torch.tensor(X_int8[train_idx].astype(np.float32), device=device)
    y_tr = torch.tensor(y_enc[train_idx].astype(np.int64), device=device)
    if sample_weight is None:
        w_tr = torch.ones_like(y_tr, dtype=torch.float32, device=device)
    else:
        w_tr = torch.tensor(np.asarray(sample_weight, dtype=np.float32)[train_idx], device=device)
    X_va = torch.tensor(X_int8[val_idx].astype(np.float32), device=device)

    model = CrossNNLinear(n_features=X_tr.shape[1], n_classes=n_classes).to(device)

    # class weights
    cw = class_weights_from_labels(y_enc[train_idx]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw, reduction="none")

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n = X_tr.shape[0]
    indices = torch.arange(n, device=device)

    best_val = -1.0
    best_state = None
    wait = 0
    losses = []

    for _ep in range(1, max_epochs + 1):
        model.train()
        perm = indices[torch.randperm(n)]
        epoch_loss = 0.0
        nb = 0

        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            xb = X_tr[bi]
            yb = y_tr[bi]
            wb = w_tr[bi]

            m = make_mask(xb, keep_fraction=keep_fraction)
            opt.zero_grad(set_to_none=True)
            logits = model(xb * m)
            loss_vec = loss_fn(logits, yb)
            loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().cpu().item())
            nb += 1

        epoch_loss /= max(1, nb)
        losses.append(epoch_loss)

        # validation
        model.eval()
        with torch.no_grad():
            mva = make_mask(X_va, keep_fraction=eval_keep_fraction)
            proba = predict_proba(model, X_va * mva).cpu().numpy()
            y_pred = proba.argmax(axis=1)
            m_dict = compute_metrics(
                y_true=y_enc[val_idx],
                y_pred=y_pred,
                y_proba=proba,
                labels=list(range(n_classes)),
            )
            score = float(m_dict.get("balanced_accuracy", float("nan")))

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if patience is not None and wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, np.array(losses, dtype=np.float32)


def cv_crossnn(
    X_int8: np.ndarray,
    y_enc: np.ndarray,
    class_labels: List[str],
    groups: Optional[np.ndarray],
    cv_group_mode: str,
    keep_fraction: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    cv_folds: int,
    random_state: int,
    device: str,
    outdir: str,
    patience: Optional[int] = None,
    max_epochs: Optional[int] = None,
    eval_keep_fraction: Optional[float] = None,
    # Leak-safe CV feature selection (fit selector on TRAIN folds only)
    feature_select: str = "none",
    topk: int = 50000,
    corr_filter: float = 0.0,
    fs_random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
):
    splitter = _make_splitter(cv_group_mode=cv_group_mode, cv_folds=cv_folds, random_state=random_state)
    make_outdir(outdir)

    if cv_group_mode == "none":
        split_iter = splitter.split(X_int8, y_enc)
    else:
        if groups is None:
            raise ValueError("groups is required when cv_group_mode != none")
        split_iter = splitter.split(X_int8, y_enc, groups)

    fold_rows = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (tr, va) in enumerate(split_iter, start=1):
        # Optional fold-wise feature selection to avoid leakage
        if str(feature_select) != "none":
            fs = FeatureSelector(
                method=str(feature_select),
                topk=int(topk),
                random_state=int(fs_random_state) + int(fold),
                corr_filter=float(corr_filter),
            )
            X_tr_sel = fs.fit_transform(X_int8[tr], y_enc[tr]).astype(np.int8)
            idx = getattr(fs, "indices_", None)
            if idx is None:
                idx = np.arange(X_int8.shape[1])
            idx = np.asarray(idx, dtype=int)
            X_va_sel = X_int8[va][:, idx].astype(np.int8)

            # Build fold-local arrays for train_one_fold() (keeps original API)
            X_fold = np.vstack([X_tr_sel, X_va_sel]).astype(np.int8, copy=False)
            y_fold = np.concatenate([y_enc[tr], y_enc[va]]).astype(np.int64, copy=False)
            weight_fold = None if sample_weight is None else np.concatenate([np.asarray(sample_weight)[tr], np.asarray(sample_weight)[va]]).astype(np.float32, copy=False)
            tr_idx = np.arange(X_tr_sel.shape[0], dtype=int)
            va_idx = np.arange(X_tr_sel.shape[0], X_tr_sel.shape[0] + X_va_sel.shape[0], dtype=int)
            n_features_fold = int(X_tr_sel.shape[1])
            X_va_eval = X_va_sel
        else:
            X_fold = X_int8
            y_fold = y_enc
            tr_idx = tr
            va_idx = va
            n_features_fold = int(X_int8.shape[1])
            X_va_eval = X_int8[va]
            weight_fold = sample_weight

        model, losses = train_one_fold(
            X_int8=X_fold,
            y_enc=y_fold,
            n_classes=len(class_labels),
            train_idx=tr_idx,
            val_idx=va_idx,
            keep_fraction=keep_fraction,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            device=device,
            seed=random_state + fold,
            patience=patience,
            max_epochs=max_epochs,
            eval_keep_fraction=eval_keep_fraction,
            sample_weight=weight_fold,
        )

        X_va_t = torch.tensor(X_va_eval.astype(np.float32), device=device)
        with torch.no_grad():
            mva = make_mask(X_va_t, keep_fraction=eval_keep_fraction if eval_keep_fraction is not None else keep_fraction)
            proba = predict_proba(model, X_va_t * mva).cpu().numpy()
            y_pred = proba.argmax(axis=1)

        m = compute_metrics(
            y_true=y_enc[va],
            y_pred=y_pred,
            y_proba=proba,
            labels=list(range(len(class_labels))),
        )
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(va)),
            "n_features": int(n_features_fold),
            **m,
        })

        all_y_true.extend(y_enc[va].tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.append(proba)

        loss_path = os.path.join(outdir, f"loss_fold{fold}.csv")
        pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(loss_path, index=False)

    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(os.path.join(outdir, "cv_metrics.csv"), index=False)

    summary = {c: {"mean": float(metrics_df[c].mean()), "std": float(metrics_df[c].std(ddof=1))}
               for c in metrics_df.columns if c not in ("fold", "n_train", "n_test")}
    with open(os.path.join(outdir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_metrics_bar(metrics_df, os.path.join(outdir, "plots", "metrics_bar.png"))

    all_y_true_arr = np.array(all_y_true, dtype=int)
    all_y_pred_arr = np.array(all_y_pred, dtype=int)
    y_proba_agg = np.vstack(all_y_proba)

    cmn = confusion_matrix_norm(all_y_true_arr, all_y_pred_arr, labels=list(range(len(class_labels))))
    np.savetxt(os.path.join(outdir, "plots", "confusion_matrix.txt"),cmn,header="\t".join(class_labels),delimiter="\t", fmt="%.2f")
    plot_confusion_matrix(cmn, class_labels, os.path.join(outdir, "plots", "confusion_matrix.png"))
    plot_macro_roc(all_y_true_arr, y_proba_agg, list(range(len(class_labels))),
                   os.path.join(outdir, "plots", "roc_macro.png"),
                   title="OvR ROC curves", class_names=class_labels)

    return metrics_df, summary

def train_final_model(
    X_int8: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    keep_fraction: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    seed: int,
    max_epochs: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
):
    set_seed(seed)
    X = torch.tensor(X_int8.astype(np.float32), device=device)
    y = torch.tensor(y_enc.astype(np.int64), device=device)
    if sample_weight is None:
        w = torch.ones_like(y, dtype=torch.float32, device=device)
    else:
        w = torch.tensor(np.asarray(sample_weight, dtype=np.float32), device=device)

    model = CrossNNLinear(n_features=X.shape[1], n_classes=n_classes).to(device)
    cw = class_weights_from_labels(y_enc).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw, reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n = X.shape[0]
    indices = torch.arange(n, device=device)
    losses = []
    total_epochs = int(max_epochs or epochs)

    for _ep in range(1, total_epochs + 1):
        model.train()
        perm = indices[torch.randperm(n)]
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            xb = X[bi]
            yb = y[bi]
            wb = w[bi]
            m = make_mask(xb, keep_fraction=keep_fraction)
            opt.zero_grad(set_to_none=True)
            logits = model(xb * m)
            loss_vec = loss_fn(logits, yb)
            loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu().item())
            nb += 1
        epoch_loss /= max(1, nb)
        losses.append(epoch_loss)

    return model, np.array(losses, dtype=np.float32)


def random_search_tune(
    X_int8: np.ndarray,
    y_enc: np.ndarray,
    class_labels: List[str],
    groups: Optional[np.ndarray],
    cv_group_mode: str,
    trials: int,
    cv_folds: int,
    random_state: int,
    device: str,
    outdir: str,
    max_epochs: int,
    patience: int,
    fixed_keep_fraction: Optional[float] = None,
    tune_metric: str = "macro_f1",
    eval_keep_fraction: float = 1.0,
    tune_keep_grid: Optional[List[float]] = None,
    tune_epoch_grid: Optional[List[int]] = None,
    lr_range: Tuple[float, float] = (3e-5, 3e-4),
    wd_range: Tuple[float, float] = (1e-7, 1e-4),
    batch_size_options: Optional[List[int]] = None,
    # Leak-safe CV feature selection params (passed into cv_crossnn)
    feature_select: str = "none",
    topk: int = 50000,
    corr_filter: float = 0.0,
    fs_random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
):
    """
    Paper-guided tuning for crossNN.

    Stage 1 (grid): tune the two most *critical* hyperparameters first:
      - keep_fraction (the Bernoulli keep probability; 1-keep_fraction is masking rate)
      - epochs (training epochs)

    Stage 2 (random): then tune optimizer-related hyperparameters in a *limited* way:
      - learning_rate
      - weight_decay
      - batch_size

    Notes
    -----
    - By default, validation/evaluation uses eval_keep_fraction=1.0 (no extra mask),
      which matches the common inference setup (missing CpGs are already encoded as 0).
    - If you want paper-style MC inference with extra Bernoulli masking at inference,
      set eval_keep_fraction to the training keep_fraction instead.
    """
    rng = np.random.RandomState(int(random_state))

    os.makedirs(outdir, exist_ok=True)

    # Default (paper-inspired) candidate grids
    paper_keep_grid = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.10]
    if fixed_keep_fraction is not None:
        keep_grid = [float(fixed_keep_fraction)]
    elif tune_keep_grid:
        keep_grid = [float(x) for x in tune_keep_grid]
    else:
        keep_grid = paper_keep_grid

    # epochs grid
    if tune_epoch_grid:
        epoch_grid = [int(e) for e in tune_epoch_grid if 1 <= int(e) <= int(max_epochs)]
        if not epoch_grid:
            epoch_grid = [int(max_epochs)]
    else:
        epoch_candidates = [1000, 800, 400, 1200, 1500, 2000, 3000, int(max_epochs)]
        epoch_grid = [int(e) for e in epoch_candidates if 1 <= int(e) <= int(max_epochs)]
        if not epoch_grid:
            epoch_grid = [int(max_epochs)]

    # Baseline optimizer settings (paper preset defaults)
    base_lr = 1e-4
    base_wd = 1e-5
    base_bs = 16

    # Schedule: prioritize paper-default (0.0025, 1000) early.
    grid_trials: List[Tuple[float, int]] = []
    for k in keep_grid:
        for e in epoch_grid:
            grid_trials.append((float(k), int(e)))
    grid_trials.sort(key=lambda ke: (abs(np.log10(ke[0]) - np.log10(0.0025)), abs(ke[1] - 1000)))

    if not batch_size_options:
        batch_size_options = [8, 16, 32]

    lr_lo, lr_hi = float(lr_range[0]), float(lr_range[1])
    wd_lo, wd_hi = float(wd_range[0]), float(wd_range[1])

    def sample_optimizer():
        lr = float(10 ** rng.uniform(np.log10(lr_lo), np.log10(lr_hi)))
        wd = float(10 ** rng.uniform(np.log10(wd_lo), np.log10(wd_hi)))
        bs = int(rng.choice(batch_size_options))
        return lr, wd, bs

    def metric_score(summary: Dict[str, Any]) -> Tuple[float, float]:
        # summary is {metric: {"mean":..., "std":...}, ...}
        metric_mean = float(summary.get(tune_metric, {}).get("mean", float("nan")))
        if not np.isfinite(metric_mean):
            metric_mean = float(summary.get("balanced_accuracy", {}).get("mean", float("nan")))
        score = -metric_mean if tune_metric == "log_loss" else metric_mean
        return metric_mean, float(score)

    best: Optional[Dict[str, Any]] = None
    trials_rows: List[Dict[str, Any]] = []

    t = 0

    # --- Stage 1: grid search over (keep_fraction, epochs) ---
    for (keep_fraction, epochs) in grid_trials:
        if t >= trials:
            break
        t += 1

        trial_dir = os.path.join(outdir, f"trial_{t:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        _metrics_df, summary = cv_crossnn(
            X_int8=X_int8,
            y_enc=y_enc,
            class_labels=class_labels,
            groups=groups,
            cv_group_mode=cv_group_mode,
            keep_fraction=float(keep_fraction),
            epochs=int(epochs),
            learning_rate=float(base_lr),
            weight_decay=float(base_wd),
            batch_size=int(base_bs),
            cv_folds=cv_folds,
            random_state=random_state + t,
            device=device,
            outdir=trial_dir,
            patience=None,
            max_epochs=None,
            eval_keep_fraction=float(eval_keep_fraction),
            feature_select=str(feature_select),
            topk=int(topk),
            corr_filter=float(corr_filter),
            fs_random_state=int(fs_random_state),
            sample_weight=sample_weight,
        )

        metric_mean, score = metric_score(summary)

        row = {
            "trial": t,
            "stage": "grid",
            "metric": str(tune_metric),
            "metric_mean": float(metric_mean),
            "score": float(score),
            "keep_fraction": float(keep_fraction),
            "mask_rate": float(1.0 - float(keep_fraction)),
            "epochs": int(epochs),
            "learning_rate": float(base_lr),
            "weight_decay": float(base_wd),
            "batch_size": int(base_bs),
            "eval_keep_fraction": float(eval_keep_fraction),
        }
        trials_rows.append(row)
        pd.DataFrame(trials_rows).to_csv(os.path.join(outdir, "tuning_trials.csv"), index=False)

        if best is None or score > float(best["score"]):
            best = {"score": float(score), "params": row}

    if best is None:
        raise RuntimeError("Tuning failed: no trials produced results")

    # --- Stage 2: limited optimizer tuning around the best (keep_fraction, epochs) ---
    while t < trials:
        t += 1
        keep_fraction = float(best["params"]["keep_fraction"])
        epochs = int(best["params"]["epochs"])
        lr, wd, bs = sample_optimizer()

        trial_dir = os.path.join(outdir, f"trial_{t:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        _metrics_df, summary = cv_crossnn(
            X_int8=X_int8,
            y_enc=y_enc,
            class_labels=class_labels,
            groups=groups,
            cv_group_mode=cv_group_mode,
            keep_fraction=keep_fraction,
            epochs=epochs,
            learning_rate=lr,
            weight_decay=wd,
            batch_size=bs,
            cv_folds=cv_folds,
            random_state=random_state + t,
            device=device,
            outdir=trial_dir,
            patience=None,
            max_epochs=None,
            eval_keep_fraction=float(eval_keep_fraction),
            feature_select=str(feature_select),
            topk=int(topk),
            corr_filter=float(corr_filter),
            fs_random_state=int(fs_random_state),
            sample_weight=sample_weight,
        )

        metric_mean, score = metric_score(summary)

        row = {
            "trial": t,
            "stage": "opt",
            "metric": str(tune_metric),
            "metric_mean": float(metric_mean),
            "score": float(score),
            "keep_fraction": keep_fraction,
            "mask_rate": float(1.0 - keep_fraction),
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "weight_decay": float(wd),
            "batch_size": int(bs),
            "eval_keep_fraction": float(eval_keep_fraction),
        }
        trials_rows.append(row)
        pd.DataFrame(trials_rows).to_csv(os.path.join(outdir, "tuning_trials.csv"), index=False)

        if score > float(best["score"]):
            best = {"score": float(score), "params": row}

    with open(os.path.join(outdir, "best_tuned_params.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    return best["params"]



def main():
    ap = argparse.ArgumentParser(description="Train the MBMMC leakage-controlled crossNN methylation classifier.")
    ap.add_argument("--version", action="version", version="MBMMC 1.0.0")

    ap.add_argument("--ref_csv", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--exclude_types", type=str, default="", help="Comma-separated Types labels to exclude (e.g., Control)")
    ap.add_argument("--beta_threshold", type=float, default=0.6)
    ap.add_argument("--equal_mode", type=str, default="positive", choices=["positive", "negative"])

    ap.add_argument("--feature_select", type=str, default="variance_topk",
                    choices=["none", "variance_topk", "f_classif_topk", "pairwise_score_topk"])
    ap.add_argument("--topk", type=int, default=200000)
    ap.add_argument("--corr_filter", type=float, default=0.0,
                    help="Absolute Pearson correlation threshold for redundancy filtering after TopK (0 disables).")

    ap.add_argument("--feature_select_cv_mode", type=str, default="foldwise",
                    choices=["foldwise", "global"],
                    help="Avoid CV leakage: foldwise fits FeatureSelector inside each CV fold (recommended). global reproduces legacy behavior and may leak when feature_select is supervised.")


    ap.add_argument("--preset", type=str, default="paper", choices=["paper", "tuned"])

    # paper-like params
    ap.add_argument("--mask_keep_fraction", type=float, default=0.0025, help="0.0025 => 99.75%% masked")
    ap.add_argument("--mask_rate", type=float, default=-1.0,
                    help="Alias: fraction masked. If set >=0, keep_fraction = 1-mask_rate (also fixes keep_fraction in tuned mode).")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=16)

    # CV and tuning
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--cv_group_mode", type=str, default="none", choices=["none", "batch", "logo"],
                    help="Internal CV split mode: none=StratifiedKFold; batch=StratifiedGroupKFold; logo=LeaveOneGroupOut. Use --cv_group_col for leakage-safe custom groups.")
    ap.add_argument("--cv_group_col", type=str, default="",
                    help="Optional metadata column used as CV group labels when cv_group_mode != none, e.g. SplitGroup or LatentID. Missing values fall back to Sample.")
    ap.add_argument("--group_by", type=str, default="platform_material",
                    choices=["platform", "material", "platform_material"],
                    help="Legacy group constructor used only when --cv_group_col is empty.")
    ap.add_argument("--tune_trials", type=int, default=25)
    ap.add_argument("--max_epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=50)

    # Tuning controls (used when --preset tuned)
    ap.add_argument("--tune_metric", type=str, default="macro_f1",
                    choices=["balanced_accuracy", "macro_f1", "macro_roc_auc_ovr", "log_loss"],
                    help="Metric to optimize during tuning (log_loss is minimized).")
    ap.add_argument("--tune_eval_keep_fraction", type=float, default=1.0,
                    help="Keep fraction used at evaluation time (validation) during tuning and final CV. 1.0 means no extra Bernoulli mask at inference.")
    ap.add_argument("--tune_keep_grid", type=str, default="",
                    help="Comma-separated keep_fraction candidates for stage-1 grid; empty uses paper grid (0.0025,0.005,0.01,0.025,0.05,0.10).")
    ap.add_argument("--tune_epoch_grid", type=str, default="",
                    help="Comma-separated epoch candidates for stage-1 grid; empty uses paper-inspired candidates clipped by --max_epochs.")
    ap.add_argument("--tune_lr_min", type=float, default=3e-5)
    ap.add_argument("--tune_lr_max", type=float, default=3e-4)
    ap.add_argument("--tune_wd_min", type=float, default=1e-7)
    ap.add_argument("--tune_wd_max", type=float, default=1e-4)
    ap.add_argument("--tune_batch_sizes", type=str, default="8,16,32")

    ap.add_argument("--sample_weight_col", type=str, default="",
                    help="Optional metadata column containing per-sample training weights. Recommended for merged real+in-silico training: RecommendedTrainingWeight.")
    ap.add_argument("--auto_sample_weight", action="store_true",
                    help="Infer real/synthetic/tumor-fraction weights from metadata when sample_weight_col is absent.")
    ap.add_argument("--synthetic_prefix", type=str, default="SIM_", help="Sample-name prefix used to infer synthetic samples.")
    ap.add_argument("--real_weight", type=float, default=1.0)
    ap.add_argument("--synthetic_weight", type=float, default=0.35)
    ap.add_argument("--low_tf_weight", type=float, default=0.10)
    ap.add_argument("--boundary_tf_weight", type=float, default=0.45)
    ap.add_argument("--high_tf_weight", type=float, default=0.35)
    ap.add_argument("--decision_tf_weight_boundary", type=float, default=0.10)
    ap.add_argument("--platform_weight_map", type=str, default="",
                    help="Optional multiplicative platform factors, e.g. TAPS:0.8,WGBS:0.6,ONT:0.5")
    ap.add_argument("--sample_regex_weight_map", type=str, default="",
                    help="Optional multiplicative regex factors, e.g. '.*lowtf.*:0.2,.*artifact.*:0.1'")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs/crossnn_run")

    args = ap.parse_args()
    make_outdir(args.outdir)

    exclude_types = [x.strip() for x in args.exclude_types.split(",") if x.strip()]
    ref = load_reference(args.ref_csv, args.meta, exclude_types=exclude_types)

    # Encode y
    le = LabelEncoder()
    y_enc = le.fit_transform(ref.y)
    class_labels = le.classes_.tolist()

    meta_weights = read_meta_for_weights(args.meta, ref.samples)
    sample_weight, sample_weight_report = resolve_sample_weight_vector(
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
        write_sample_weight_audit(args.outdir, ref.samples, sample_weight, sample_weight_report, meta_weights)
        print(f"[sample_weights] enabled: min={sample_weight.min():.4g}, median={np.median(sample_weight):.4g}, max={sample_weight.max():.4g}, synthetic={sample_weight_report.get('n_synthetic_inferred')}")
    else:
        sample_weight = None
        sample_weight_report = None

    groups = _resolve_cv_groups(
        cv_group_mode=args.cv_group_mode,
        cv_group_col=args.cv_group_col,
        meta_aligned=meta_weights,
        samples=ref.samples,
        group_by=args.group_by,
        platform=ref.platform,
        material=ref.material,
    )
    if groups is not None:
        uniq_g = np.unique(groups)
        label = args.cv_group_col if args.cv_group_col else args.group_by
        print(f"[CV Groups] source={label}, n_unique_groups={len(uniq_g)}")
        per_cls = {int(c): int(len(np.unique(groups[y_enc == c]))) for c in np.unique(y_enc)}
        print("[CV Groups] n_groups_per_class:", per_cls)

    # Binarize
    print(f"二值化前的X矩阵X_beta：{ref.X_beta.shape}")
    binarizer = BetaBinarizer(threshold=args.beta_threshold, equal_mode=args.equal_mode)
    X_bin = binarizer.fit_transform(ref.X_beta).astype(np.int8)
    print (f"二值化后的X矩阵X_bin：{X_bin.shape}")

    # Remove constant features (zero-variance after binarization; paper-compatible)
    col_min = X_bin.min(axis=0)
    col_max = X_bin.max(axis=0)
    non_const = (col_min != col_max)
    X_bin = X_bin[:, non_const]
    feat_names = [f for f, keep in zip(ref.features, non_const.tolist()) if keep]
    print(f"常数筛选后的X矩阵X_bin：{X_bin.shape}")

    # Feature selection
    # IMPORTANT: supervised feature selection (e.g., f_classif_topk / pairwise_score_topk) must be fit on
    # TRAINING folds only during CV/tuning to avoid data leakage. We therefore support:
    #   - foldwise (default): CV/tuning fits FeatureSelector inside each fold; final model is still fit on ALL data.
    #   - global: legacy behavior (fits once on all data BEFORE CV; may leak if supervised).
    selector_full = FeatureSelector(method=args.feature_select, topk=args.topk, random_state=args.random_state, corr_filter=args.corr_filter)
    X_sel = selector_full.fit_transform(X_bin, y_enc).astype(np.int8)
    selected_idx = getattr(selector_full, "indices_", None)
    if selected_idx is None:
        selected_idx = np.arange(X_bin.shape[1])
    selected_idx = np.asarray(selected_idx, dtype=int)
    selected_feat_names = [feat_names[i] for i in selected_idx]
    print(f"FeatureSelector后的X矩阵X_sel (for final model): {X_sel.shape}")

    # Matrices used for CV/tuning
    if args.feature_select_cv_mode == "foldwise" and args.feature_select != "none":
        X_cv = X_bin
        cv_feature_select = str(args.feature_select)
    else:
        X_cv = X_sel
        cv_feature_select = "none"

    with open(os.path.join(args.outdir, "preprocess.json"), "w", encoding="utf-8") as f:
        json.dump({
            "beta_threshold": args.beta_threshold,
            "equal_mode": args.equal_mode,
            "constant_feature_removed": int((~non_const).sum()),
            "feature_select": args.feature_select,
            "topk": args.topk,
            "corr_filter": args.corr_filter,
            "feature_select_cv_mode": str(args.feature_select_cv_mode),
            "n_features_after_constant": int(X_bin.shape[1]),
            "n_features_selected_full": int(X_sel.shape[1]),
            "cv_feature_select": str(cv_feature_select),
            "n_features_final": int(X_sel.shape[1]),
            "cv_group_mode": args.cv_group_mode,
            "cv_group_col": args.cv_group_col,
            "group_by": args.group_by,
            "tune_metric": args.tune_metric,
            "eval_keep_fraction": args.tune_eval_keep_fraction,
            "sample_weight_report": sample_weight_report,
        }, f, indent=2, ensure_ascii=False)

    # Resolve keep_fraction (paper and tuned both support mask_rate alias)
    keep_fraction_fixed = None
    if args.mask_rate >= 0:
        keep_fraction_fixed = max(0.0, min(1.0, 1.0 - float(args.mask_rate)))
    else:
        keep_fraction_fixed = float(args.mask_keep_fraction)

    if args.preset == "paper":
        keep_fraction = float(keep_fraction_fixed)
        lr = float(args.learning_rate)
        wd = float(args.weight_decay)
        bs = int(args.batch_size)
        epochs = int(args.epochs)
    else:
        tune_keep_grid = _parse_csv_floats(args.tune_keep_grid) if str(args.tune_keep_grid).strip() else None
        tune_epoch_grid = _parse_csv_ints(args.tune_epoch_grid) if str(args.tune_epoch_grid).strip() else None
        tune_batch_sizes = _parse_csv_ints(args.tune_batch_sizes) if str(args.tune_batch_sizes).strip() else None

        best_params = random_search_tune(
            X_int8=X_cv,
            y_enc=y_enc,
            class_labels=class_labels,
            groups=groups,
            cv_group_mode=args.cv_group_mode,
            trials=args.tune_trials,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            device=args.device,
            outdir=args.outdir,
            max_epochs=args.max_epochs,
            patience=args.patience,
            fixed_keep_fraction=(keep_fraction_fixed if args.mask_rate >= 0 else None),
            tune_metric=args.tune_metric,
            eval_keep_fraction=args.tune_eval_keep_fraction,
            tune_keep_grid=tune_keep_grid,
            tune_epoch_grid=tune_epoch_grid,
            lr_range=(args.tune_lr_min, args.tune_lr_max),
            wd_range=(args.tune_wd_min, args.tune_wd_max),
            batch_size_options=tune_batch_sizes,
            feature_select=str(cv_feature_select),
            topk=int(args.topk),
            corr_filter=float(args.corr_filter),
            fs_random_state=int(args.random_state),
            sample_weight=sample_weight,
        )
        keep_fraction = float(best_params["keep_fraction"])
        lr = float(best_params["learning_rate"])
        wd = float(best_params["weight_decay"])
        bs = int(best_params["batch_size"])
        epochs = int(best_params.get("epochs", args.max_epochs))

    # Final CV report with chosen params
    cv_dir = os.path.join(args.outdir, "final_cv")
    os.makedirs(cv_dir, exist_ok=True)
    metrics_df, summary = cv_crossnn(
        X_int8=X_cv,
        y_enc=y_enc,
        class_labels=class_labels,
        groups=groups,
        cv_group_mode=args.cv_group_mode,
        keep_fraction=keep_fraction,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        batch_size=bs,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        device=args.device,
        outdir=cv_dir,
        patience=None,
        max_epochs=None,
        eval_keep_fraction=args.tune_eval_keep_fraction,
        feature_select=str(cv_feature_select),
        topk=int(args.topk),
        corr_filter=float(args.corr_filter),
        fs_random_state=int(args.random_state),
        sample_weight=sample_weight,
    )

    # Train final model on all data
    final_model, losses = train_final_model(
        X_int8=X_sel,
        y_enc=y_enc,
        n_classes=len(class_labels),
        keep_fraction=keep_fraction,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        batch_size=bs,
        device=args.device,
        seed=args.random_state,
        max_epochs=epochs,
        sample_weight=sample_weight,
    )

    # Save bundle (avoid pickling sklearn objects)
    bundle = {
        "state_dict": final_model.state_dict(),
        "n_features": int(X_sel.shape[1]),
        "n_classes": int(len(class_labels)),
        "class_names": class_labels,
        "label_to_index": {str(k): int(i) for i, k in enumerate(class_labels)},
        "feature_names": selected_feat_names,
        "beta_threshold": float(args.beta_threshold),
        "equal_mode": str(args.equal_mode),
        "mask_keep_fraction": float(keep_fraction),
        "learning_rate": float(lr),
        "weight_decay": float(wd),
        "epochs": int(epochs),
        "batch_size": int(bs),
        "device": str(args.device),
        "created_at": str(pd.Timestamp.now()),
        "cv_summary": summary,
        "cv_group_mode": str(args.cv_group_mode),
        "cv_group_col": str(args.cv_group_col or ""),
        "group_by": str(args.group_by),
        "tune_metric": str(args.tune_metric),
        "eval_keep_fraction": float(args.tune_eval_keep_fraction),
        "sample_weight_report": sample_weight_report,
    }
    torch.save(bundle, os.path.join(args.outdir, "final_crossnn_bundle.pt"))
    with open(os.path.join(args.outdir,'features.txt'), 'w',encoding="utf-8") as fff:
        for feature in selected_feat_names:
            fff.write(f"{feature}\n")
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(
        os.path.join(args.outdir, "final_train_loss.csv"), index=False
    )

    with open(os.path.join(args.outdir, "final_params.json"), "w", encoding="utf-8") as f:
        json.dump({
            "mask_keep_fraction": keep_fraction,
            "learning_rate": lr,
            "weight_decay": wd,
            "epochs": epochs,
            "batch_size": bs,
            "cv_group_mode": args.cv_group_mode,
            "cv_group_col": args.cv_group_col,
            "group_by": args.group_by,
            "tune_metric": args.tune_metric,
            "eval_keep_fraction": args.tune_eval_keep_fraction,
            "sample_weight_report": sample_weight_report,
        }, f, indent=2, ensure_ascii=False)

    print("\n=== Done ===")
    print(f"Saved final crossNN bundle: {os.path.join(args.outdir, 'final_crossnn_bundle.pt')}")


if __name__ == "__main__":
    main()
