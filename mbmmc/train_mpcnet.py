from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from .utils.io import load_matrix_and_labels, set_seed, software_versions, write_json
from .utils.sample_weights import read_meta_for_weights, resolve_sample_weight_vector, write_sample_weight_audit
from .utils.features import fit_feature_bundle, transform_by_feature_bundle
from .utils.metrics import entropy, expected_calibration_error, multiclass_metrics
from .mpcnet_model import MPCNet, coverage_covariates


MODEL_FILENAME = "mpcnet_model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Train the MBMMC MPCNet sparse/hybrid methylation classifier")
    p.add_argument("--version", action="version", version="MBMMC 1.0.0")
    p.add_argument("--matrix", default="", help="Row-wise methylation matrix: sample_id + feature columns")
    p.add_argument("--labels", default="", help="Row-wise label table: sample_id + label column")
    p.add_argument("--ref_csv", default="", help="Legacy reference CSV: probe_id,chr,pos,sample1,sample2,...")
    p.add_argument("--meta", default="", help="Legacy meta table with Sample and Types columns")
    p.add_argument("--outdir", required=True)
    p.add_argument("--sample_id_col", default="sample_id")
    p.add_argument("--label_col", default="label")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--val_group_col", default="", help="Optional label/meta column for group-aware internal split, e.g. Patient, Study, Platform, Material.")
    p.add_argument("--feature_select", default="none",
                   choices=["none", "technical_only", "technical_only_qc", "variance_topk", "f_classif_topk", "pairwise_score_topk"])
    p.add_argument("--topk", type=int, default=50000)
    p.add_argument("--min_observed_per_feature", type=int, default=3)
    p.add_argument("--min_observed_fraction", type=float, default=0.0)
    p.add_argument("--min_variance_observed", type=float, default=0.0)
    p.add_argument("--max_missing_fraction", type=float, default=1.0)
    p.add_argument("--input_compression", default="auto", choices=["auto", "dense", "hash"])
    p.add_argument("--hash_bins", type=int, default=32768)
    p.add_argument("--max_dense_features", type=int, default=30000)
    p.add_argument("--hash_seed", type=int, default=17)
    p.add_argument("--value_mode", default="raw", choices=["raw", "ternary", "hybrid"],
                   help="MPCNet input value representation. raw preserves beta; ternary uses -1/+1 after beta_threshold; hybrid concatenates raw and ternary channels.")
    p.add_argument("--beta_threshold", type=float, default=0.60,
                   help="Beta cutoff for value_mode=ternary/hybrid. Kept in the model bundle and reused at prediction.")
    p.add_argument("--beta_threshold_equal_mode", default="negative", choices=["negative", "positive"],
                   help="Boundary rule for beta exactly equal to beta_threshold. "
                        "negative: beta > threshold is positive and beta <= threshold is negative; "
                        "positive: beta >= threshold is positive and beta < threshold is negative.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--sample_weight_col", default="",
                   help="Optional labels/meta column containing per-sample training weights. Recommended for merged real+in-silico training: RecommendedTrainingWeight.")
    p.add_argument("--auto_sample_weight", action="store_true",
                   help="Infer real/synthetic/tumor-fraction weights from labels/meta when sample_weight_col is absent.")
    p.add_argument("--synthetic_prefix", default="SIM_", help="Sample-name prefix used to infer synthetic samples.")
    p.add_argument("--real_weight", type=float, default=1.0)
    p.add_argument("--synthetic_weight", type=float, default=0.35)
    p.add_argument("--low_tf_weight", type=float, default=0.10)
    p.add_argument("--boundary_tf_weight", type=float, default=0.45)
    p.add_argument("--high_tf_weight", type=float, default=0.35)
    p.add_argument("--decision_tf_weight_boundary", type=float, default=0.10)
    p.add_argument("--platform_weight_map", default="",
                   help="Optional multiplicative platform factors, e.g. TAPS:0.8,WGBS:0.6,ONT:0.5")
    p.add_argument("--sample_regex_weight_map", default="",
                   help="Optional multiplicative regex factors, e.g. '.*lowtf.*:0.2,.*artifact.*:0.1'")
    p.add_argument("--final_refit", default="all", choices=["none", "all"])
    p.add_argument("--prob_threshold", type=float, default=0.70)
    p.add_argument("--min_coverage_accept", type=float, default=0.01)
    p.add_argument("--temperature_scaling", default="on", choices=["on", "off"],
                   help="Fit scalar temperature on internal validation and apply it at prediction time.")
    p.add_argument("--temperature_min", type=float, default=0.05,
                   help="Lower bound for fitted temperature. Use 1.0 to prevent over-sharpening for sparse/cross-platform deployment.")
    p.add_argument("--temperature_max", type=float, default=20.0)
    p.add_argument("--train_mask_aug", default="off", choices=["off", "mcar", "mcar_beta_rate", "block", "empirical", "mixed"],
                   help="Train-time sparse masking applied to value and mask tensors only during optimization.")
    p.add_argument("--train_mask_keep_fractions", default="1.0,0.75,0.50",
                   help="Comma-separated keep fractions sampled during train-time mask augmentation. Example: 1.0,0.75,0.50.")
    p.add_argument("--train_mask_aug_prob", type=float, default=0.50,
                   help="Probability of applying train-time mask augmentation to each batch.")
    p.add_argument("--train_mask_beta_kappa", type=float, default=20.0,
                   help="Concentration for per-sample Beta keep fractions in mcar_beta_rate augmentation.")
    p.add_argument("--train_mask_block_fraction", type=float, default=0.10,
                   help="Approximate contiguous model-feature block width for block masking, as fraction of n_model_features.")
    p.add_argument("--empirical_mask_matrix", default="",
                   help="Optional unlabeled sample-wide matrix whose observed CpG patterns are used as empirical mask templates; labels are not used.")
    p.add_argument("--prototype_weight", type=float, default=0.35,
                   help="Weight of the prototype head when enabled. Lower values reduce prototype-driven overconfidence.")
    p.add_argument("--prototype_norm", action="store_true",
                   help="Use cosine-normalized prototypes instead of squared Euclidean prototype distances.")
    p.add_argument("--prototype_temperature", type=float, default=1.0,
                   help="Temperature applied to cosine prototype logits when --prototype_norm is used.")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Cross-entropy label smoothing. Use 0.03-0.10 to reduce over-confident sparse/domain-shift predictions.")
    p.add_argument("--auto_accept_thresholds", action="store_true",
                   help="Tune prob/margin/entropy accept/no-call gates on the internal validation split. "
                        "Thresholds are diagnostic only unless --apply_auto_accept_thresholds is also set.")
    p.add_argument("--apply_auto_accept_thresholds", action="store_true",
                   help="Store internally tuned acceptance gates in the deployment bundle. "
                        "Unsafe for cross-platform clinical deployment unless the validation split is an independently locked target-domain calibration set.")
    p.add_argument("--target_accept_error", type=float, default=0.05,
                   help="Maximum accepted-call error rate targeted when --auto_accept_thresholds is enabled.")
    p.add_argument("--min_accept_fraction", type=float, default=0.20,
                   help="Minimum validation fraction accepted during automatic threshold search when feasible.")
    p.add_argument("--threshold_grid_size", type=int, default=21,
                   help="Number of quantile grid points per score for automatic threshold search.")
    p.add_argument("--disable_mask_stream", action="store_true")
    p.add_argument("--disable_coverage_embedding", action="store_true")
    p.add_argument("--disable_prototype_head", action="store_true")
    p.add_argument("--disable_feature_gates", action="store_true")
    return p.parse_args()


def _resolve_device(device: str) -> str:
    d = str(device).lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda requested but CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return d


def tensors(values, mask, y=None):
    v = torch.tensor(values, dtype=torch.float32)
    m = torch.tensor(mask, dtype=torch.float32)
    cov = coverage_covariates(v, m)
    if y is None:
        return TensorDataset(v, m, cov)
    yy = torch.tensor(y, dtype=torch.long)
    return TensorDataset(v, m, cov, yy)


def predict_logits(model, values, mask, batch_size=256, device="cpu") -> np.ndarray:
    model.eval()
    ds = tensors(values, mask, y=None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits_all = []
    with torch.no_grad():
        for batch in dl:
            v, m, cov = [b.to(device) for b in batch]
            logits, _ = model(v, m, cov)
            logits_all.append(logits.detach().cpu())
    return torch.cat(logits_all, dim=0).numpy()


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    z = np.asarray(logits, dtype=np.float64) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / (e.sum(axis=1, keepdims=True) + 1e-12)
    return p.astype(np.float32)


def _parse_float_list(text: str, default: Sequence[float] = (1.0,)) -> list[float]:
    vals = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            pass
    vals = [min(max(v, 1e-6), 1.0) for v in vals if np.isfinite(v)]
    return vals or list(default)


def _apply_mpcnet_mask_augmentation(v: torch.Tensor, m: torch.Tensor, args, keep_fractions: Sequence[float], empirical_templates: Optional[torch.Tensor] = None):
    """Apply sparse methylome augmentation to both value and mask tensors.

    The operation is intentionally label-free and only active inside the training
    loop. It simulates reduced CpG observability without changing validation or
    prediction data. The mask may be fractional after hash compression; multiplying
    by a binary keep matrix remains valid and preserves the observed-coverage logic.
    """
    alg = str(getattr(args, "train_mask_aug", "off") or "off").lower()
    prob = float(getattr(args, "train_mask_aug_prob", 0.0))
    if alg == "off" or prob <= 0.0 or v.numel() == 0:
        return v, m
    if torch.rand((), device=v.device) > min(max(prob, 0.0), 1.0):
        return v, m

    choices = list(keep_fractions) or [1.0]
    n, d = v.shape
    augmented_values = v.clone()
    augmented_mask = m.clone()

    active_alg = alg
    if alg == "mixed":
        pool = ["mcar", "mcar_beta_rate", "block"]
        if empirical_templates is not None and empirical_templates.numel() > 0:
            pool.append("empirical")
        active_alg = pool[int(torch.randint(0, len(pool), (1,), device=v.device).item())]

    if active_alg == "empirical" and empirical_templates is not None and empirical_templates.numel() > 0:
        idx = torch.randint(0, empirical_templates.shape[0], (n,), device=v.device)
        keep = empirical_templates[idx].to(device=v.device, dtype=v.dtype)
        # For hashed templates mask values can be fractional. Convert to observed/not-observed.
        keep = (keep > 0).to(dtype=v.dtype)
        return augmented_values * keep, augmented_mask * keep

    keep = torch.ones((n, d), device=v.device, dtype=v.dtype)

    if active_alg == "mcar":
        kidx = torch.randint(0, len(choices), (n,), device=v.device)
        sample_keep = torch.tensor(choices, dtype=v.dtype, device=v.device)[kidx].view(n, 1)
        keep = (torch.rand((n, d), device=v.device, dtype=v.dtype) < sample_keep).to(dtype=v.dtype)

    elif active_alg == "mcar_beta_rate":
        # Sample keep fractions around one of the requested keep_fraction centers.
        kidx = torch.randint(0, len(choices), (n,), device=v.device)
        centers = torch.tensor(choices, dtype=v.dtype, device=v.device)[kidx].clamp(1e-4, 1.0 - 1e-4)
        kappa = max(float(getattr(args, "train_mask_beta_kappa", 20.0)), 1e-3)
        alpha = (centers * kappa).detach().cpu().numpy()
        beta = ((1.0 - centers) * kappa).detach().cpu().numpy()
        # torch.distributions.Beta is slower on some CPU builds; numpy sampling is deterministic enough after set_seed for augmentation.
        sampled = np.random.beta(alpha, beta).astype("float32")
        sample_keep = torch.tensor(sampled, dtype=v.dtype, device=v.device).view(n, 1).clamp(1e-6, 1.0)
        keep = (torch.rand((n, d), device=v.device, dtype=v.dtype) < sample_keep).to(dtype=v.dtype)

    elif active_alg == "block":
        kidx = torch.randint(0, len(choices), (n,), device=v.device)
        sample_keep_vals = torch.tensor(choices, dtype=v.dtype, device=v.device)[kidx]
        block_frac = min(max(float(getattr(args, "train_mask_block_fraction", 0.10)), 1.0 / max(d, 1)), 1.0)
        block_len = max(1, int(round(block_frac * d)))
        for i in range(n):
            drop_total = int(round((1.0 - float(sample_keep_vals[i].item())) * d))
            if drop_total <= 0:
                continue
            dropped = 0
            while dropped < drop_total:
                start = int(torch.randint(0, max(d, 1), (1,), device=v.device).item())
                end = min(d, start + block_len, start + (drop_total - dropped))
                if end <= start:
                    break
                keep[i, start:end] = 0.0
                dropped += (end - start)
    else:
        return v, m

    return augmented_values * keep, augmented_mask * keep


def _load_empirical_mask_templates(path: str, sample_id_col: str, feature_bundle: dict) -> Optional[np.ndarray]:
    if not path:
        return None
    try:
        from .utils.io import read_table
        for col in [sample_id_col, "Sample", "sample_id"]:
            try:
                Xemp = read_table(path, sample_id_col=col).apply(pd.to_numeric, errors="coerce")
                _, emp_mask, _ = transform_by_feature_bundle(Xemp, feature_bundle)
                if emp_mask.size == 0:
                    return None
                return emp_mask.astype(np.float32)
            except Exception:
                continue
    except Exception as e:
        print(f"[WARN] empirical mask template loading failed: {e}")
    return None


def fit_temperature(logits: np.ndarray, y: np.ndarray, min_temperature: float = 0.05, max_temperature: float = 20.0) -> Tuple[float, float, float]:
    """Return temperature, raw NLL, calibrated NLL."""
    if logits.shape[0] < 3 or logits.shape[1] < 2:
        return 1.0, float("nan"), float("nan")
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    raw_nll = float(torch.nn.functional.cross_entropy(lt, yt).item())
    log_t = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(float(min_temperature), float(max_temperature))
        loss = torch.nn.functional.cross_entropy(lt / t, yt)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        temp = float(torch.exp(log_t).detach().clamp(float(min_temperature), float(max_temperature)).item())
        cal_nll = float(torch.nn.functional.cross_entropy(lt / temp, yt).item())
        if not np.isfinite(temp) or not np.isfinite(cal_nll) or cal_nll > raw_nll + 1e-4:
            return 1.0, raw_nll, raw_nll
        return temp, raw_nll, cal_nll
    except Exception as e:
        print(f"[WARN] temperature scaling failed: {e}")
        return 1.0, raw_nll, raw_nll


def evaluate(model, values, mask, y, batch_size=256, device="cpu", temperature=1.0):
    logits = predict_logits(model, values, mask, batch_size=batch_size, device=device)
    probs = softmax_np(logits, temperature=temperature)
    pred = probs.argmax(axis=1)
    loss = float(torch.nn.functional.cross_entropy(torch.tensor(logits / max(float(temperature), 1e-6)), torch.tensor(y)).item())
    return loss, probs, pred, logits


def margin_np(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] <= 1:
        return np.ones(probs.shape[0], dtype=np.float32)
    s = np.sort(np.asarray(probs, dtype=np.float64), axis=1)
    return (s[:, -1] - s[:, -2]).astype(np.float32)


def _grid_from_values(values: np.ndarray, n: int, extras: Sequence[float] = ()) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    pts = list(extras)
    if values.size:
        qs = np.linspace(0.0, 1.0, max(2, int(n)))
        pts.extend(np.quantile(values, qs).tolist())
    pts = np.asarray(pts, dtype=float)
    pts = pts[np.isfinite(pts)]
    if pts.size == 0:
        return np.asarray([0.0], dtype=float)
    return np.unique(np.round(pts, 6))


def tune_acceptance_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    coverage: np.ndarray,
    args,
) -> dict:
    """Tune accept/no-call gates from validation labels.

    The search deliberately uses only monotonic, clinically interpretable gates:
    max probability >= threshold, margin >= threshold, entropy <= threshold,
    and feature coverage >= threshold. The objective is to accept as many
    validation samples as possible while keeping accepted-call error below a
    user-specified target when feasible. This is a pragmatic selective
    classification layer; it should be re-estimated on a locked external
    calibration set before clinical use.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    correct = pred == y_true
    maxp = probs.max(axis=1)
    marg = margin_np(probs)
    ent = entropy(probs)
    cov = np.asarray(coverage, dtype=float)
    cov_thr = float(getattr(args, "min_coverage_accept", 0.01))
    n = int(len(y_true))
    if n == 0:
        return {"enabled": False, "reason": "empty_validation"}

    grid_n = int(max(5, getattr(args, "threshold_grid_size", 21)))
    prob_grid = _grid_from_values(maxp, grid_n, extras=[float(getattr(args, "prob_threshold", 0.70)), 0.50, 0.70, 0.84, 0.90, 0.95, 0.99])
    margin_grid = _grid_from_values(marg, grid_n, extras=[0.0, 0.10, 0.30, 0.50, 0.70, 0.90])
    entropy_grid = _grid_from_values(ent, grid_n, extras=[0.25, 0.50, 0.75, 1.0, float(np.log(max(probs.shape[1], 2)))])

    target_error = float(np.clip(getattr(args, "target_accept_error", 0.05), 0.0, 1.0))
    min_accept_n = max(1, int(np.ceil(float(np.clip(getattr(args, "min_accept_fraction", 0.20), 0.0, 1.0)) * n)))
    feasible = []
    fallback = []
    for pthr in prob_grid:
        for mthr in margin_grid:
            base = (maxp >= pthr) & (marg >= mthr) & (cov >= cov_thr)
            if not base.any():
                continue
            for ethr in entropy_grid:
                acc = base & (ent <= ethr)
                k = int(acc.sum())
                if k == 0:
                    continue
                err = float(1.0 - np.mean(correct[acc]))
                row = {
                    "prob_threshold": float(pthr),
                    "margin_threshold": float(mthr),
                    "entropy_threshold": float(ethr),
                    "min_coverage_accept": float(cov_thr),
                    "accepted_n": k,
                    "accepted_fraction": float(k / n),
                    "accepted_accuracy": float(np.mean(correct[acc])),
                    "accepted_error": err,
                    "wrong_accepted_n": int((~correct & acc).sum()),
                    "correct_accepted_n": int((correct & acc).sum()),
                }
                fallback.append(row)
                if err <= target_error and k >= min_accept_n:
                    feasible.append(row)
    if feasible:
        # Maximize accepted calls, then minimize accepted-call error. When several
        # gates select the same validation rows, prefer the stricter gate: larger
        # probability/margin cutoffs and a smaller entropy cutoff.
        best = sorted(
            feasible,
            key=lambda r: (
                r["accepted_n"],
                -r["accepted_error"],
                r["prob_threshold"],
                r["margin_threshold"],
                -r["entropy_threshold"],
            ),
            reverse=True,
        )[0]
        best["enabled"] = True
        best["selection_status"] = "target_met"
    elif fallback:
        best = sorted(
            fallback,
            key=lambda r: (
                -r["accepted_error"],
                r["accepted_n"],
                r["prob_threshold"],
                r["margin_threshold"],
                -r["entropy_threshold"],
            ),
            reverse=True,
        )[0]
        best["enabled"] = True
        best["selection_status"] = "target_not_met_best_available"
    else:
        best = {"enabled": False, "selection_status": "no_nonempty_gate"}
    best["target_accept_error"] = target_error
    best["min_accept_fraction"] = float(getattr(args, "min_accept_fraction", 0.20))
    best["n_validation"] = n
    best["score_summary"] = {
        "correct_n": int(correct.sum()),
        "wrong_n": int((~correct).sum()),
        "max_prob_correct_min": float(np.min(maxp[correct])) if correct.any() else None,
        "max_prob_wrong_max": float(np.max(maxp[~correct])) if (~correct).any() else None,
        "margin_correct_min": float(np.min(marg[correct])) if correct.any() else None,
        "margin_wrong_max": float(np.max(marg[~correct])) if (~correct).any() else None,
        "entropy_correct_max": float(np.max(ent[correct])) if correct.any() else None,
        "entropy_wrong_min": float(np.min(ent[~correct])) if (~correct).any() else None,
    }
    return best


def train_loop(args, train_pack, val_pack, n_classes, class_weights=None, fixed_epochs=None, empirical_mask_templates=None):
    device = _resolve_device(args.device)
    if len(train_pack) == 4:
        tr_values, tr_mask, tr_y, tr_w = train_pack
    else:
        tr_values, tr_mask, tr_y = train_pack
        tr_w = None
    if val_pack is not None:
        va_values, va_mask, va_y = val_pack[:3]
    else:
        va_values, va_mask, va_y = None, None, None
    model = MPCNet(
        n_features=tr_values.shape[1], n_classes=n_classes, hidden=args.hidden, depth=args.depth,
        dropout=args.dropout, use_mask_stream=not args.disable_mask_stream,
        use_coverage_embedding=not args.disable_coverage_embedding,
        use_prototype_head=not args.disable_prototype_head,
        prototype_weight=float(args.prototype_weight),
        use_feature_gates=not args.disable_feature_gates,
        prototype_norm=bool(args.prototype_norm),
        prototype_temperature=float(args.prototype_temperature),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    try:
        crit = torch.nn.CrossEntropyLoss(weight=cw, reduction="none", label_smoothing=float(getattr(args, "label_smoothing", 0.0)))
    except TypeError:
        if float(getattr(args, "label_smoothing", 0.0)) > 0:
            print("[WARN] This PyTorch build does not support CrossEntropyLoss(label_smoothing); continuing without label smoothing.")
        crit = torch.nn.CrossEntropyLoss(weight=cw, reduction="none")
    keep_fractions = _parse_float_list(getattr(args, "train_mask_keep_fractions", "1.0"), default=(1.0,))
    emp_t = None
    if empirical_mask_templates is not None:
        emp_arr = np.asarray(empirical_mask_templates, dtype=np.float32)
        if emp_arr.ndim == 2 and emp_arr.shape[1] == tr_values.shape[1]:
            emp_t = torch.tensor(emp_arr, dtype=torch.float32, device=device)
        else:
            print("[WARN] empirical mask templates ignored because shape does not match model features.")
    if tr_w is None:
        tr_w = np.ones(len(tr_y), dtype=np.float32)
    ds = TensorDataset(
        torch.tensor(tr_values, dtype=torch.float32),
        torch.tensor(tr_mask, dtype=torch.float32),
        torch.tensor(tr_y, dtype=torch.long),
        torch.tensor(np.asarray(tr_w, dtype=np.float32), dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    best = {"epoch": 0, "val_loss": float("inf"), "state": None}
    max_epochs = int(fixed_epochs or args.epochs)
    bad = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in dl:
            v, m, yb, wb = [b.to(device) for b in batch]
            v, m = _apply_mpcnet_mask_augmentation(v, m, args, keep_fractions, empirical_templates=emp_t)
            cov = coverage_covariates(v, m)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(v, m, cov)
            loss_vec = crit(logits, yb)
            loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_pack is not None:
            val_loss, _, _, _ = evaluate(model, va_values, va_mask, va_y, batch_size=args.batch_size, device=device, temperature=1.0)
            row["val_loss"] = val_loss
            if val_loss < best["val_loss"] - 1e-5:
                best = {"epoch": epoch, "val_loss": val_loss, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            if bad >= args.patience and fixed_epochs is None:
                history.append(row)
                break
        else:
            best = {"epoch": epoch, "val_loss": None, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        history.append(row)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, history, best["epoch"] or max_epochs


def _load_training_data_from_args(args):
    if args.ref_csv and args.meta:
        from .utils.data_io import load_reference
        ref = load_reference(args.ref_csv, args.meta, exclude_types=None)
        X = pd.DataFrame(ref.X_beta, index=pd.Index(ref.samples, name=args.sample_id_col), columns=ref.features)
        y_text = pd.Series(ref.y.astype(str), index=X.index, name=args.label_col)
        labels_df = pd.DataFrame({args.label_col: y_text}, index=X.index)
        # Preserve all metadata columns (e.g. Scenario/TumorFraction/RecommendedTrainingWeight)
        # so sample weighting and group splits can use them.
        try:
            meta_all = read_meta_for_weights(args.meta, list(X.index.astype(str)))
            meta_all = meta_all.set_index(pd.Index(X.index, name=args.sample_id_col))
            for c in meta_all.columns:
                if c not in labels_df.columns and c != args.sample_id_col:
                    labels_df[c] = meta_all[c].to_numpy()
        except Exception as e:
            print(f"[WARN] Could not preserve full metadata for MPCNet sample weighting: {e}")
        if ref.platform is not None and "Platform" not in labels_df.columns:
            labels_df["Platform"] = ref.platform
        if ref.material is not None and "Material" not in labels_df.columns:
            labels_df["Material"] = ref.material
        return X, y_text, labels_df
    if args.matrix and args.labels:
        return load_matrix_and_labels(args.matrix, args.labels, args.sample_id_col, args.label_col)
    raise ValueError("Provide either --matrix + --labels or --ref_csv + --meta.")


def _internal_split(y: np.ndarray, labels_df: pd.DataFrame, args) -> Tuple[np.ndarray, np.ndarray, str]:
    idx = np.arange(len(y))
    if args.val_group_col and args.val_group_col in labels_df.columns:
        groups = labels_df[args.val_group_col].astype(str).to_numpy()
        if len(np.unique(groups)) >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.val_fraction), random_state=int(args.seed))
            tr, va = next(splitter.split(idx, y, groups))
            return tr, va, f"group_shuffle:{args.val_group_col}"
        print(f"[WARN] val_group_col={args.val_group_col!r} has <2 groups; falling back to stratified/random split.")
    strat = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    tr, va = train_test_split(idx, test_size=args.val_fraction, random_state=args.seed, stratify=strat)
    return tr, va, "stratified_random" if strat is not None else "random"


def _class_weights(y: np.ndarray, n_classes: int) -> Optional[np.ndarray]:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return counts.sum() / (len(counts) * counts)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    args.device = _resolve_device(args.device)

    X, y_text, labels_df = _load_training_data_from_args(args)
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    meta_for_weights = labels_df.copy()
    meta_for_weights["Sample"] = X.index.astype(str)
    sample_weight, sample_weight_report = resolve_sample_weight_vector(
        meta=meta_for_weights.reset_index(drop=True),
        samples=list(X.index.astype(str)),
        y=y,
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
        write_sample_weight_audit(str(outdir), list(X.index.astype(str)), sample_weight, sample_weight_report, meta_for_weights.reset_index(drop=True))
        print(f"[sample_weights] enabled: min={sample_weight.min():.4g}, median={np.median(sample_weight):.4g}, max={sample_weight.max():.4g}, synthetic={sample_weight_report.get('n_synthetic_inferred')}")
    else:
        sample_weight = None
        sample_weight_report = None

    tr_idx, va_idx, split_mode = _internal_split(y, labels_df, args)

    qc = {
        "min_observed_per_feature": args.min_observed_per_feature,
        "min_observed_fraction": args.min_observed_fraction,
        "min_variance_observed": args.min_variance_observed,
        "max_missing_fraction": args.max_missing_fraction,
    }

    f_internal = fit_feature_bundle(
        X.iloc[tr_idx], y[tr_idx], feature_select=args.feature_select, topk=args.topk,
        technical_qc=qc, input_compression=args.input_compression, hash_bins=args.hash_bins,
        max_dense_features=args.max_dense_features, hash_seed=args.hash_seed,
        value_mode=args.value_mode, beta_threshold=args.beta_threshold,
        beta_threshold_equal_mode=args.beta_threshold_equal_mode
    )
    tr_values, tr_mask, _ = transform_by_feature_bundle(X.iloc[tr_idx], f_internal)
    va_values, va_mask, va_cov_orig = transform_by_feature_bundle(X.iloc[va_idx], f_internal)

    empirical_masks_internal = _load_empirical_mask_templates(args.empirical_mask_matrix, args.sample_id_col, f_internal)

    class_weights = _class_weights(y[tr_idx], len(le.classes_)) if args.class_weight else None
    model, history, best_epoch = train_loop(
        args, (tr_values, tr_mask, y[tr_idx], None if sample_weight is None else sample_weight[tr_idx]), (va_values, va_mask, y[va_idx]),
        n_classes=len(le.classes_), class_weights=class_weights, empirical_mask_templates=empirical_masks_internal
    )

    # Internal validation, then scalar temperature calibration on the same internal validation split.
    _, raw_probs, raw_pred, val_logits = evaluate(model, va_values, va_mask, y[va_idx], batch_size=args.batch_size, device=args.device, temperature=1.0)
    if args.temperature_scaling == "on":
        temperature, raw_nll, cal_nll = fit_temperature(val_logits, y[va_idx], min_temperature=args.temperature_min, max_temperature=args.temperature_max)
    else:
        temperature, raw_nll, cal_nll = 1.0, float("nan"), float("nan")
    val_probs = softmax_np(val_logits, temperature=temperature)
    val_pred = val_probs.argmax(axis=1)
    val_loss = float(torch.nn.functional.cross_entropy(torch.tensor(val_logits / max(float(temperature), 1e-6)), torch.tensor(y[va_idx])).item())

    val_labels = le.inverse_transform(y[va_idx])
    pred_labels = le.inverse_transform(val_pred)
    metrics = multiclass_metrics(val_labels, pred_labels, list(le.classes_))
    ece, calib_df = expected_calibration_error(y[va_idx], val_probs)
    raw_ece, raw_calib_df = expected_calibration_error(y[va_idx], raw_probs)
    auto_thresholds = tune_acceptance_thresholds(y[va_idx], val_probs, va_cov_orig, args) if args.auto_accept_thresholds else {"enabled": False, "selection_status": "disabled"}

    metrics.update({
        "ece": float(ece),
        "raw_ece": float(raw_ece),
        "val_loss": float(val_loss),
        "raw_nll": float(raw_nll),
        "calibrated_nll": float(cal_nll),
        "temperature": float(temperature),
        "best_epoch_internal": int(best_epoch),
        "n_train_internal": int(len(tr_idx)),
        "n_val_internal": int(len(va_idx)),
        "internal_split_mode": split_mode,
        "value_mode": str(args.value_mode),
        "beta_threshold": float(args.beta_threshold),
        "beta_threshold_equal_mode": str(args.beta_threshold_equal_mode),
        "train_mask_aug": str(args.train_mask_aug),
        "train_mask_keep_fractions": str(args.train_mask_keep_fractions),
        "train_mask_aug_prob": float(args.train_mask_aug_prob),
        "sample_weight_enabled": bool(sample_weight is not None),
        "label_smoothing": float(args.label_smoothing),
        "prototype_weight": float(args.prototype_weight),
        "prototype_norm": bool(args.prototype_norm),
        "auto_accept_thresholds": auto_thresholds,
    })
    if val_probs.shape[1] >= 3:
        top3 = np.argsort(val_probs, axis=1)[:, -3:]
        metrics["top3_accuracy"] = float(np.mean([y[va_idx][i] in top3[i] for i in range(len(va_idx))]))

    pd.DataFrame(history).to_csv(outdir / "mpcnet_training_history.csv", index=False)
    calib_df.to_csv(outdir / "mpcnet_calibration_bins_internal.csv", index=False)
    raw_calib_df.to_csv(outdir / "mpcnet_calibration_bins_internal_raw.csv", index=False)
    write_json(auto_thresholds, outdir / "mpcnet_acceptance_thresholds_internal.json")
    pd.DataFrame({
        "Sample": X.index[va_idx].astype(str),
        "sample_id": X.index[va_idx].astype(str),
        "true_label": val_labels,
        "pred_label": pred_labels,
        "PredictedClass": pred_labels,
        "max_proba_mean": val_probs.max(axis=1),
        "max_prob_raw": raw_probs.max(axis=1),
        "entropy_mean": entropy(val_probs),
        "feature_coverage": va_cov_orig,
        "internal_split_mode": split_mode,
        "value_mode": str(args.value_mode),
        "beta_threshold": float(args.beta_threshold),
        "beta_threshold_equal_mode": str(args.beta_threshold_equal_mode),
        "train_mask_aug": str(args.train_mask_aug),
        "train_mask_keep_fractions": str(args.train_mask_keep_fractions),
        "train_mask_aug_prob": float(args.train_mask_aug_prob),
        "sample_weight_enabled": bool(sample_weight is not None),
    }).to_csv(outdir / "mpcnet_internal_validation_predictions.csv", index=False)

    final_feature_bundle = f_internal
    final_model = model
    if args.final_refit == "all":
        final_feature_bundle = fit_feature_bundle(
            X, y, feature_select=args.feature_select, topk=args.topk,
            technical_qc=qc, input_compression=args.input_compression, hash_bins=args.hash_bins,
            max_dense_features=args.max_dense_features, hash_seed=args.hash_seed,
            value_mode=args.value_mode, beta_threshold=args.beta_threshold
        )
        all_values, all_mask, _ = transform_by_feature_bundle(X, final_feature_bundle)
        empirical_masks_final = _load_empirical_mask_templates(args.empirical_mask_matrix, args.sample_id_col, final_feature_bundle)
        final_class_weights = _class_weights(y, len(le.classes_)) if args.class_weight else None
        final_model, refit_hist, _ = train_loop(
            args, (all_values, all_mask, y, sample_weight), None,
            n_classes=len(le.classes_), class_weights=final_class_weights, fixed_epochs=max(1, int(best_epoch)), empirical_mask_templates=empirical_masks_final
        )
        pd.DataFrame(refit_hist).to_csv(outdir / "mpcnet_final_refit_history.csv", index=False)

    # Internal automatic threshold search is useful for diagnostics, but an
    # internal/simulated validation split is not a valid clinical deployment
    # calibration set under TAPS/WGBS/ONT domain shift. Keep user-specified
    # conservative thresholds unless the caller explicitly opts in.
    apply_auto_accept_thresholds = bool(
        args.auto_accept_thresholds
        and getattr(args, "apply_auto_accept_thresholds", False)
        and bool(auto_thresholds.get("enabled", False))
    )
    deployment_thresholds = {
        "prob_threshold": (
            float(auto_thresholds.get("prob_threshold", args.prob_threshold))
            if apply_auto_accept_thresholds else float(args.prob_threshold)
        ),
        "min_coverage_accept": (
            float(auto_thresholds.get("min_coverage_accept", args.min_coverage_accept))
            if apply_auto_accept_thresholds else float(args.min_coverage_accept)
        ),
        "margin_threshold": (
            float(auto_thresholds["margin_threshold"])
            if apply_auto_accept_thresholds and auto_thresholds.get("margin_threshold") is not None else None
        ),
        "entropy_threshold": (
            float(auto_thresholds["entropy_threshold"])
            if apply_auto_accept_thresholds and auto_thresholds.get("entropy_threshold") is not None else None
        ),
    }

    bundle = {
        "model_type": "MPCNetSparseHybrid",
        "label_classes": list(map(str, le.classes_)),
        "feature_bundle": final_feature_bundle,
        "args": vars(args),
        "thresholds": {
            **deployment_thresholds,
            "auto_accept_thresholds": auto_thresholds,
            "auto_accept_thresholds_applied_to_bundle": bool(apply_auto_accept_thresholds),
            "temperature": float(temperature),
            "temperature_scaling": str(args.temperature_scaling),
            "temperature_min": float(args.temperature_min),
            "temperature_max": float(args.temperature_max),
            "beta_threshold": float(args.beta_threshold),
            "value_mode": str(args.value_mode),
            "label_smoothing": float(args.label_smoothing),
            "prototype_weight": float(args.prototype_weight),
            "prototype_norm": bool(args.prototype_norm),
            "prototype_temperature": float(args.prototype_temperature),
        },
        "internal_metrics": metrics,
        "sample_weight_report": sample_weight_report,
        "audit_flags": {
            "primary_mpcnet_feature_policy": final_feature_bundle.get("feature_policy"),
            "supervised_feature_selection_used": args.feature_select not in {"none", "technical_only", "technical_only_qc"},
            "final_refit_all_reference": args.final_refit == "all",
            "internal_split_mode": split_mode,
            "external_validation_locked_model_required": True,
            "train_mask_augmentation_used": str(args.train_mask_aug) != "off",
            "value_mode": str(args.value_mode),
            "beta_threshold_equal_mode": str(args.beta_threshold_equal_mode),
            "hash_bins": int(args.hash_bins),
            "sample_weight_enabled": bool(sample_weight is not None),
            "label_smoothing": float(args.label_smoothing),
            "auto_accept_thresholds_enabled": bool(args.auto_accept_thresholds),
            "auto_accept_thresholds_applied_to_bundle": bool(apply_auto_accept_thresholds),
        },
        "software_versions": software_versions(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ckpt = {
        "state_dict": final_model.cpu().state_dict(),
        "n_features": int(final_feature_bundle["n_model_features"]),
        "n_classes": int(len(le.classes_)),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "prototype_weight": float(args.prototype_weight),
        "prototype_norm": bool(args.prototype_norm),
        "prototype_temperature": float(args.prototype_temperature),
        "label_smoothing": float(args.label_smoothing),
        "use_mask_stream": not args.disable_mask_stream,
        "use_coverage_embedding": not args.disable_coverage_embedding,
        "use_prototype_head": not args.disable_prototype_head,
        "use_feature_gates": not args.disable_feature_gates,
        "bundle": bundle,
    }
    torch.save(ckpt, outdir / MODEL_FILENAME)
    write_json(bundle, outdir / "mpcnet_bundle.json")
    pd.Series(final_feature_bundle["original_feature_list"]).to_csv(
        outdir / "model_features_mpcnet.txt", index=False, header=False
    )
    write_json(metrics, outdir / "mpcnet_internal_metrics.json")
    print(json.dumps({"outdir": str(outdir), "metrics": metrics, "n_model_features": final_feature_bundle["n_model_features"],
                      "input_compression": final_feature_bundle["input_compression"], "temperature": temperature}, indent=2))


if __name__ == "__main__":
    main()
