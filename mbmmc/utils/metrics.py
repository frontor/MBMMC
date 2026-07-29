from __future__ import annotations
"""
Combined metric utilities.

This file intentionally merges:
- metrics required by MPCNet, RF, and crossNN training modules
- compatibility aliases retained for the unified public interface
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    log_loss: Optional[float] = None
    macro_roc_auc_ovr: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def compute_metrics(y_true, y_pred, y_proba: Optional[np.ndarray], labels: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if y_proba is not None:
        try:
            out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
        except Exception:
            out["log_loss"] = float("nan")
        try:
            out["macro_roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            out["macro_roc_auc_ovr"] = float("nan")
    return out

def entropy_from_proba(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def confusion_matrix_norm(y_true, y_pred, labels: List[str]) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(np.float32)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return cm / row_sum

# Unified public metric aliases
def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def entropy(probs, eps=1e-12):
    return entropy_from_proba(probs, eps=eps)

def expected_calibration_error(y_true_idx, probs, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not np.any(m):
            continue
        bin_acc = float(acc[m].mean())
        bin_conf = float(conf[m].mean())
        w = float(m.mean())
        ece += w * abs(bin_acc - bin_conf)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": int(m.sum()),
                     "accuracy": bin_acc, "confidence": bin_conf})
    return float(ece), pd.DataFrame(rows)

def multiclass_metrics(y_true, y_pred, labels=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def bootstrap_ci_metric(y_true, y_pred, metric_fn, n_boot=1000, seed=17):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    n = len(y_true)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    return {
        "mean": float(np.mean(vals)),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
    }
