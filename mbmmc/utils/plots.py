from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize

def plot_metrics_bar(metrics_df, out_png: str):
    # expects columns: accuracy, balanced_accuracy, macro_f1, (optional) log_loss, macro_roc_auc_ovr
    cols = [c for c in ["accuracy","balanced_accuracy","macro_f1","macro_roc_auc_ovr"] if c in metrics_df.columns]
    means = [metrics_df[c].mean() for c in cols]
    stds = [metrics_df[c].std(ddof=1) for c in cols]

    plt.figure(figsize=(8,4.5))
    x = np.arange(len(cols))
    plt.bar(x, means, yerr=stds)
    plt.xticks(x, cols, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("CV metrics (mean ± std)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_matrix(cm_norm: np.ndarray, labels: List[str], out_png: str, title: str = "Confusion matrix (row-normalized)"):
    plt.figure(figsize=(7,6))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_macro_roc(y_true, y_proba: np.ndarray, class_ids: List[int], out_png: str, title: str = "OvR ROC curves", class_names: Optional[List[str]] = None):
    """Plot OvR ROC curves for multi-class.
    - y_true should be encoded as integers matching class_ids.
    - y_proba shape: (n_samples, n_classes)
    - class_ids: e.g. [0,1,2,3,4]
    - class_names: optional display names with same length as class_ids
    """
    y_true_bin = label_binarize(y_true, classes=class_ids)

    if class_names is None:
        class_names = [str(c) for c in class_ids]

    plt.figure(figsize=(7,5.5))
    for i, name in enumerate(class_names):
        try:
            RocCurveDisplay.from_predictions(y_true_bin[:, i], y_proba[:, i], name=str(name))
        except Exception:
            continue
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()



def plot_prob_bar(proba: np.ndarray, class_labels: List[str], out_png: str, title: str = "Prediction probabilities"):
    proba = np.asarray(proba).reshape(-1)
    plt.figure(figsize=(8,4.5))
    x = np.arange(len(class_labels))
    plt.bar(x, proba)
    plt.xticks(x, class_labels, rotation=45, ha="right")
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
