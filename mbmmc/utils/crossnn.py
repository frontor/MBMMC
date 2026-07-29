"""Core layers and training helpers for the MBMMC crossNN implementation.

The scientific design is informed by Yuan et al. (Nature Cancer, 2025,
doi:10.1038/s43018-025-00976-5). This module is independently implemented
and is not an official release of the original crossNN project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import math

class CrossNNLinear(nn.Module):
    """Single-layer perceptron (linear classifier) for crossNN-style training."""
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes,bias=False) ##文献中使用的no bias
#        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


def make_mask(x: torch.Tensor, keep_fraction: float, rng: Optional[torch.Generator] = None) -> torch.Tensor:
    """Bernoulli keep mask. keep_fraction=0.0025 means keep 0.25% features (mask 99.75%)."""
    if keep_fraction >= 1.0:
        return torch.ones_like(x, dtype=torch.float32)
    if keep_fraction <= 0.0:
        return torch.zeros_like(x, dtype=torch.float32)
    # generate in float then threshold
    probs = torch.rand(x.shape, device=x.device, generator=rng)
    return (probs < keep_fraction).to(torch.float32)


def class_weights_from_labels(y: np.ndarray) -> torch.Tensor:
    # inverse frequency weights
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    w = np.zeros((int(classes.max()) + 1,), dtype=np.float32)
    for c, cnt in zip(classes, counts):
        w[int(c)] = 1.0 / math.sqrt(float(cnt))
#        w[int(c)] = 1.0 / float(cnt)
    # normalize to mean=1
    w = w / (w.mean() + 1e-12)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return torch.softmax(logits, dim=1)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
