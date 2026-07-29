from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class MPCNet(nn.Module):
    """Mask-aware, coverage-aware, prototype-calibrated methylome classifier."""
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 256,
        depth: int = 2,
        dropout: float = 0.2,
        use_mask_stream: bool = True,
        use_coverage_embedding: bool = True,
        use_prototype_head: bool = True,
        prototype_weight: float = 0.35,
        use_feature_gates: bool = True,
        prototype_norm: bool = False,
        prototype_temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.hidden = int(hidden)
        self.use_mask_stream = bool(use_mask_stream)
        self.use_coverage_embedding = bool(use_coverage_embedding)
        self.use_prototype_head = bool(use_prototype_head)
        self.prototype_weight = float(prototype_weight)
        self.use_feature_gates = bool(use_feature_gates)
        self.prototype_norm = bool(prototype_norm)
        self.prototype_temperature = float(max(prototype_temperature, 1e-6))

        self.value_gate = nn.Parameter(torch.zeros(n_features)) if use_feature_gates else None
        self.value_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_mask_stream:
            self.mask_proj = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.mask_proj = None

        if self.use_coverage_embedding:
            self.coverage_proj = nn.Sequential(
                nn.Linear(3, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
        else:
            self.coverage_proj = None

        blocks = []
        for _ in range(int(depth)):
            blocks += [
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden, n_classes)
        if self.use_prototype_head:
            self.prototypes = nn.Parameter(torch.randn(n_classes, hidden) * 0.02)
            self.prototype_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, value, mask, coverage=None):
        if self.use_feature_gates:
            gate = torch.sigmoid(self.value_gate).unsqueeze(0)
            value = value * gate
        h = self.value_proj(value)
        if self.use_mask_stream:
            h = h + self.mask_proj(mask)
        if self.use_coverage_embedding:
            if coverage is None:
                obs = mask.mean(dim=1, keepdim=True)
                mean_val = (value.abs() * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                coverage = torch.cat([obs, mean_val, mask.std(dim=1, keepdim=True)], dim=1)
            h = h + self.coverage_proj(coverage)
        h = self.encoder(h)
        logits = self.classifier(h)
        if self.use_prototype_head:
            if self.prototype_norm:
                # Cosine prototype head is less sensitive to latent-vector norm inflation
                # than raw squared Euclidean distance, which helps avoid brittle
                # over-confident calls under cross-platform/domain shift.
                h_proto = F.normalize(h, dim=1)
                proto = F.normalize(self.prototypes, dim=1)
                proto_logits = torch.matmul(h_proto, proto.t()) * self.prototype_scale.abs() / self.prototype_temperature
            else:
                # Negative squared distance to class prototypes. Kept for backward compatibility.
                d = torch.cdist(h, self.prototypes, p=2) ** 2
                proto_logits = -self.prototype_scale.abs() * d
            logits = (1.0 - self.prototype_weight) * logits + self.prototype_weight * proto_logits
        return logits, h

def coverage_covariates(value, mask):
    obs = mask.mean(dim=1, keepdim=True)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_val = (value.abs() * mask).sum(dim=1, keepdim=True) / denom
    sd_mask = mask.std(dim=1, keepdim=True)
    return torch.cat([obs, mean_val, sd_mask], dim=1)
