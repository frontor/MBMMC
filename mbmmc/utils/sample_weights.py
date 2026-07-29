from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight


def parse_key_value_weights(text: str) -> Dict[str, float]:
    """Parse mappings like 'TAPS:0.6,WGBS:0.4,ONT:0.25' into a dict."""
    out: Dict[str, float] = {}
    if text is None:
        return out
    text = str(text).strip()
    if not text:
        return out
    for item in re.split(r"[,;]", text):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid weight map item '{item}'. Expected key:value")
        k, v = item.split(":", 1)
        out[k.strip()] = float(v)
    return out


def _norm_colmap(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def _get_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cmap = _norm_colmap(df)
    for n in names:
        if n.lower() in cmap:
            return cmap[n.lower()]
    return None


def read_meta_for_weights(meta_path: str, sample_order: List[str]) -> pd.DataFrame:
    """Read metadata and align it to the already loaded sample order.

    This intentionally keeps all user-provided columns (e.g. TumorFraction,
    Scenario, IsSynthetic, RecommendedTrainingWeight), while preserving the
    original training code's strict sample alignment.
    """
    if not meta_path:
        return pd.DataFrame({"Sample": list(sample_order)})
    try:
        meta = pd.read_csv(meta_path, sep=None, engine="python")
    except Exception:
        meta = pd.read_csv(meta_path)
    meta.columns = [str(c).strip() for c in meta.columns]
    sample_col = _get_col(meta, ["Sample", "sample", "SampleID", "sample_id"])
    if sample_col is None:
        raise ValueError("meta file must contain a Sample column for sample-weight alignment")
    meta[sample_col] = meta[sample_col].astype(str)
    meta = meta.drop_duplicates(subset=[sample_col], keep="first")
    aligned = pd.DataFrame({"Sample": list(map(str, sample_order))})
    aligned = aligned.merge(meta, left_on="Sample", right_on=sample_col, how="left", suffixes=("", "_meta"))
    # If the source Sample column had a different name, keep the canonical one.
    if sample_col != "Sample" and sample_col in aligned.columns:
        aligned = aligned.drop(columns=[sample_col])
    return aligned


def infer_is_synthetic(
    meta: pd.DataFrame,
    samples: List[str],
    synthetic_prefix: str = "SIM_",
) -> np.ndarray:
    samples_arr = np.asarray(samples, dtype=object)
    is_syn = np.zeros(len(samples_arr), dtype=bool)
    if synthetic_prefix:
        is_syn |= np.char.startswith(samples_arr.astype(str), str(synthetic_prefix))

    platform_col = _get_col(meta, ["Platform", "platform", "SourcePlatform", "source_platform"])
    if platform_col is not None:
        plat = meta[platform_col].fillna("").astype(str).str.upper().to_numpy(dtype=object)
        # ARRAY/Illumina EPIC are treated as real reference unless explicitly marked synthetic.
        is_syn |= np.isin(plat, ["TAPS", "WGBS", "ONT", "NANOPORE", "LONGREAD_TAPS", "SIM"])

    src_col = _get_col(meta, ["SourceDomain", "source_domain", "Domain", "domain"])
    if src_col is not None:
        src = meta[src_col].fillna("").astype(str).str.lower().to_numpy(dtype=object)
        is_syn |= np.array([("sim" in s) or ("synthetic" in s) or ("in_silico" in s) for s in src], dtype=bool)

    scenario_col = _get_col(meta, ["Scenario", "scenario", "SimulationScenario", "simulation_scenario"])
    if scenario_col is not None:
        sc = meta[scenario_col].fillna("").astype(str).str.lower().to_numpy(dtype=object)
        is_syn |= np.array([("sim" in s) or ("tumor" in s) or ("control" in s) or ("artifact" in s) for s in sc], dtype=bool)

    explicit_col = _get_col(meta, ["IsSynthetic", "is_synthetic", "Synthetic", "synthetic"])
    if explicit_col is not None:
        val = meta[explicit_col].fillna("").astype(str).str.lower().to_numpy(dtype=object)
        is_syn |= np.isin(val, ["1", "true", "yes", "y", "synthetic", "sim", "in_silico"])
    return is_syn


def resolve_sample_weight_vector(
    meta: pd.DataFrame,
    samples: List[str],
    y: Optional[np.ndarray] = None,
    sample_weight_col: str = "",
    platform_weight_map: str = "",
    sample_regex_weight_map: str = "",
    synthetic_prefix: str = "SIM_",
    synthetic_weight: float = 0.35,
    real_weight: float = 1.0,
    low_tf_weight: float = 0.10,
    boundary_tf_weight: float = 0.45,
    high_tf_weight: float = 0.35,
    decision_tf: float = 0.10,
    auto_weight_synthetic: bool = False,
    use_class_balance: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a per-sample weight vector for mixed real/simulated training.

    Priority:
    1) Explicit sample_weight_col if provided and present in metadata.
    2) Optional automatic domain/tumor-fraction weights for synthetic data.
    3) Optional platform and sample-regex multiplicative adjustments.
    4) Optional class-balance multiplier.

    Scientific intent: real Heidelberg array samples remain the anchor domain;
    in-silico samples regularize cross-platform and tumor-purity robustness but
    should not dominate the empirical methylation reference distribution.
    """
    n = len(samples)
    weights = np.full(n, float(real_weight), dtype=np.float64)
    reasons = np.array(["real_default"] * n, dtype=object)
    meta = meta.copy() if meta is not None else pd.DataFrame({"Sample": list(samples)})

    explicit_used = False
    if sample_weight_col:
        col = _get_col(meta, [sample_weight_col])
        if col is not None:
            explicit = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(explicit) & (explicit > 0)
            weights[ok] = explicit[ok]
            reasons[ok] = f"metadata:{col}"
            explicit_used = bool(ok.any())
        else:
            # Do not fail: this keeps old runs compatible when only real array data are used.
            explicit_used = False

    is_syn = infer_is_synthetic(meta, samples, synthetic_prefix=synthetic_prefix)

    if auto_weight_synthetic and not explicit_used:
        weights[is_syn] = float(synthetic_weight)
        reasons[is_syn] = "synthetic_default"

        scenario_col = _get_col(meta, ["Scenario", "scenario", "SimulationScenario", "simulation_scenario"])
        scenario = np.array([""] * n, dtype=object)
        if scenario_col is not None:
            scenario = meta[scenario_col].fillna("").astype(str).str.lower().to_numpy(dtype=object)

        # Control/artifact simulations are useful as negative/no-call stressors,
        # but should not overwhelm true tumor subtype boundaries.
        control_like = is_syn & np.array([("control" in s) for s in scenario], dtype=bool)
        artifact_like = is_syn & np.array([("artifact" in s) or ("fp" in s) or ("false" in s) for s in scenario], dtype=bool)
        weights[control_like] = min(float(synthetic_weight), float(high_tf_weight))
        reasons[control_like] = "synthetic_control"
        weights[artifact_like] = min(float(synthetic_weight), float(low_tf_weight) * 2.0)
        reasons[artifact_like] = "synthetic_artifact"

        tf_col = _get_col(meta, ["TumorFraction", "tumor_fraction", "Purity", "TumorPurity", "tf"])
        if tf_col is not None:
            tf = pd.to_numeric(meta[tf_col], errors="coerce").to_numpy(dtype=np.float64)
            # Accept both [0,1] and percent-like [0,100] encodings.
            tf = np.where(tf > 1.5, tf / 100.0, tf)
            finite = np.isfinite(tf)
            low = is_syn & finite & (tf < float(decision_tf))
            boundary = is_syn & finite & (tf >= float(decision_tf)) & (tf < 0.20)
            high = is_syn & finite & (tf >= 0.20)
            weights[low] = float(low_tf_weight)
            reasons[low] = "synthetic_low_tf"
            weights[boundary] = float(boundary_tf_weight)
            reasons[boundary] = "synthetic_boundary_tf"
            weights[high] = float(high_tf_weight)
            reasons[high] = "synthetic_high_tf"

    # Multiplicative platform factors, useful after empirical TAPS/WGBS/ONT lockbox calibration.
    pmap = parse_key_value_weights(platform_weight_map)
    platform_col = _get_col(meta, ["Platform", "platform", "SourcePlatform", "source_platform"])
    if pmap and platform_col is not None:
        plat = meta[platform_col].fillna("").astype(str).to_numpy(dtype=object)
        for k, fac in pmap.items():
            idx = np.array([str(x).lower() == str(k).lower() for x in plat], dtype=bool)
            weights[idx] *= float(fac)
            reasons[idx] = np.char.add(reasons[idx].astype(str), f"*platform:{k}")

    # Multiplicative regex factors for hand-curated exceptions.
    rmap = parse_key_value_weights(sample_regex_weight_map)
    if rmap:
        sample_strings = np.asarray(samples, dtype=str)
        for pat, fac in rmap.items():
            rx = re.compile(pat)
            idx = np.array([bool(rx.search(s)) for s in sample_strings], dtype=bool)
            weights[idx] *= float(fac)
            reasons[idx] = np.char.add(reasons[idx].astype(str), f"*regex:{pat}")

    if use_class_balance:
        if y is None:
            raise ValueError("use_class_balance=True requires encoded labels y")
        cw = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
        weights *= cw
        reasons = np.char.add(reasons.astype(str), "*class_balance")

    # Guard against accidental zero/negative/NaN weights.
    weights = np.asarray(weights, dtype=np.float64)
    weights[~np.isfinite(weights)] = float(real_weight)
    weights = np.clip(weights, 1e-6, None)

    report = {
        "enabled_by_explicit_column": bool(explicit_used),
        "auto_weight_synthetic": bool(auto_weight_synthetic),
        "sample_weight_col": str(sample_weight_col or ""),
        "n_samples": int(n),
        "n_synthetic_inferred": int(is_syn.sum()),
        "min_weight": float(np.min(weights)) if n else None,
        "median_weight": float(np.median(weights)) if n else None,
        "max_weight": float(np.max(weights)) if n else None,
        "mean_weight": float(np.mean(weights)) if n else None,
        "sum_weight": float(np.sum(weights)) if n else None,
        "reason_counts": {str(k): int(v) for k, v in pd.Series(reasons).value_counts().to_dict().items()},
        "defaults": {
            "real_weight": float(real_weight),
            "synthetic_weight": float(synthetic_weight),
            "low_tf_weight": float(low_tf_weight),
            "boundary_tf_weight": float(boundary_tf_weight),
            "high_tf_weight": float(high_tf_weight),
            "decision_tf": float(decision_tf),
        },
    }
    return weights, report


def write_sample_weight_audit(
    outdir: str,
    samples: List[str],
    weights: np.ndarray,
    report: Dict[str, Any],
    meta: Optional[pd.DataFrame] = None,
) -> None:
    import os

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({"Sample": list(samples), "TrainingWeight": np.asarray(weights, dtype=float)})
    if meta is not None:
        keep_cols = []
        for c in ["Types", "Platform", "Material", "Scenario", "TumorFraction", "TumorFractionBin", "IsSynthetic", "SourceDomain"]:
            cc = _get_col(meta, [c])
            if cc is not None and cc not in keep_cols:
                keep_cols.append(cc)
        if keep_cols:
            extra = meta[keep_cols].copy()
            for c in extra.columns:
                if c in df.columns:
                    extra = extra.drop(columns=[c])
            df = pd.concat([df, extra.reset_index(drop=True)], axis=1)
    df.to_csv(os.path.join(outdir, "sample_weights.csv"), index=False)
    with open(os.path.join(outdir, "sample_weight_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
