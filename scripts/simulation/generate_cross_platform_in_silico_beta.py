#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-platform in silico methylation beta simulator with source and quality-control auditing.

Designed for CNS tumor methylation classifier development when the reference training
data are Heidelberg-style Illumina array beta matrices, while deployment/validation may
include TAPS, WGBS and ONT methylation calls.

Core scientific model
---------------------
1) Source governance:
   - Tumor and control source samples can be restricted by a SourceSplit metadata column.
   - Each simulated latent profile records the exact source samples and weights in --out_source_map.
   - Tumor profiles can use an anchor-patient model instead of averaging many patients.
   - Controls retain semantic family/subtype metadata while the classifier target can remain CONTROL.

2) Plasma-background hierarchy:
   - Healthy plasma cfDNA can be used as an anchor family.
   - Leukocyte/cell-type references can be used as background components.
   - Normal CNS tissue and reactive microenvironment samples can remain real negative controls without
     being forced into every cfDNA background mixture.
   - Background families are sampled hierarchically, preventing large families from dominating simply
     because they contain more array samples.

3) Platform observation:
   The same latent biological profile can be rendered as TAPS, WGBS and ONT using:
       observed_probability = sensitivity * latent_beta + false_positive * (1 - latent_beta)

4) Measurement:
   - Production default is beta-binomial finite-depth measurement.
   - Zero-depth values can be retained as missing, floored, or resampled.
   - Gaussian mode remains available and can use depth-calibrated variance.
   - Extra missingness is separate from depth-derived missingness.

5) Reproducibility:
   - Paired latent profiles are retained across platforms.
   - Source maps, complete parameter tables and a JSON manifest are emitted.
   - Generator split, technical batch and source provenance are written to metadata.

Recommended primary design
--------------------------
Use --platforms TAPS,WGBS,ONT --platform_design paired_latent.
This renders the same latent biological mixtures separately through platform-specific mechanisms
instead of collapsing platforms into an artificial composite platform.
"""

from __future__ import annotations

# Public packaging note:
# This source retains the audited generation engine and formal-run safeguards used for
# donor-held-out SourceSplit cohorts. The stable public interface includes source-pool
# auditing, donor-unique background sampling, tumor-role gating, platform-specific
# measurement controls, configurable tumor-fraction components, strict replicate-group
# isolation, balanced tumor-anchor scheduling, source-usage auditing, and observed-
# missingness reporting.


import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NA_VALUES = ["NA", "NaN", "nan", "N/A", "", "null", "None"]

ROLE_ALIASES = {
    "direct_background": "plasma_anchor",
    "background_component": "background_component",
    "stress_inflammation": "hard_negative",
    "stress_tissue_injury": "hard_negative",
    "brain_cell_reference": "negative_only",
    "brain_tissue_background": "negative_only",
    "non_tumor_reference": "negative_only",
    "background_validation": "negative_only",
    "plasma_anchor": "plasma_anchor",
    "hard_negative": "hard_negative",
    "negative_only": "negative_only",
    "technical_only": "technical_only",
}

TRUE_VALUES = {"1", "true", "t", "yes", "y", "include", "included", "pass", "ok"}


@dataclass
class PlatformPreset:
    name: str
    observation_model: str

    # P(call methylated | truly methylated), sampled per synthetic sample.
    sens_min: float
    sens_mean: float
    sens_max: float
    sens_kappa: float

    # P(call methylated | truly unmethylated), sampled per synthetic sample.
    fp_min: float
    fp_mean: float
    fp_max: float
    fp_kappa: float

    # Per-locus jitter around sample-level sensitivity/FP; useful for chemistry/caller locus bias.
    locus_sens_sd: float
    locus_fp_sd: float

    # Sample-level domain/batch bias on logit(beta).
    sample_logit_bias_sd: float

    # Beta-space noise if --measurement_model gaussian.
    gaussian_noise_kappa: float

    # Sequencing-like depth if --measurement_model binomial/beta_binomial.
    depth_mean: float
    depth_cv: float
    depth_min: int
    depth_max: int

    # Optional stress-test missingness. Ignored when --missing_mode none.
    missing_min: float
    missing_mean: float
    missing_max: float
    missing_kappa: float


PLATFORM_PRESETS: Dict[str, PlatformPreset] = {
    "TAPS": PlatformPreset(
        name="TAPS",
        observation_model="taps_direct",
        sens_min=0.70, sens_mean=0.85, sens_max=0.99, sens_kappa=20.0,
        fp_min=0.0005, fp_mean=0.0030, fp_max=0.0100, fp_kappa=45.0,
        locus_sens_sd=0.025, locus_fp_sd=0.0015, sample_logit_bias_sd=0.06,
        gaussian_noise_kappa=250.0,
        depth_mean=35.0, depth_cv=0.65, depth_min=2, depth_max=300,
        missing_min=0.005, missing_mean=0.06, missing_max=0.35, missing_kappa=24.0,
    ),
    "WGBS": PlatformPreset(
        name="WGBS",
        observation_model="bisulfite_like",
        # Methylated C should be retained/called as methylated. Small methylated loss is allowed.
        sens_min=0.975, sens_mean=0.987, sens_max=0.9998, sens_kappa=90.0,
        # Incomplete conversion of unmethylated C creates false-positive methylation.
        fp_min=0.0005, fp_mean=0.0030, fp_max=0.0200, fp_kappa=60.0,
        locus_sens_sd=0.004, locus_fp_sd=0.0020, sample_logit_bias_sd=0.04,
        gaussian_noise_kappa=180.0,
        depth_mean=18.0, depth_cv=0.85, depth_min=1, depth_max=180,
        missing_min=0.02, missing_mean=0.16, missing_max=0.55, missing_kappa=18.0,
    ),
    "ONT": PlatformPreset(
        name="ONT",
        observation_model="native_caller",
        # Native methylation caller sensitivity; adjust by caller/chemistry/model if known.
        sens_min=0.90, sens_mean=0.965, sens_max=0.995, sens_kappa=35.0,
        # Native caller false methylation/background; can be higher than array/TAPS at low-abundance sites.
        fp_min=0.002, fp_mean=0.012, fp_max=0.050, fp_kappa=30.0,
        locus_sens_sd=0.020, locus_fp_sd=0.0080, sample_logit_bias_sd=0.12,
        gaussian_noise_kappa=90.0,
        depth_mean=10.0, depth_cv=1.0, depth_min=1, depth_max=120,
        missing_min=0.05, missing_mean=0.25, missing_max=0.70, missing_kappa=12.0,
    ),
    "ARRAY": PlatformPreset(
        name="ARRAY",
        observation_model="array_like",
        sens_min=0.995, sens_mean=1.0, sens_max=1.0, sens_kappa=100.0,
        fp_min=0.0, fp_mean=0.0002, fp_max=0.0010, fp_kappa=80.0,
        locus_sens_sd=0.0, locus_fp_sd=0.0, sample_logit_bias_sd=0.015,
        gaussian_noise_kappa=600.0,
        depth_mean=0.0, depth_cv=0.0, depth_min=0, depth_max=0,
        missing_min=0.0, missing_mean=0.0, missing_max=0.02, missing_kappa=80.0,
    ),
}


TF_COMPONENTS_DEFAULT = [
    # name, lower, upper, beta_a, beta_b, mixture_weight
    ("ultra_low", 0.001, 0.030, 1.5, 5.5, 0.16),
    ("low",       0.030, 0.100, 2.0, 4.0, 0.30),
    ("boundary",  0.080, 0.200, 2.5, 2.5, 0.32),
    ("mid",       0.200, 0.500, 2.5, 3.5, 0.16),
    ("high",      0.500, 0.850, 2.0, 2.0, 0.06),
]


def parse_csv_list(x: str) -> List[str]:
    if x is None:
        return []
    return [t.strip() for t in str(x).split(",") if t.strip()]


def normalize_role(value: object) -> str:
    key = str(value).strip().lower()
    return ROLE_ALIASES.get(key, key)


def _truthy_mask(series: pd.Series, accepted: Sequence[str]) -> pd.Series:
    allowed = {str(x).strip().lower() for x in accepted}
    return series.fillna("").astype(str).str.strip().str.lower().isin(allowed)


def filter_metadata(meta: pd.DataFrame, args) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply explicit inclusion/QC filters without silently dropping unannotated metadata."""
    report: Dict[str, object] = {"input_rows": int(len(meta)), "filters": []}
    out = meta.copy()

    include_col = str(getattr(args, "include_column", "") or "").strip()
    if include_col and include_col in out.columns:
        vals = parse_csv_list(getattr(args, "include_values", "YES,TRUE,1,Y"))
        keep = _truthy_mask(out[include_col], vals)
        report["filters"].append({
            "column": include_col, "kept": int(keep.sum()), "removed": int((~keep).sum())
        })
        out = out.loc[keep].copy()

    qc_col = str(getattr(args, "qc_status_column", "") or "").strip()
    if qc_col and qc_col in out.columns:
        vals = parse_csv_list(getattr(args, "qc_pass_values", "PASS,OK,INCLUDE"))
        keep = _truthy_mask(out[qc_col], vals)
        report["filters"].append({
            "column": qc_col, "kept": int(keep.sum()), "removed": int((~keep).sum())
        })
        out = out.loc[keep].copy()

    report["output_rows"] = int(len(out))
    if out.empty:
        raise ValueError("No metadata rows remain after inclusion/QC filtering.")
    return out, report


def validate_metadata_governance(meta: pd.DataFrame, args) -> Dict[str, object]:
    """Validate formal metadata governance before source-pool selection.

    Formal source isolation has two non-negotiable units:
    (1) donor identity and (2) any declared technical/biological replicate group.
    Neither may occur in more than one SourceSplit. This protects the later
    sim_train / sim_test / sim2 comparison from source leakage.
    """
    report: Dict[str, object] = {"warnings": [], "duplicate_groups": [], "split_isolation": {}}
    donor_col = str(getattr(args, "donor_column", "DonorID") or "").strip()
    split_col = str(getattr(args, "source_split_column", "SourceSplit") or "").strip()
    replicate_group_col = str(getattr(args, "replicate_group_column", "ReplicateGroup") or "").strip()

    def _check_split_isolation(column: str, label: str) -> None:
        if not column or column not in meta.columns or split_col not in meta.columns:
            report["warnings"].append(
                f"{label} split-isolation check skipped because column {column!r} or {split_col!r} is absent."
            )
            return
        tmp = meta[[column, split_col]].copy()
        tmp[column] = tmp[column].fillna("").astype(str).str.strip()
        tmp[split_col] = tmp[split_col].fillna("").astype(str).str.strip()
        tmp = tmp[
            (tmp[column] != "")
            & (tmp[split_col] != "")
            & (~tmp[split_col].str.lower().isin({"unassigned", "na", "none"}))
        ]
        multi = tmp.groupby(column)[split_col].nunique()
        bad = multi[multi > 1]
        report["split_isolation"][label] = {
            "column": column,
            "n_nonempty_groups": int(tmp[column].nunique()),
            "n_cross_split_groups": int(len(bad)),
            "examples": bad.index.astype(str).tolist()[:20],
        }
        if len(bad):
            raise ValueError(
                f"{label} leakage across SourceSplit detected for {len(bad)} groups. "
                f"Examples={bad.index.astype(str).tolist()[:20]}"
            )

    _check_split_isolation(donor_col, "DonorID")
    _check_split_isolation(replicate_group_col, "ReplicateGroup")

    if split_col in meta.columns:
        report["source_split_counts"] = (
            meta.groupby(split_col, dropna=False)
            .size()
            .reset_index(name="n")
            .to_dict(orient="records")
        )
        strat_cols = [
            c for c in [split_col, "SourceDataset", "Types", "BackgroundRole", "ControlFamily", "ControlSubtype"]
            if c in meta.columns
        ]
        if len(strat_cols) >= 2:
            report["source_split_strata_counts"] = (
                meta.groupby(strat_cols, dropna=False)
                .size()
                .reset_index(name="n")
                .to_dict(orient="records")
            )

    # Optional informative audit for repeated metadata combinations. This does not
    # redefine a technical replicate; ReplicateGroup above is the formal leakage unit.
    group_cols = [
        c for c in parse_csv_list(getattr(args, "replicate_group_columns", ""))
        if c in meta.columns
    ]
    if len(group_cols) >= 2:
        groups = meta.groupby(group_cols, dropna=False).size().reset_index(name="n")
        dup = groups[groups["n"] > 1].copy()
        if not dup.empty:
            report["duplicate_groups"] = dup.head(100).to_dict(orient="records")
            msg = (
                f"Found {len(dup)} repeated metadata combinations using {group_cols}. "
                "Review them as possible technical/biological repeats; formal cross-split "
                "protection is enforced using --replicate_group_column."
            )
            policy = str(getattr(args, "replicate_policy", "warn")).lower()
            if policy == "error":
                raise ValueError(msg)
            if policy == "warn":
                report["warnings"].append(msg)
                print(f"[WARN] {msg}", flush=True)

    mode = str(getattr(args, "missing_mode", "measurement_only")).lower()
    if mode == "mcar_beta":
        msg = (
            "missing_mode=mcar_beta is retained only as a backward-compatible alias of MCAR; "
            "it does not model a validated beta-dependent dropout mechanism."
        )
        report["warnings"].append(msg)
        print(f"[WARN] {msg}", flush=True)
    if mode in {"locus_weighted", "hybrid"} and not bool(
        getattr(args, "estimate_locus_missing_from_reference", False)
    ):
        msg = (
            f"missing_mode={mode} is being used without "
            "--estimate_locus_missing_from_reference; extra missingness will be uniform across loci."
        )
        report["warnings"].append(msg)
        print(f"[WARN] {msg}", flush=True)
    return report

def detect_sep(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "\t" in s:
                return "\t"
            if "," in s:
                return ","
            return r"\s+"
    return r"\s+"


def read_meta(path: Path) -> pd.DataFrame:
    sep = detect_sep(path)
    if sep == r"\s+":
        df = pd.read_csv(path, sep=sep, engine="python")
    else:
        df = pd.read_csv(path, sep=sep)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    req = {"Sample", "Types"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"meta is missing required columns: {sorted(missing)}; found={list(df.columns)}")
    df["Sample"] = df["Sample"].astype(str)
    df["Types"] = df["Types"].astype(str)
    return df


def read_beta_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline().rstrip("\n\r")
    cols = [c.strip() for c in line.split(",")]
    if len(cols) < 4:
        raise ValueError("beta CSV must have at least 3 ID columns plus >=1 sample column.")
    return cols


def clean_label(x: str) -> str:
    y = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x).strip())
    return y.strip("_") or "LABEL"


def find_background_labels(meta: pd.DataFrame, explicit: List[str], regex: str, match_columns: str) -> List[str]:
    all_types = list(dict.fromkeys(meta["Types"].astype(str).tolist()))
    if explicit:
        exp = set(explicit)
        found = [t for t in all_types if t in exp]
        missing = sorted(exp - set(found))
        if missing:
            raise ValueError(f"explicit --background_labels not found in meta Types: {missing[:20]}")
        return found

    mode = str(match_columns).lower()
    if mode not in {"types", "sample", "both"}:
        raise ValueError("--background_match_columns must be types, sample or both")
    pat = re.compile(regex, flags=re.IGNORECASE)
    found: List[str] = []
    if mode in {"types", "both"}:
        found.extend([t for t in all_types if pat.search(t)])
    if mode in {"sample", "both"}:
        mask = meta["Sample"].astype(str).map(lambda s: bool(pat.search(s)))
        found.extend(meta.loc[mask, "Types"].astype(str).tolist())
    s = set(found)
    out = [t for t in all_types if t in s]
    if not out:
        raise ValueError(
            "Could not infer background/control labels. Use --background_labels, or adjust "
            "--background_regex. Default supports CONTROL, CONTROL_XX, CONTR_XX, CONTR-XX, CTRL_XX, Normal, Healthy."
        )
    return out


def source_split_balance_report(
    meta: pd.DataFrame,
    args,
    stage: str = "filtered_all_splits",
    selected_pool: bool = False,
) -> Dict[str, object]:
    """Audit SourceSplit composition without assigning splits.

    Thresholds are evaluated on unique donor/source identities, not array rows.
    This prevents technical replicates from falsely satisfying minimum source-pool
    criteria. Dataset sharing across splits is reported—not treated as donor leakage—
    because the current reference design is donor-disjoint but not dataset-disjoint.
    """
    mode = str(getattr(args, "source_split_audit", "warn") or "warn").lower()
    report: Dict[str, object] = {
        "stage": stage,
        "selected_pool": bool(selected_pool),
        "enabled": mode != "off",
        "warnings": [],
    }
    if mode == "off":
        return report

    split_col = str(getattr(args, "source_split_column", "SourceSplit") or "").strip()
    type_col = "Types"
    role_col = str(getattr(args, "background_role_column", "BackgroundRole") or "").strip()
    family_col = str(getattr(args, "control_family_column", "ControlFamily") or "").strip()
    dataset_col = str(getattr(args, "source_dataset_column", "SourceDataset") or "").strip()
    donor_col = str(getattr(args, "donor_column", "DonorID") or "").strip()

    if split_col not in meta.columns:
        report["warnings"].append(f"SourceSplit audit skipped: column {split_col!r} not found.")
        return report

    work = meta.copy()
    work[split_col] = work[split_col].fillna("").astype(str).str.strip()
    if donor_col in work.columns:
        work["_AuditDonor"] = work[donor_col].fillna("").astype(str).str.strip()
    else:
        work["_AuditDonor"] = work["Sample"].astype(str)
    report["n_rows"] = int(len(work))
    report["n_unique_donors"] = int(work["_AuditDonor"].nunique())
    report["split_counts"] = work[split_col].value_counts(dropna=False).to_dict()
    report["split_unique_donors"] = (
        work.groupby(split_col, dropna=False)["_AuditDonor"].nunique().to_dict()
    )

    expected = parse_csv_list(getattr(args, "expected_source_splits", "sim_train,sim_test,sim2"))
    if expected and not selected_pool:
        present = {x for x in work[split_col].astype(str).unique().tolist() if x}
        missing = [x for x in expected if x not in present]
        if missing:
            report["warnings"].append(f"Expected SourceSplit values absent from metadata: {missing}")

        if dataset_col in work.columns:
            dataset_split_n = work.groupby(dataset_col)[split_col].nunique()
            shared = dataset_split_n[dataset_split_n > 1].index.astype(str).tolist()
            report["datasets_shared_across_splits"] = shared
            report["dataset_split_counts"] = {
                str(k): int(v) for k, v in dataset_split_n.items()
            }

    group_cols = [
        c for c in parse_csv_list(
            getattr(args, "split_balance_columns", "SourceDataset,BackgroundRole,ControlFamily,Types")
        )
        if c in work.columns
    ]
    if group_cols:
        cols = group_cols + [split_col]
        report["stratum_by_split_top"] = (
            work.groupby(cols, dropna=False)
            .agg(n_rows=("Sample", "size"), n_unique_donors=("_AuditDonor", "nunique"))
            .reset_index()
            .sort_values(group_cols + [split_col])
            .head(int(getattr(args, "split_balance_top_n", 200)))
            .to_dict(orient="records")
        )

    if selected_pool:
        requested_roles = {normalize_role(x) for x in parse_csv_list(getattr(args, "background_roles", ""))}
        if role_col in work.columns and requested_roles:
            roles = work[role_col].fillna("").astype(str).map(normalize_role)
            eligible_bg = work[roles.isin(requested_roles)].copy()
            n_bg_rows = int(len(eligible_bg))
            n_bg_donors = int(eligible_bg["_AuditDonor"].nunique())
            report["eligible_background_rows_in_selected_pool"] = n_bg_rows
            report["eligible_background_unique_donors_in_selected_pool"] = n_bg_donors
            if n_bg_donors < int(getattr(args, "min_background_sources", 8)):
                report["warnings"].append(
                    "Selected source pool has only "
                    f"{n_bg_donors} unique eligible background donors for roles={sorted(requested_roles)}."
                )
            anchor_family = str(getattr(args, "background_anchor_family", "CFDNA_HEALTHY") or "").strip()
            if family_col in eligible_bg.columns and anchor_family:
                anchor = eligible_bg[
                    eligible_bg[family_col].fillna("").astype(str).str.upper() == anchor_family.upper()
                ].copy()
                n_anchor_rows = int(len(anchor))
                n_anchor_donors = int(anchor["_AuditDonor"].nunique())
                report["eligible_anchor_rows_in_selected_pool"] = n_anchor_rows
                report["eligible_anchor_unique_donors_in_selected_pool"] = n_anchor_donors
                if n_anchor_donors < int(getattr(args, "min_anchor_sources", 1)):
                    report["warnings"].append(
                        "Selected source pool has "
                        f"{n_anchor_donors} unique donors from background_anchor_family={anchor_family!r}."
                    )

        if type_col in work.columns:
            explicit = parse_csv_list(getattr(args, "background_labels", ""))
            background_labels = find_background_labels(
                work,
                explicit=explicit,
                regex=str(getattr(args, "background_regex", "")),
                match_columns=str(getattr(args, "background_match_columns", "both")),
            )
            tumor_roles = {normalize_role(x) for x in parse_csv_list(getattr(args, "tumor_roles", "tumor_source"))}
            if role_col in work.columns and tumor_roles:
                tumor_meta = work[
                    work[role_col].fillna("").astype(str).map(normalize_role).isin(tumor_roles)
                ].copy()
            else:
                tumor_meta = work[~work[type_col].astype(str).isin(set(background_labels))].copy()
            tumor_counts = (
                tumor_meta.groupby(type_col)["_AuditDonor"].nunique().astype(int).to_dict()
            )
            report["tumor_source_unique_donor_counts_in_selected_pool"] = tumor_counts
            min_tumor = int(getattr(args, "min_tumor_sources_per_type", 3))
            low = {str(k): int(v) for k, v in tumor_counts.items() if int(v) < min_tumor}
            if low:
                report["warnings"].append(
                    f"Selected source pool has tumor classes below --min_tumor_sources_per_type={min_tumor} unique donors: {low}"
                )

    if report["warnings"]:
        msg = f"SourceSplit audit warnings at {stage}: " + " | ".join(report["warnings"])
        if mode == "error":
            raise ValueError(msg)
        if mode == "warn":
            print(f"[WARN] {msg}", flush=True)
    return report

def bounded_beta(rng: np.random.Generator, n: int, lo: float, hi: float, mean: float, kappa: float) -> np.ndarray:
    lo, hi, mean = float(lo), float(hi), float(mean)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if hi < lo:
        raise ValueError(f"Invalid bounded beta range: lo={lo}, hi={hi}")
    if abs(hi - lo) < 1e-12:
        return np.full(n, lo, dtype=np.float32)
    mu = (mean - lo) / (hi - lo)
    mu = float(np.clip(mu, 1e-4, 1.0 - 1e-4))
    k = max(float(kappa), 1e-3)
    a = mu * k
    b = (1.0 - mu) * k
    return (lo + rng.beta(a, b, size=n) * (hi - lo)).astype(np.float32)


def lognormal_from_mean_cv(rng: np.random.Generator, n: int, mean: float, cv: float, lo: int, hi: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int32)
    mean = float(mean)
    if mean <= 0:
        return np.zeros(n, dtype=np.int32)
    cv = max(float(cv), 1e-6)
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    x = rng.lognormal(mean=mu, sigma=sigma, size=n)
    x = np.rint(np.clip(x, int(lo), int(hi))).astype(np.int32)
    return x


def sample_tumor_fraction(
    rng: np.random.Generator,
    mode: str,
    n: int,
    min_tf: float,
    max_tf: float,
    fixed_tf: float,
    decision_tf: float,
    component_weight_map: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Sample tumor fraction with separate training-mixture and validation-bin semantics.

    ``clinical_mixture`` deliberately allows a boundary component that overlaps the
    3–10% and 10–20% clinical reporting intervals: it is a training augmentation.
    ``stratified_bins`` is different: it uses non-overlapping reporting bins, so a
    sim_test request gives balanced and interpretable LOD strata.
    """
    min_tf = float(min_tf)
    max_tf = float(max_tf)
    decision_tf = float(decision_tf)
    if n <= 0:
        return np.zeros(0, dtype=np.float32), []
    if max_tf < min_tf:
        raise ValueError("--max_tf must be >= --min_tf")
    if not (0.0 < decision_tf < 1.0):
        raise ValueError("--decision_tf must lie in (0, 1)")
    mode = str(mode).lower()
    if mode == "fixed":
        vals = np.full(n, fixed_tf, dtype=np.float32)
        return np.clip(vals, min_tf, max_tf), ["fixed"] * n
    if mode == "uniform":
        vals = rng.uniform(min_tf, max_tf, size=n).astype(np.float32)
        return vals, ["uniform"] * n
    if mode == "beta_low":
        x = rng.beta(2.0, 8.0, size=n)
        vals = min_tf + x * (max_tf - min_tf)
        return vals.astype(np.float32), ["beta_low"] * n
    if mode == "boundary":
        lo = max(min_tf, decision_tf * 0.55)
        hi = min(max_tf, decision_tf * 2.0)
        if hi <= lo:
            lo, hi = min_tf, max_tf
        x = rng.beta(2.5, 2.5, size=n)
        vals = lo + x * (hi - lo)
        return vals.astype(np.float32), ["boundary"] * n
    if mode not in {"clinical_mixture", "stratified_bins"}:
        raise ValueError(
            "--tf_mode must be clinical_mixture, stratified_bins, uniform, beta_low, boundary or fixed"
        )

    if mode == "stratified_bins":
        # Non-overlapping bins exactly match tf_bin() and the reporting plan.
        candidates = [
            ("ultra_low", min_tf, min(max_tf, 0.030), 1.5, 5.5),
            ("low", max(min_tf, 0.030), min(max_tf, decision_tf), 2.0, 4.0),
            ("boundary", max(min_tf, decision_tf), min(max_tf, 0.200), 2.5, 2.5),
            ("mid", max(min_tf, 0.200), min(max_tf, 0.500), 2.5, 3.5),
            ("high", max(min_tf, 0.500), max_tf, 2.0, 2.0),
        ]
        bins = [(name, lo, hi, a, b) for name, lo, hi, a, b in candidates if hi > lo + 1e-8]
        if not bins:
            return np.full(n, min_tf, dtype=np.float32), ["degenerate"] * n
        base = n // len(bins)
        remainder = n % len(bins)
        counts = np.full(len(bins), base, dtype=int)
        # Distribute the remainder randomly so no bin is systematically favored across seeds.
        if remainder:
            counts[rng.choice(len(bins), size=remainder, replace=False)] += 1
        vals: List[float] = []
        names: List[str] = []
        for (name, lo, hi, a, b), count in zip(bins, counts):
            if count <= 0:
                continue
            x = rng.beta(a, b, size=int(count))
            vals.extend((lo + x * (hi - lo)).tolist())
            names.extend([name] * int(count))
        order = rng.permutation(len(vals))
        return np.asarray([vals[i] for i in order], dtype=np.float32), [names[i] for i in order]

    # Training-only mixture. The boundary component intentionally overlaps adjacent
    # clinical bins to enrich the 10% QC decision region.
    component_weight_map = component_weight_map or {}
    comps = []
    for name, lo0, hi0, a, b, w_default in TF_COMPONENTS_DEFAULT:
        lo = max(min_tf, lo0)
        hi = min(max_tf, hi0)
        w = float(component_weight_map.get(str(name).upper(), w_default))
        if hi > lo + 1e-8 and w > 0:
            comps.append((name, lo, hi, a, b, w))
    if not comps:
        return np.full(n, min_tf, dtype=np.float32), ["degenerate"] * n
    weights = np.asarray([c[-1] for c in comps], dtype=np.float64)
    weights = weights / weights.sum()
    comp_idx = rng.choice(len(comps), size=n, replace=True, p=weights)
    vals = np.empty(n, dtype=np.float32)
    for k, comp in enumerate(comps):
        idx = np.where(comp_idx == k)[0]
        if idx.size == 0:
            continue
        _name, lo, hi, a, b, _w = comp
        x = rng.beta(a, b, size=idx.size)
        vals[idx] = (lo + x * (hi - lo)).astype(np.float32)
    names = [comps[int(i)][0] for i in comp_idx]
    return vals, names

def tf_bin(tf: float, decision_tf: float) -> str:
    tf = float(tf)
    if tf <= 0:
        return "control"
    if tf < 0.03:
        return "<3%"
    if tf < float(decision_tf):
        return f"3%-{int(float(decision_tf)*100)}%"
    if tf < 0.20:
        return f"{int(float(decision_tf)*100)}%-20%"
    if tf < 0.50:
        return "20%-50%"
    return ">50%"


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_locus_keys(id_part: Optional[pd.DataFrame]) -> Optional[List[str]]:
    """Return stable row keys from the first identifier columns of the beta matrix."""
    if id_part is None:
        return None
    return id_part.astype(str).agg("|".join, axis=1).tolist()


def stable_uniform01(keys: Sequence[str], salt: str) -> np.ndarray:
    """Deterministic U(0,1) values from row keys; independent of chunk size/order."""
    out = np.empty(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        h = hashlib.blake2b((str(salt) + "|" + str(k)).encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little", signed=False)
        out[i] = (v + 0.5) / 18446744073709551616.0
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def stable_normal(keys: Sequence[str], salt: str) -> np.ndarray:
    """Deterministic standard normal values from row keys via Box-Muller."""
    u1 = stable_uniform01(keys, salt + ":u1")
    u2 = stable_uniform01(keys, salt + ":u2")
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def locus_variability_multiplier(locus_variability: Optional[np.ndarray], args) -> np.ndarray | float:
    """Scale technical noise at reference-variable loci without changing the latent beta."""
    if locus_variability is None or not bool(getattr(args, "use_reference_variability_noise", False)):
        return 1.0
    v = np.asarray(locus_variability, dtype=np.float32).reshape(-1, 1)
    finite = np.isfinite(v[:, 0])
    if not finite.any():
        return 1.0
    med = float(np.nanmedian(v[finite]))
    mad = float(np.nanmedian(np.abs(v[finite] - med))) + 1e-6
    z = np.clip((v - med) / (1.4826 * mad), 0.0, float(args.reference_variability_z_clip))
    mult = 1.0 + float(args.reference_variability_noise_scale) * z
    return np.clip(mult, 1.0, float(args.reference_variability_max_multiplier)).astype(np.float32)


def dirichlet_weights(rng: np.random.Generator, n_sources: int, n_sims: int, alpha: float) -> np.ndarray:
    if n_sources <= 0:
        raise ValueError("n_sources must be >0")
    if n_sims <= 0:
        return np.zeros((n_sources, 0), dtype=np.float32)
    alpha = max(float(alpha), 1e-4)
    return rng.dirichlet(np.full(n_sources, alpha, dtype=np.float64), size=n_sims).T.astype(np.float32)


def dirichlet_weights_subset(
    rng: np.random.Generator,
    n_sources: int,
    n_sims: int,
    alpha: float,
    max_sources_per_latent: int,
) -> np.ndarray:
    """Dirichlet mixing with optional sparse source subsets.

    If max_sources_per_latent <= 0, all source profiles are used for each latent profile.
    If positive, each latent profile samples up to K source profiles without replacement.
    Across many simulations all source samples remain eligible, but each synthetic patient is
    less over-smoothed than a mixture of every reference profile.
    """
    if max_sources_per_latent is None or int(max_sources_per_latent) <= 0 or int(max_sources_per_latent) >= n_sources:
        return dirichlet_weights(rng, n_sources, n_sims, alpha)
    if n_sources <= 0:
        raise ValueError("n_sources must be >0")
    if n_sims <= 0:
        return np.zeros((n_sources, 0), dtype=np.float32)
    k_max = max(1, min(int(max_sources_per_latent), n_sources))
    alpha = max(float(alpha), 1e-4)
    out = np.zeros((n_sources, n_sims), dtype=np.float32)
    for j in range(n_sims):
        k_j = int(rng.integers(1, k_max + 1))
        idx = rng.choice(n_sources, size=k_j, replace=False)
        out[idx, j] = rng.dirichlet(np.full(k_j, alpha, dtype=np.float64)).astype(np.float32)
    return out



def parse_named_float_map(text: str) -> Dict[str, float]:
    """Parse NAME:value pairs, preserving case-insensitive matching."""
    out: Dict[str, float] = {}
    if text is None or not str(text).strip():
        return out
    for item in re.split(r"[,;]", str(text)):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid map item {item!r}; expected NAME:value")
        key, value = item.split(":", 1)
        value_f = float(value)
        if value_f < 0:
            raise ValueError(f"Map weights must be non-negative: {item!r}")
        out[key.strip().upper()] = value_f
    return out


def anchor_sparse_weights(
    rng: np.random.Generator,
    n_sources: int,
    n_sims: int,
    alpha: float,
    max_sources_per_latent: int,
    anchor_weight_min: float,
    anchor_weight_max: float,
    source_sampling: str = "uniform",
) -> np.ndarray:
    """Generate patient-like tumor mixtures with a dominant anchor source.

    ``balanced_anchor`` schedules anchors so each available tumor source appears
    either floor(n_latents/n_sources) or ceil(n_latents/n_sources) times. This
    avoids an accidental over-representation of a small held-out sim_test donor
    subset while preserving stochastic selection of any optional secondary sources.
    """
    if n_sources <= 0:
        raise ValueError("n_sources must be >0")
    if n_sims <= 0:
        return np.zeros((n_sources, 0), dtype=np.float32)
    k_max = n_sources if int(max_sources_per_latent) <= 0 else min(int(max_sources_per_latent), n_sources)
    k_max = max(1, k_max)
    lo = float(np.clip(anchor_weight_min, 0.0, 1.0))
    hi = float(np.clip(anchor_weight_max, lo, 1.0))
    alpha = max(float(alpha), 1e-4)
    mode = str(source_sampling or "uniform").lower()
    if mode not in {"uniform", "balanced_anchor"}:
        raise ValueError("--tumor_source_sampling must be uniform or balanced_anchor")

    if mode == "balanced_anchor":
        repeats = int(math.ceil(n_sims / n_sources))
        schedule = np.tile(rng.permutation(n_sources), repeats)[:n_sims]
        rng.shuffle(schedule)
    else:
        schedule = rng.integers(0, n_sources, size=n_sims)

    out = np.zeros((n_sources, n_sims), dtype=np.float32)
    all_idx = np.arange(n_sources, dtype=int)
    for j in range(n_sims):
        k_j = int(rng.integers(1, k_max + 1))
        anchor = int(schedule[j])
        if k_j == 1:
            out[anchor, j] = 1.0
            continue
        other_pool = all_idx[all_idx != anchor]
        rest = rng.choice(other_pool, size=k_j - 1, replace=False)
        anchor_w = float(rng.uniform(lo, hi))
        rest_w = rng.dirichlet(np.full(len(rest), alpha, dtype=np.float64)) * (1.0 - anchor_w)
        out[anchor, j] = np.float32(anchor_w)
        out[rest, j] = rest_w.astype(np.float32)
    return out

def _metadata_value(row: pd.Series, column: str, default: str) -> str:
    if column and column in row.index:
        value = str(row[column]).strip()
        if value and value.lower() not in {"nan", "none", "na"}:
            return value
    return default


def prepare_control_metadata(
    meta: pd.DataFrame,
    background_labels: Sequence[str],
    args,
) -> pd.DataFrame:
    """Create normalized control metadata and select eligible cfDNA background sources."""
    bg = meta[meta["Types"].astype(str).isin(set(map(str, background_labels)))].copy()
    if bg.empty:
        raise ValueError("No control/background rows remain after source filtering.")

    fam_col = str(getattr(args, "control_family_column", "") or "")
    subtype_col = str(getattr(args, "control_subtype_column", "") or "")
    role_col = str(getattr(args, "background_role_column", "") or "")
    dataset_col = str(getattr(args, "source_dataset_column", "") or "")
    donor_col = str(getattr(args, "donor_column", "") or "")
    state_col = str(getattr(args, "biological_state_column", "") or "")
    split_col = str(getattr(args, "source_split_column", "") or "")

    bg["_ControlFamily"] = [
        _metadata_value(row, fam_col, str(row["Types"])) for _, row in bg.iterrows()
    ]
    bg["_ControlSubtype"] = [
        _metadata_value(row, subtype_col, str(row["Types"])) for _, row in bg.iterrows()
    ]
    bg["_BackgroundRole"] = [
        normalize_role(_metadata_value(row, role_col, "background_component")) for _, row in bg.iterrows()
    ]
    bg["_SourceDataset"] = [
        _metadata_value(row, dataset_col, "UNKNOWN") for _, row in bg.iterrows()
    ]
    bg["_DonorID"] = [
        _metadata_value(row, donor_col, str(row["Sample"])) for _, row in bg.iterrows()
    ]
    bg["_BiologicalState"] = [
        _metadata_value(row, state_col, "unspecified") for _, row in bg.iterrows()
    ]
    bg["_SourceSplit"] = [
        _metadata_value(row, split_col, "") for _, row in bg.iterrows()
    ]

    requested_roles = {x.lower() for x in parse_csv_list(getattr(args, "background_roles", ""))}
    if requested_roles:
        eligible = bg[bg["_BackgroundRole"].astype(str).str.lower().isin(requested_roles)].copy()
        if eligible.empty:
            role_values = sorted(bg["_BackgroundRole"].astype(str).unique().tolist())
            raise ValueError(
                "No controls match --background_roles. "
                f"Requested={sorted(requested_roles)}; available={role_values}"
            )
    else:
        eligible = bg.copy()

    eligible = eligible.drop_duplicates(subset=["Sample"], keep="first").reset_index(drop=True)
    return eligible


def hierarchical_background_weights(
    rng: np.random.Generator,
    background_meta: pd.DataFrame,
    latent_ids: Sequence[str],
    args,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Hierarchically sample control families, then source samples within families."""
    n_sources = len(background_meta)
    n_sims = len(latent_ids)
    if n_sources <= 0:
        raise ValueError("No eligible background sources.")
    if n_sims <= 0:
        return np.zeros((n_sources, 0), dtype=np.float32), []

    mode = str(getattr(args, "background_sampling_mode", "hierarchical")).lower()
    max_sources = int(getattr(args, "background_max_sources_per_latent", 0))
    source_alpha = max(float(getattr(args, "background_dirichlet_alpha", 0.8)), 1e-4)

    if mode == "flat":
        W = dirichlet_weights_subset(rng, n_sources, n_sims, source_alpha, max_sources)
    elif mode != "hierarchical":
        raise ValueError("--background_sampling_mode must be flat or hierarchical")
    else:
        families = background_meta["_ControlFamily"].astype(str).to_numpy(dtype=object)
        unique_families = list(dict.fromkeys(families.tolist()))
        family_to_idx = {
            fam: np.where(families == fam)[0] for fam in unique_families
        }
        family_map = parse_named_float_map(getattr(args, "background_family_weight_map", ""))
        priors = np.array(
            [family_map.get(str(f).upper(), 1.0) for f in unique_families],
            dtype=np.float64,
        )
        if np.all(priors <= 0):
            raise ValueError("All background family prior weights are zero.")
        priors = priors / priors.sum()

        anchor_family = str(getattr(args, "background_anchor_family", "") or "").strip()
        anchor_index = None
        for i, fam in enumerate(unique_families):
            if anchor_family and str(fam).upper() == anchor_family.upper():
                anchor_index = i
                break
        if anchor_family and anchor_index is None and bool(getattr(args, "require_background_anchor", False)):
            raise ValueError(
                f"Required anchor family {anchor_family!r} is absent. "
                f"Available families={unique_families}"
            )

        max_families = int(getattr(args, "background_max_families_per_latent", 3))
        max_families = max(1, min(max_families, len(unique_families)))
        min_families = int(getattr(args, "background_min_families_per_latent", 1))
        min_families = max(1, min(min_families, max_families))
        family_concentration = max(
            float(getattr(args, "background_family_concentration", 12.0)), 1e-3
        )
        anchor_lo = float(np.clip(getattr(args, "background_anchor_weight_min", 0.65), 0, 1))
        anchor_hi = float(np.clip(getattr(args, "background_anchor_weight_max", 0.90), anchor_lo, 1))

        anchor_only_probability = float(np.clip(
            getattr(args, "background_anchor_only_probability", 0.0), 0.0, 1.0
        ))

        W = np.zeros((n_sources, n_sims), dtype=np.float32)
        for j in range(n_sims):
            # Separate the probability of retaining an empirical whole-cfDNA anchor
            # from the composition of constructed multi-family backgrounds. This avoids
            # an implicit 1/K frequency of pure anchors when min_families=1.
            if anchor_index is not None and rng.random() < anchor_only_probability:
                n_families_j = 1
            else:
                min_for_mixture = min_families
                if anchor_index is not None and max_families >= 2:
                    min_for_mixture = max(min_for_mixture, 2)
                n_families_j = int(rng.integers(min_for_mixture, max_families + 1))
            if anchor_index is not None:
                remaining_candidates = [i for i in range(len(unique_families)) if i != anchor_index]
                n_other = min(n_families_j - 1, len(remaining_candidates))
                chosen_family_idx = [anchor_index]
                if n_other > 0:
                    p = priors[remaining_candidates]
                    p = p / p.sum()
                    sampled = rng.choice(
                        remaining_candidates, size=n_other, replace=False, p=p
                    ).tolist()
                    chosen_family_idx.extend(sampled)
                anchor_w = float(rng.uniform(anchor_lo, anchor_hi))
                fam_weights = np.zeros(len(chosen_family_idx), dtype=np.float64)
                fam_weights[0] = anchor_w
                if len(chosen_family_idx) > 1:
                    other_priors = np.array(
                        [priors[i] for i in chosen_family_idx[1:]], dtype=np.float64
                    )
                    other_priors = other_priors / other_priors.sum()
                    other_draw = rng.dirichlet(
                        np.maximum(other_priors * family_concentration, 1e-3)
                    )
                    fam_weights[1:] = other_draw * (1.0 - anchor_w)
            else:
                chosen_family_idx = rng.choice(
                    len(unique_families),
                    size=n_families_j,
                    replace=False,
                    p=priors,
                ).tolist()
                chosen_priors = np.array([priors[i] for i in chosen_family_idx], dtype=np.float64)
                chosen_priors = chosen_priors / chosen_priors.sum()
                fam_weights = rng.dirichlet(
                    np.maximum(chosen_priors * family_concentration, 1e-3)
                )

            if max_sources <= 0:
                total_k = n_sources
            else:
                max_k = max(len(chosen_family_idx), min(max_sources, n_sources))
                total_k = int(rng.integers(len(chosen_family_idx), max_k + 1))
            slot_probs = fam_weights / fam_weights.sum()
            extra_slots = max(0, total_k - len(chosen_family_idx))
            slots = np.ones(len(chosen_family_idx), dtype=int)
            if extra_slots > 0:
                slots += rng.multinomial(extra_slots, slot_probs)

            for local_i, fam_i in enumerate(chosen_family_idx):
                fam = unique_families[fam_i]
                source_idx = family_to_idx[fam]
                k_fam = max(1, min(int(slots[local_i]), len(source_idx)))
                source_sampling = str(getattr(args, "background_source_sampling", "donor_unique")).lower()
                if source_sampling == "donor_unique":
                    fam_meta = background_meta.iloc[source_idx].copy()
                    donors_arr = fam_meta["_DonorID"].astype(str).to_numpy(dtype=object)
                    unique_donors = list(dict.fromkeys(donors_arr.tolist()))
                    k_donors = max(1, min(int(k_fam), len(unique_donors)))
                    chosen_donors = rng.choice(unique_donors, size=k_donors, replace=False)
                    chosen_sources_list = []
                    for donor in chosen_donors:
                        local_candidates = np.where(donors_arr == donor)[0]
                        if len(local_candidates) == 0:
                            continue
                        local_pick = int(rng.choice(local_candidates))
                        chosen_sources_list.append(int(source_idx[local_pick]))
                    chosen_sources = np.array(chosen_sources_list, dtype=int)
                    if chosen_sources.size == 0:
                        raise RuntimeError("donor_unique background sampling produced no sources.")
                    k_eff = int(chosen_sources.size)
                elif source_sampling == "donor_balanced":
                    donors = background_meta.iloc[source_idx]["_DonorID"].astype(str)
                    donor_counts = donors.map(donors.value_counts()).to_numpy(dtype=np.float64)
                    source_prob = 1.0 / np.maximum(donor_counts, 1.0)
                    source_prob = source_prob / source_prob.sum()
                    chosen_sources = rng.choice(source_idx, size=k_fam, replace=False, p=source_prob)
                    k_eff = int(k_fam)
                elif source_sampling == "dataset_donor_balanced":
                    # Equalize selection opportunity across contributing studies within a
                    # family, then sample a donor within the selected study. This prevents
                    # a large public cohort from acting as a de facto batch prior merely
                    # because it contributes more eligible arrays.
                    fam_meta = background_meta.iloc[source_idx].copy()
                    datasets_arr = fam_meta["_SourceDataset"].astype(str).to_numpy(dtype=object)
                    donors_arr = fam_meta["_DonorID"].astype(str).to_numpy(dtype=object)
                    selected_local = []
                    used_donors = set()
                    available_local = set(range(len(source_idx)))
                    while len(selected_local) < k_fam and available_local:
                        active_by_dataset = {}
                        for local in available_local:
                            donor = str(donors_arr[local])
                            if donor in used_donors:
                                continue
                            active_by_dataset.setdefault(str(datasets_arr[local]), []).append(local)
                        if not active_by_dataset:
                            break
                        dataset = str(rng.choice(list(active_by_dataset.keys())))
                        local = int(rng.choice(active_by_dataset[dataset]))
                        selected_local.append(local)
                        used_donors.add(str(donors_arr[local]))
                        available_local.remove(local)
                    chosen_sources = source_idx[np.asarray(selected_local, dtype=int)]
                    if chosen_sources.size == 0:
                        raise RuntimeError("dataset_donor_balanced background sampling produced no sources.")
                    k_eff = int(chosen_sources.size)
                elif source_sampling == "uniform":
                    chosen_sources = rng.choice(source_idx, size=k_fam, replace=False, p=None)
                    k_eff = int(k_fam)
                else:
                    raise ValueError("--background_source_sampling must be donor_unique, donor_balanced, dataset_donor_balanced or uniform")
                within = rng.dirichlet(
                    np.full(k_eff, source_alpha, dtype=np.float64)
                )
                W[chosen_sources, j] += (within * fam_weights[local_i]).astype(np.float32)

            col_sum = float(W[:, j].sum())
            if col_sum <= 0:
                raise RuntimeError("Background weight generation produced an empty column.")
            W[:, j] /= np.float32(col_sum)

    source_map: List[Dict[str, object]] = []
    for j, latent_id in enumerate(latent_ids):
        nz = np.where(W[:, j] > 0)[0]
        for i in nz:
            row = background_meta.iloc[int(i)]
            source_map.append({
                "LatentID": str(latent_id),
                "SourceRole": "BACKGROUND",
                "SourceSample": str(row["Sample"]),
                "SourceType": str(row["Types"]),
                "ControlFamily": str(row["_ControlFamily"]),
                "ControlSubtype": str(row["_ControlSubtype"]),
                "BackgroundRole": str(row["_BackgroundRole"]),
                "SourceDataset": str(row["_SourceDataset"]),
                "DonorID": str(row["_DonorID"]),
                "BiologicalState": str(row.get("_BiologicalState", "unspecified")),
                "SourceSplit": str(row.get("_SourceSplit", "")),
                "Weight": float(W[int(i), j]),
            })
    return W.astype(np.float32), source_map


def source_weight_map(
    W: np.ndarray,
    source_samples: Sequence[str],
    latent_ids: Sequence[str],
    source_role: str,
    source_type: str,
    source_meta: Optional[pd.DataFrame] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    lookup = {}
    if source_meta is not None and not source_meta.empty:
        lookup = {
            str(r["Sample"]): r for _, r in source_meta.drop_duplicates("Sample").iterrows()
        }
    for j, latent_id in enumerate(latent_ids):
        nz = np.where(W[:, j] > 0)[0]
        for i in nz:
            sample = str(source_samples[int(i)])
            meta_row = lookup.get(sample)
            row = {
                "LatentID": str(latent_id),
                "SourceRole": str(source_role),
                "SourceSample": sample,
                "SourceType": str(source_type),
                "ControlFamily": "",
                "ControlSubtype": "",
                "BackgroundRole": "",
                "SourceDataset": "",
                "DonorID": sample,
                "BiologicalState": "unspecified",
                "SourceSplit": "",
                "Weight": float(W[int(i), j]),
            }
            if meta_row is not None:
                for out_col, src_col in [
                    ("SourceDataset", "_SourceDataset"),
                    ("DonorID", "_DonorID"),
                    ("BiologicalState", "_BiologicalState"),
                    ("SourceSplit", "_SourceSplit"),
                ]:
                    if src_col in meta_row.index:
                        row[out_col] = str(meta_row[src_col])
            rows.append(row)
    return rows


def weighted_nan_average(X: np.ndarray, W: np.ndarray, min_weight: float = 1e-8) -> np.ndarray:
    if W.shape[1] == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    obs = ~np.isnan(X)
    X0 = np.nan_to_num(X, nan=0.0).astype(np.float32, copy=False)
    numerator = X0 @ W
    denominator = obs.astype(np.float32) @ W
    out = numerator / np.maximum(denominator, min_weight)
    out[denominator <= min_weight] = np.nan
    return out.astype(np.float32, copy=False)


def add_record(
    records: List[Dict[str, object]],
    rng: np.random.Generator,
    preset: PlatformPreset,
    platform: str,
    sid: str,
    label: str,
    material: str,
    scenario: str,
    tumor_fraction: float,
    tf_component: str,
    latent_id: str,
    args,
) -> None:
    sens = bounded_beta(rng, 1, preset.sens_min, preset.sens_max, preset.sens_mean, preset.sens_kappa)[0]
    fp = bounded_beta(rng, 1, preset.fp_min, preset.fp_max, preset.fp_mean, preset.fp_kappa)[0]
    if str(args.missing_mode).lower() in {"none", "measurement_only"}:
        miss = np.float32(0.0)
    else:
        miss = bounded_beta(rng, 1, preset.missing_min, preset.missing_max, preset.missing_mean, preset.missing_kappa)[0]
    depth = int(lognormal_from_mean_cv(rng, 1, preset.depth_mean, preset.depth_cv, preset.depth_min, preset.depth_max)[0])
    bias = float(rng.normal(0.0, preset.sample_logit_bias_sd))
    platform = platform.upper()
    records.append({
        "Sample": sid,
        "Types": label,
        "Platform": platform,
        "Material": material,
        "Scenario": scenario,
        "TumorFraction": float(tumor_fraction),
        "TumorFractionBin": tf_bin(float(tumor_fraction), float(args.decision_tf)),
        "TumorFractionComponent": tf_component,
        "ObservationModel": preset.observation_model,
        "MethylatedSensitivity": float(sens),
        "FalsePositiveRate": float(fp),
        "ConversionEfficiency": float(sens),  # compatibility alias; see README.
        "UnmodifiedFalsePositive": float(fp),  # compatibility alias; for WGBS this is incomplete conversion.
        "TAPSMethylatedConversionEfficiency": float(sens) if platform == "TAPS" else np.nan,
        "WGBSMethylatedRetention": float(sens) if platform == "WGBS" else np.nan,
        "ONTCallerSensitivity": float(sens) if platform == "ONT" else np.nan,
        "WGBSIncompleteConversionRate": float(fp) if platform == "WGBS" else np.nan,
        "TAPSUnmodifiedFalsePositive": float(fp) if platform == "TAPS" else np.nan,
        "ONTCallerFalsePositive": float(fp) if platform == "ONT" else np.nan,
        "TargetMissingRate": float(miss),
        "MeanDepth": int(depth),
        "SampleLogitBias": float(bias),
        "LatentID": latent_id,
        "PlatformDesign": str(args.platform_design),
        "Seed": int(args.seed),
        "GeneratorSplit": str(getattr(args, "generator_split", "unspecified")),
    })


def build_records(
    rng: np.random.Generator,
    tumor_types: List[str],
    background_labels: List[str],
    platform_presets: Dict[str, PlatformPreset],
    args,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    platforms = [p.upper() for p in parse_csv_list(args.platforms)]
    if not platforms:
        raise ValueError("--platforms cannot be empty")
    for p in platforms:
        if p not in platform_presets:
            raise ValueError(f"Unknown platform {p!r}; available={sorted(platform_presets)}")

    n_per_type = int(args.n_per_type_per_platform)
    n_boundary = int(args.n_boundary_per_type_per_platform)
    n_controls = int(args.n_control_per_platform)
    design = str(args.platform_design).lower()
    if design not in {"paired_latent", "independent"}:
        raise ValueError("--platform_design must be paired_latent or independent")

    def add_tumor_latent(t: str, group_name: str, i: int, tf: float, comp: str, platform: str, latent_id: str):
        safe_t = clean_label(t)
        if design == "paired_latent":
            sid = f"SIM_{platform}_{safe_t}_{group_name}_{i+1:05d}"
        else:
            sid = f"SIM_{platform}_{safe_t}_{group_name}_{i+1:05d}"
        add_record(
            records, rng, platform_presets[platform], platform, sid, t, args.material,
            f"tumor_{group_name}_{comp}", float(tf), comp, latent_id, args,
        )

    if design == "paired_latent":
        for t in tumor_types:
            safe_t = clean_label(t)
            for group_name, n, mode in [("clinical", n_per_type, args.tf_mode), ("boundary_boost", n_boundary, "boundary")]:
                if n <= 0:
                    continue
                tfs, comps = sample_tumor_fraction(
                    rng, mode, n, args.min_tf, args.max_tf, args.fixed_tf, args.decision_tf,
                    component_weight_map=parse_named_float_map(getattr(args, "tf_component_weight_map", "")),
                )
                for i in range(n):
                    latent_id = f"LATENT_{safe_t}_{group_name}_{i+1:05d}"
                    for platform in platforms:
                        add_tumor_latent(t, group_name, i, float(tfs[i]), comps[i], platform, latent_id)
        for i in range(n_controls):
            label = args.control_output_label
            safe_label = clean_label(label)
            latent_id = f"LATENT_{safe_label}_control_{i+1:05d}"
            for platform in platforms:
                sid = f"SIM_{platform}_{safe_label}_control_{i+1:05d}"
                add_record(records, rng, platform_presets[platform], platform, sid, label, args.material,
                           "control_pure", 0.0, "control", latent_id, args)
    else:
        for platform in platforms:
            for t in tumor_types:
                safe_t = clean_label(t)
                for group_name, n, mode in [("clinical", n_per_type, args.tf_mode), ("boundary_boost", n_boundary, "boundary")]:
                    if n <= 0:
                        continue
                    tfs, comps = sample_tumor_fraction(
                    rng, mode, n, args.min_tf, args.max_tf, args.fixed_tf, args.decision_tf,
                    component_weight_map=parse_named_float_map(getattr(args, "tf_component_weight_map", "")),
                )
                    for i in range(n):
                        latent_id = f"LATENT_{platform}_{safe_t}_{group_name}_{i+1:05d}"
                        add_tumor_latent(t, group_name, i, float(tfs[i]), comps[i], platform, latent_id)
            for i in range(n_controls):
                label = args.control_output_label
                safe_label = clean_label(label)
                sid = f"SIM_{platform}_{safe_label}_control_{i+1:05d}"
                latent_id = f"LATENT_{platform}_{safe_label}_control_{i+1:05d}"
                add_record(records, rng, platform_presets[platform], platform, sid, label, args.material,
                           "control_pure", 0.0, "control", latent_id, args)

    if not records:
        raise ValueError("No simulations requested; increase sample counts.")
    df = pd.DataFrame.from_records(records)

    # Mark poor-QC control stress samples.
    control_mask = df["TumorFraction"].astype(float) <= 0
    for platform, preset in platform_presets.items():
        idx = control_mask & (df["Platform"].astype(str).str.upper() == platform)
        bad = idx & (
            (df["FalsePositiveRate"].astype(float) > max(preset.fp_mean * 1.8, preset.fp_mean + 0.003)) |
            (df["MethylatedSensitivity"].astype(float) < preset.sens_mean * 0.95) |
            (df["TargetMissingRate"].astype(float) > preset.missing_mean * 1.5)
        )
        df.loc[bad, "Scenario"] = "control_artifact"

    # Batch-correlated sample bias. It is deliberately low-dimensional and recorded.
    n_batches = max(1, int(getattr(args, "technical_batches_per_platform", 1)))
    batch_sd = max(0.0, float(getattr(args, "batch_logit_bias_sd", 0.0)))
    df["TechnicalBatch"] = ""
    df["BatchLogitBias"] = np.float32(0.0)
    for platform in sorted(df["Platform"].astype(str).unique()):
        idx = np.where(df["Platform"].astype(str).to_numpy() == platform)[0]
        if idx.size == 0:
            continue
        batch_ids = np.arange(idx.size, dtype=int) % n_batches
        rng.shuffle(batch_ids)
        effects = rng.normal(0.0, batch_sd, size=n_batches).astype(np.float32)
        df.loc[df.index[idx], "TechnicalBatch"] = [
            f"{platform}_B{int(b)+1:02d}" for b in batch_ids
        ]
        df.loc[df.index[idx], "BatchLogitBias"] = effects[batch_ids]
    return df


def platform_measurement_value(args, preset: PlatformPreset, name: str, fallback: float) -> float:
    """Resolve a platform-specific measurement parameter, otherwise use the global fallback.

    This is deliberately separate from chemistry/caller sensitivity presets. Beta-binomial
    overdispersion and locus-depth dispersion represent platform-specific count/coverage
    behavior and should not be shared across TAPS, WGBS and ONT in formal runs.
    """
    key = f"{str(preset.name).lower()}_{name}"
    value = getattr(args, key, None)
    return float(fallback if value is None else value)


def apply_platform_observation(
    rng: np.random.Generator,
    true_beta: np.ndarray,
    records: pd.DataFrame,
    preset: PlatformPreset,
    args,
    id_part: Optional[pd.DataFrame] = None,
    locus_variability: Optional[np.ndarray] = None,
) -> np.ndarray:
    if true_beta.shape[1] == 0:
        return true_beta.astype(np.float32)

    # Preserve upstream reference-missing loci. In real Illumina array matrices,
    # a latent tumor/background weighted average can remain NaN when every selected
    # source is missing at a locus. Random binomial/beta-binomial samplers cannot
    # accept NaN probabilities, so use a finite placeholder only for stochastic
    # drawing and restore NaN at the end. This makes zero-depth missingness and
    # reference-missingness distinct, which is essential for array-to-sequencing
    # simulation.
    source_missing_mask = ~np.isfinite(true_beta)
    p = np.clip(np.nan_to_num(true_beta.astype(np.float32, copy=True), nan=0.0), 0.0, 1.0)
    n_rows, n_sims = p.shape

    sens = records["MethylatedSensitivity"].to_numpy(dtype=np.float32).reshape(1, -1)
    fp = records["FalsePositiveRate"].to_numpy(dtype=np.float32).reshape(1, -1)
    loc_mult = locus_variability_multiplier(locus_variability, args)
    locus_seed = getattr(args, "locus_effect_seed", None)
    locus_seed = int(args.seed if locus_seed is None else locus_seed)
    locus_keys = make_locus_keys(id_part) if bool(getattr(args, "deterministic_locus_effects", True)) else None

    if preset.locus_sens_sd > 0:
        if locus_keys is not None:
            z = stable_normal(locus_keys, f"{locus_seed}|{preset.name}|locus_sens").reshape(-1, 1).astype(np.float32)
            sj = z * float(preset.locus_sens_sd) * loc_mult
        else:
            sj = rng.normal(0.0, preset.locus_sens_sd, size=(n_rows, 1)).astype(np.float32) * loc_mult
        sens_eff = expit(safe_logit(sens) + sj)
        sens_eff = np.clip(sens_eff, preset.sens_min, preset.sens_max)
    else:
        sens_eff = sens

    if preset.locus_fp_sd > 0:
        if locus_keys is not None:
            z = stable_normal(locus_keys, f"{locus_seed}|{preset.name}|locus_fp").reshape(-1, 1).astype(np.float32)
            fj = z * float(preset.locus_fp_sd) * loc_mult
        else:
            fj = rng.normal(0.0, preset.locus_fp_sd, size=(n_rows, 1)).astype(np.float32) * loc_mult
        fp_eff = np.clip(fp + fj, preset.fp_min, preset.fp_max)
    else:
        fp_eff = fp

    q = sens_eff * p + fp_eff * (1.0 - p)
    q = np.clip(q, 0.0, 1.0)

    bias = records["SampleLogitBias"].to_numpy(dtype=np.float32).reshape(1, -1)
    if "BatchLogitBias" in records.columns:
        bias = bias + records["BatchLogitBias"].to_numpy(dtype=np.float32).reshape(1, -1)
    if np.any(np.abs(bias) > 0):
        q = expit(safe_logit(q) + bias).astype(np.float32)
        q = np.clip(q, 0.0, 1.0)

    model = str(args.measurement_model).lower()
    if source_missing_mask.any():
        q = q.astype(np.float32, copy=True)
        q[source_missing_mask] = np.nan

    if model == "none":
        return q.astype(np.float32)

    sample_depth = records["MeanDepth"].to_numpy(dtype=np.float32).reshape(1, -1)

    if model == "gaussian":
        variance_mode = str(getattr(args, "gaussian_variance_mode", "sample_depth")).lower()
        if variance_mode == "fixed_kappa":
            denom = np.full((1, n_sims), max(float(preset.gaussian_noise_kappa), 1.0) + 1.0, dtype=np.float32)
        elif variance_mode == "sample_depth":
            denom = np.maximum(sample_depth, 1.0) + 1.0
        else:
            raise ValueError("--gaussian_variance_mode must be sample_depth or fixed_kappa")
        sd = np.sqrt(np.clip(q * (1.0 - q), 0.0, 0.25) / denom).astype(np.float32)
        sd = sd * loc_mult
        q_finite = np.nan_to_num(q, nan=0.0)
        sd = np.nan_to_num(sd, nan=0.0)
        out = q_finite + rng.normal(0.0, 1.0, size=q.shape).astype(np.float32) * sd
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        if source_missing_mask.any():
            out[source_missing_mask] = np.nan
        return out

    if model not in {"binomial", "beta_binomial"}:
        raise ValueError("--measurement_model must be none, gaussian, binomial or beta_binomial")
    if sample_depth.max() <= 0:
        return q.astype(np.float32)

    depth_sigma = max(
        0.0,
        platform_measurement_value(
            args, preset, "locus_depth_sigma", float(args.locus_depth_sigma)
        ),
    )
    if locus_keys is not None:
        z_depth = stable_normal(locus_keys, f"{locus_seed}|{preset.name}|locus_depth").reshape(-1, 1).astype(np.float32)
        locus_depth_factor = np.exp(-0.5 * depth_sigma ** 2 + depth_sigma * z_depth).astype(np.float32)
    else:
        locus_depth_factor = rng.lognormal(
            mean=-0.5 * depth_sigma ** 2,
            sigma=depth_sigma,
            size=(n_rows, 1),
        ).astype(np.float32)

    lam = np.clip(sample_depth * locus_depth_factor, 0.0, float(args.depth_lambda_clip))
    depth = rng.poisson(lam=lam).astype(np.int32)

    policy = str(getattr(args, "zero_depth_policy", "missing")).lower()
    min_depth = max(1, int(getattr(args, "minimum_observed_depth", 1)))
    if policy == "floor":
        depth = np.maximum(depth, min_depth)
    elif policy == "resample":
        zero = depth < min_depth
        max_rounds = max(1, int(getattr(args, "zero_depth_resample_rounds", 4)))
        for _ in range(max_rounds):
            if not zero.any():
                break
            redraw = rng.poisson(lam=lam[zero]).astype(np.int32)
            depth[zero] = redraw
            zero = depth < min_depth
        if zero.any():
            depth[zero] = min_depth
    elif policy != "missing":
        raise ValueError("--zero_depth_policy must be missing, floor or resample")

    q_draw = np.clip(np.nan_to_num(q, nan=0.0), 0.0, 1.0).astype(np.float32)
    if model == "binomial":
        counts = rng.binomial(depth, q_draw).astype(np.float32)
    else:
        kappa = max(platform_measurement_value(args, preset, "beta_binomial_kappa", float(args.beta_binomial_kappa)), 1.0)
        a = np.clip(q_draw * kappa, 1e-4, None)
        b = np.clip((1.0 - q_draw) * kappa, 1e-4, None)
        q2 = np.clip(rng.beta(a, b).astype(np.float32), 0.0, 1.0)
        counts = rng.binomial(depth, q2).astype(np.float32)

    out = counts / np.maximum(depth, 1).astype(np.float32)
    if policy == "missing":
        out[depth < min_depth] = np.nan
    if source_missing_mask.any():
        out[source_missing_mask] = np.nan
    return out.astype(np.float32)


def infer_row_missing_weights(beta_csv: Path, id_cols: Sequence[str], needed_samples: Sequence[str], chunk_rows: int, enabled: bool) -> Optional[np.ndarray]:
    if not enabled:
        return None
    weights: List[np.ndarray] = []
    usecols = list(id_cols) + list(needed_samples)
    for chunk in pd.read_csv(beta_csv, usecols=usecols, na_values=NA_VALUES, chunksize=int(chunk_rows)):
        X = chunk[list(needed_samples)].to_numpy(dtype=np.float32, copy=False)
        miss = np.isnan(X).mean(axis=1).astype(np.float32)
        weights.append(miss + 0.05)
    if not weights:
        return None
    w = np.concatenate(weights)
    return (w / max(float(np.nanmean(w)), 1e-6)).astype(np.float32)


def apply_missingness(
    rng: np.random.Generator,
    values: np.ndarray,
    records: pd.DataFrame,
    row_missing_weight: Optional[np.ndarray],
    row_offset: int,
    args,
) -> np.ndarray:
    mode = str(args.missing_mode).lower()
    if mode in {"none", "measurement_only"} or values.shape[1] == 0:
        return values
    out = values.copy()
    n_rows, n_sims = out.shape
    target = records["TargetMissingRate"].to_numpy(dtype=np.float32).reshape(1, -1)
    if np.all(target <= 0):
        return out

    if row_missing_weight is None:
        weights = np.ones((n_rows, 1), dtype=np.float32)
    else:
        weights = row_missing_weight[row_offset:row_offset + n_rows].astype(np.float32).reshape(-1, 1)
        weights = weights / max(float(np.nanmean(weights)), 1e-6)
        weights = np.clip(weights, 0.05, float(args.max_locus_missing_weight))

    if mode in {"mcar", "mcar_beta", "locus_weighted", "hybrid"}:
        prob = np.clip(target * weights, 0.0, 0.98)
        mask = rng.random(out.shape, dtype=np.float32) < prob
        out[mask] = np.nan
        return out
    if mode == "feature_subset":
        weights1 = weights[:, 0].astype(np.float64)
        weights1 = np.maximum(weights1, 1e-12)
        weights1 = weights1 / weights1.sum()
        for j in range(n_sims):
            k = int(round(float(target[0, j]) * n_rows))
            if k <= 0:
                continue
            idx = rng.choice(n_rows, size=min(k, n_rows), replace=False, p=weights1)
            out[idx, j] = np.nan
        return out
    raise ValueError(
        "--missing_mode must be none, measurement_only, mcar, mcar_beta, "
        "locus_weighted, feature_subset or hybrid"
    )


def apply_overrides(args) -> Dict[str, PlatformPreset]:
    presets = {k: PlatformPreset(**asdict(v)) for k, v in PLATFORM_PRESETS.items()}

    def set_range(preset: PlatformPreset, prefix: str):
        # Sensitivity
        for attr in ["sens_min", "sens_mean", "sens_max", "sens_kappa", "fp_min", "fp_mean", "fp_max", "fp_kappa"]:
            val = getattr(args, f"{prefix}_{attr}", None)
            if val is not None:
                setattr(preset, attr, float(val))
        for attr in ["locus_sens_sd", "locus_fp_sd", "sample_logit_bias_sd", "gaussian_noise_kappa", "depth_mean", "depth_cv", "depth_min", "depth_max", "missing_min", "missing_mean", "missing_max", "missing_kappa"]:
            val = getattr(args, f"{prefix}_{attr}", None)
            if val is not None:
                if attr in {"depth_min", "depth_max"}:
                    setattr(preset, attr, int(val))
                else:
                    setattr(preset, attr, float(val))

    set_range(presets["TAPS"], "taps")
    set_range(presets["WGBS"], "wgbs")
    set_range(presets["ONT"], "ont")

    # Backward-compatible aliases retained for earlier parameter files.
    taps = presets["TAPS"]
    for old, new in [("taps_conv_min", "sens_min"), ("taps_conv_mean", "sens_mean"), ("taps_conv_max", "sens_max"), ("taps_conv_kappa", "sens_kappa")]:
        val = getattr(args, old, None)
        if val is not None:
            setattr(taps, new, float(val))
    if args.taps_unmodified_false_positive is not None:
        v = float(args.taps_unmodified_false_positive)
        taps.fp_min = taps.fp_mean = taps.fp_max = v
        taps.fp_kappa = 1e6

    wgbs = presets["WGBS"]
    for old, new in [
        ("wgbs_methylated_retention_min", "sens_min"),
        ("wgbs_methylated_retention_mean", "sens_mean"),
        ("wgbs_methylated_retention_max", "sens_max"),
        ("wgbs_methylated_retention_kappa", "sens_kappa"),
        ("wgbs_incomplete_conversion_min", "fp_min"),
        ("wgbs_incomplete_conversion_mean", "fp_mean"),
        ("wgbs_incomplete_conversion_max", "fp_max"),
        ("wgbs_incomplete_conversion_kappa", "fp_kappa"),
    ]:
        val = getattr(args, old, None)
        if val is not None:
            setattr(wgbs, new, float(val))

    # Validate ranges.
    for name, pr in presets.items():
        if not (0 <= pr.sens_min <= pr.sens_mean <= pr.sens_max <= 1):
            raise ValueError(f"{name} sensitivity must satisfy 0 <= min <= mean <= max <= 1: {pr}")
        if not (0 <= pr.fp_min <= pr.fp_mean <= pr.fp_max <= 1):
            raise ValueError(f"{name} false-positive must satisfy 0 <= min <= mean <= max <= 1: {pr}")
    return presets



def parse_weight_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if text is None or not str(text).strip():
        return out
    for item in re.split(r"[,;]", str(text)):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid weight-map item {item!r}; expected key:value")
        k, v = item.split(":", 1)
        out[k.strip().upper()] = float(v)
    return out


def assign_recommended_training_weights(records: pd.DataFrame, args) -> pd.DataFrame:
    """Annotate metadata with weights for merged real-array + in-silico training.

    Rationale: synthetic platform renders should regularize cross-platform and
    tumor-purity robustness, but should not dominate the empirical Heidelberg
    array reference distribution. Very-low tumor-fraction positive simulations
    are most useful for no-call/limit-of-detection calibration rather than as
    full-strength subtype labels.
    """
    df = records.copy()
    w = np.full(len(df), float(args.sim_training_weight), dtype=np.float64)
    scenario = df.get("Scenario", pd.Series([""] * len(df))).fillna("").astype(str).str.lower().to_numpy(dtype=object)
    tf = pd.to_numeric(df.get("TumorFraction", pd.Series([np.nan] * len(df))), errors="coerce").to_numpy(dtype=np.float64)
    low = np.isfinite(tf) & (tf > 0) & (tf < float(args.decision_tf))
    boundary = np.isfinite(tf) & (tf >= float(args.decision_tf)) & (tf < 0.20)
    high = np.isfinite(tf) & (tf >= 0.20)
    control = np.isfinite(tf) & (tf <= 0)
    artifact = np.array([("artifact" in x) or ("false" in x) or ("fp" in x) for x in scenario], dtype=bool)
    w[control] = float(args.control_training_weight)
    w[artifact] = float(args.artifact_training_weight)
    w[low] = float(args.low_tf_training_weight)
    w[boundary] = float(args.boundary_tf_training_weight)
    w[high] = float(args.high_tf_training_weight)
    pmap = parse_weight_map(getattr(args, "platform_training_weight_map", ""))
    if pmap:
        plat = df["Platform"].astype(str).str.upper().to_numpy(dtype=object)
        for k, fac in pmap.items():
            w[plat == k] *= float(fac)
    df["RecommendedTrainingWeight"] = np.clip(w, 1e-6, None).astype(np.float32)
    df["IsSynthetic"] = True
    df["SourceDomain"] = "in_silico_" + df["Platform"].astype(str)
    return df


def validate_formal_run_configuration(args, platforms: Sequence[str]) -> None:
    """Fail early when a claimed formal run relies on hidden platform defaults.

    The simulator retains defaults for backward compatibility only. Formal runs must
    make every platform calibration parameter explicit in the command/README so the
    generated cohort can be reproduced and scientifically audited.
    """
    if not bool(getattr(args, "require_explicit_platform_parameters", False)):
        return
    fields = [
        "sens_min", "sens_mean", "sens_max", "sens_kappa",
        "fp_min", "fp_mean", "fp_max", "fp_kappa",
        "locus_sens_sd", "locus_fp_sd", "sample_logit_bias_sd",
        "gaussian_noise_kappa", "depth_mean", "depth_cv", "depth_min",
        "depth_max", "missing_min", "missing_mean", "missing_max", "missing_kappa",
        "beta_binomial_kappa", "locus_depth_sigma",
    ]
    missing = []
    for platform in platforms:
        prefix = str(platform).lower()
        for field in fields:
            if getattr(args, f"{prefix}_{field}", None) is None:
                missing.append(f"--{prefix}_{field}")
    if missing:
        raise ValueError(
            "--require_explicit_platform_parameters was requested, but formal-run "
            f"parameters were omitted: {missing[:30]}"
            + (" ..." if len(missing) > 30 else "")
        )


def validate_formal_meta_schema(meta: pd.DataFrame, args) -> None:
    """Validate the fixed metadata contract used for formal simulation runs."""
    if not bool(getattr(args, "require_formal_meta_schema", False)):
        return
    required = [
        "Sample", "Types", "SourceDataset", "DonorID", "ReplicateGroup",
        "ControlFamily", "ControlSubtype", "BackgroundRole", "BiologicalState",
        "TissueOrCellType", "IncludeForSimulation", "QCStatus", "SourceSplit",
    ]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Formal metadata schema is missing required columns: {missing}")
    blank = {}
    for c in required:
        n_blank = int(
            meta[c].isna().sum()
            + meta[c].astype(str).str.strip().str.lower().isin({"", "nan", "none", "na"}).sum()
        )
        if n_blank:
            blank[c] = n_blank
    if blank:
        raise ValueError(f"Formal metadata schema has blank required values: {blank}")

    allowed_roles = {
        "tumor_source", "plasma_anchor", "background_component",
        "negative_only", "hard_negative", "technical_only",
    }
    roles = meta["BackgroundRole"].astype(str).map(normalize_role)
    invalid = sorted(set(roles) - allowed_roles)
    if invalid:
        raise ValueError(f"Formal metadata schema has unsupported BackgroundRole values: {invalid}")

    tumor_bad = meta[(roles == "tumor_source") & (meta["Types"].astype(str).str.upper() == str(args.control_output_label).upper())]
    control_bad = meta[(roles != "tumor_source") & (meta["Types"].astype(str).str.upper() != str(args.control_output_label).upper())]
    if not tumor_bad.empty or not control_bad.empty:
        raise ValueError(
            "Formal metadata role/type mismatch: tumor_source must have non-CONTROL Types, "
            "and non-tumor roles must have CONTROL Types."
        )


def build_source_usage_audit(source_map_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how frequently each source contributes to simulated latents."""
    if source_map_df.empty:
        return pd.DataFrame(
            columns=[
                "SourceRole", "TumorType", "SourceDataset", "DonorID",
                "SourceSample", "NLatents", "MeanWeight", "MinWeight",
                "MaxWeight", "NSourceSplits",
            ]
        )
    work = source_map_df.copy()
    if "TumorType" not in work.columns:
        work["TumorType"] = np.where(
            work.get("SourceRole", pd.Series("", index=work.index)).astype(str).eq("TUMOR"),
            work.get("SourceType", pd.Series("", index=work.index)).astype(str),
            "",
        )
    for col, default in [
        ("SourceRole", "UNKNOWN"), ("TumorType", ""), ("SourceDataset", "UNKNOWN"),
        ("DonorID", "UNKNOWN"), ("SourceSample", "UNKNOWN"), ("SourceSplit", ""),
    ]:
        if col not in work.columns:
            work[col] = default
    return (
        work.groupby(
            ["SourceRole", "TumorType", "SourceDataset", "DonorID", "SourceSample"],
            dropna=False,
        )
        .agg(
            NLatents=("LatentID", "nunique"),
            MeanWeight=("Weight", "mean"),
            MinWeight=("Weight", "min"),
            MaxWeight=("Weight", "max"),
            NSourceSplits=("SourceSplit", "nunique"),
        )
        .reset_index()
        .sort_values(["SourceRole", "TumorType", "NLatents", "SourceSample"], ascending=[True, True, False, True])
    )

def generate(args) -> None:
    beta_csv = Path(args.beta_csv)
    meta_path = Path(args.meta)
    out_beta = Path(args.out_beta_csv)
    out_meta = Path(args.out_meta)
    out_params = Path(args.out_params)
    out_manifest = Path(args.out_manifest) if args.out_manifest else None
    out_source_map = Path(args.out_source_map) if getattr(args, "out_source_map", "") else None
    out_validation = Path(args.out_validation_report) if getattr(args, "out_validation_report", "") else None
    out_source_usage_audit = Path(args.out_source_usage_audit) if getattr(args, "out_source_usage_audit", "") else None

    if not beta_csv.exists():
        raise FileNotFoundError(beta_csv)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    rng = np.random.default_rng(int(args.seed))
    meta = read_meta(meta_path)
    meta, filter_report = filter_metadata(meta, args)
    validate_formal_meta_schema(meta, args)
    governance_report = validate_metadata_governance(meta, args)
    source_split_balance_all = source_split_balance_report(meta, args, stage="filtered_all_splits", selected_pool=False)

    # Source-disjoint generation is implemented by preparing SourceSplit in metadata
    # and running train/calibration/test generation separately.
    split_col = str(getattr(args, "source_split_column", "") or "").strip()
    split_value = str(getattr(args, "source_split_value", "") or "").strip()
    if split_value:
        if not split_col or split_col not in meta.columns:
            raise ValueError(
                "--source_split_value was supplied but --source_split_column is absent "
                f"from metadata. column={split_col!r}"
            )
        keep_values = set(parse_csv_list(split_value))
        meta = meta[meta[split_col].astype(str).isin(keep_values)].copy()
        if meta.empty:
            raise ValueError(
                f"No metadata rows match {split_col} in {sorted(keep_values)}"
            )

    source_pool_balance_selected = source_split_balance_report(meta, args, stage="selected_source_pool", selected_pool=True)

    if meta["Sample"].astype(str).duplicated().any():
        dup = meta.loc[meta["Sample"].astype(str).duplicated(), "Sample"].astype(str).tolist()
        raise ValueError(f"Duplicate Sample IDs in metadata. Examples={dup[:10]}")

    cols = read_beta_header(beta_csv)
    id_cols = cols[:3]
    sample_cols = [str(c).strip() for c in cols[3:]]
    sample_set = set(sample_cols)

    missing_in_beta = [
        s for s in meta["Sample"].astype(str).tolist() if s not in sample_set
    ]
    if missing_in_beta:
        raise ValueError(
            f"{len(missing_in_beta)} meta samples missing from beta CSV columns. "
            f"Examples={missing_in_beta[:10]}"
        )

    background_labels = find_background_labels(
        meta,
        parse_csv_list(args.background_labels),
        args.background_regex,
        args.background_match_columns,
    )
    role_col_for_tumor = str(getattr(args, "background_role_column", "BackgroundRole") or "").strip()
    tumor_roles = {normalize_role(x) for x in parse_csv_list(getattr(args, "tumor_roles", "tumor_source"))}
    if role_col_for_tumor and role_col_for_tumor in meta.columns and tumor_roles:
        tumor_source_meta = meta[
            meta[role_col_for_tumor].fillna("").astype(str).map(normalize_role).isin(tumor_roles)
        ].copy()
    else:
        # Compatibility fallback only for metadata without a semantic role column.
        bg_set = set(background_labels)
        tumor_source_meta = meta[~meta["Types"].astype(str).isin(bg_set)].copy()
    tumor_types = list(pd.unique(tumor_source_meta["Types"].astype(str)))

    if args.include_types:
        include = set(parse_csv_list(args.include_types))
        tumor_types = [t for t in tumor_types if t in include]
    if args.exclude_types:
        exclude = set(parse_csv_list(args.exclude_types))
        tumor_types = [t for t in tumor_types if t not in exclude]
    if not tumor_types:
        raise ValueError("No tumor types selected after explicit --tumor_roles / include / exclude filtering.")

    samples_by_type: Dict[str, List[str]] = {}
    for t, sub in tumor_source_meta.groupby("Types", sort=False):
        samples_by_type[str(t)] = sub["Sample"].astype(str).tolist()

    dataset_col = str(getattr(args, "source_dataset_column", "SourceDataset") or "")
    donor_col = str(getattr(args, "donor_column", "DonorID") or "")
    state_col = str(getattr(args, "biological_state_column", "BiologicalState") or "")
    source_meta_all = meta.copy()
    source_meta_all["_SourceDataset"] = [
        _metadata_value(row, dataset_col, "UNKNOWN") for _, row in source_meta_all.iterrows()
    ]
    source_meta_all["_DonorID"] = [
        _metadata_value(row, donor_col, str(row["Sample"])) for _, row in source_meta_all.iterrows()
    ]
    source_meta_all["_BiologicalState"] = [
        _metadata_value(row, state_col, "unspecified") for _, row in source_meta_all.iterrows()
    ]
    source_meta_all["_SourceSplit"] = [
        _metadata_value(row, split_col, "") for _, row in source_meta_all.iterrows()
    ]

    background_meta = prepare_control_metadata(meta, background_labels, args)
    background_samples = background_meta["Sample"].astype(str).tolist()
    if not background_samples:
        raise ValueError("No eligible background samples after role filtering.")

    presets = apply_overrides(args)
    requested_platforms = [p.upper() for p in parse_csv_list(args.platforms)]
    validate_formal_run_configuration(args, requested_platforms)
    records = build_records(rng, tumor_types, background_labels, presets, args)
    records = assign_recommended_training_weights(records, args)

    latent_values = records["LatentID"].astype(str).to_numpy(dtype=object)
    plan: Dict[str, Dict[str, object]] = {}
    source_map_rows: List[Dict[str, object]] = []

    for t in tumor_types:
        rec_idx = np.where(records["Types"].to_numpy(dtype=object) == t)[0]
        if rec_idx.size == 0:
            continue
        tumor_samples = samples_by_type.get(t, [])
        if not tumor_samples:
            raise ValueError(f"No reference samples for tumor type {t}")

        unique_latents = list(dict.fromkeys(latent_values[rec_idx].tolist()))
        latent_to_col = {lat: i for i, lat in enumerate(unique_latents)}
        col_index = np.asarray(
            [latent_to_col[lat] for lat in latent_values[rec_idx]], dtype=np.int64
        )

        tumor_mode = str(getattr(args, "tumor_mixture_mode", "anchor")).lower()
        if tumor_mode == "anchor":
            W_tumor = anchor_sparse_weights(
                rng,
                len(tumor_samples),
                len(unique_latents),
                args.tumor_dirichlet_alpha,
                args.tumor_max_sources_per_latent,
                args.tumor_anchor_weight_min,
                args.tumor_anchor_weight_max,
                getattr(args, "tumor_source_sampling", "balanced_anchor"),
            )
        elif tumor_mode == "dirichlet":
            W_tumor = dirichlet_weights_subset(
                rng,
                len(tumor_samples),
                len(unique_latents),
                args.tumor_dirichlet_alpha,
                args.tumor_max_sources_per_latent,
            )
        else:
            raise ValueError("--tumor_mixture_mode must be anchor or dirichlet")

        W_background, bg_map = hierarchical_background_weights(
            rng, background_meta, unique_latents, args
        )
        source_map_rows.extend(bg_map)
        source_map_rows.extend(
            source_weight_map(
                W_tumor,
                tumor_samples,
                unique_latents,
                "TUMOR",
                t,
                source_meta=source_meta_all[source_meta_all["Types"].astype(str) == str(t)],
            )
        )

        plan[t] = {
            "record_indices": rec_idx,
            "tumor_samples": tumor_samples,
            "W_tumor_base": W_tumor,
            "W_background_base": W_background,
            "latent_col_index": col_index,
        }

    control_idx = np.where(
        records["TumorFraction"].to_numpy(dtype=float) <= 0
    )[0]
    if control_idx.size > 0:
        ctrl_latents = list(dict.fromkeys(latent_values[control_idx].tolist()))
        ctrl_latent_to_col = {lat: i for i, lat in enumerate(ctrl_latents)}
        ctrl_col_index = np.asarray(
            [ctrl_latent_to_col[lat] for lat in latent_values[control_idx]],
            dtype=np.int64,
        )
        W_control_base, control_map = hierarchical_background_weights(
            rng, background_meta, ctrl_latents, args
        )
        source_map_rows.extend(control_map)
    else:
        ctrl_latents = []
        ctrl_col_index = np.zeros(0, dtype=np.int64)
        W_control_base = np.zeros(
            (len(background_samples), 0), dtype=np.float32
        )

    source_map_df = pd.DataFrame(source_map_rows)
    if not source_map_df.empty:
        summary = (
            source_map_df.groupby(["LatentID", "SourceRole"], sort=False)
            .agg(
                SourceCount=("SourceSample", "nunique"),
                EffectiveSourceCount=("Weight", lambda x: float(1.0 / np.sum(np.square(x)))),
                MaxSourceWeight=("Weight", "max"),
            )
            .reset_index()
        )
        bg_summary = summary[summary["SourceRole"] == "BACKGROUND"].drop(
            columns=["SourceRole"]
        )
        bg_summary = bg_summary.rename(
            columns={
                "SourceCount": "BackgroundSourceCount",
                "EffectiveSourceCount": "BackgroundEffectiveSourceCount",
                "MaxSourceWeight": "BackgroundMaxSourceWeight",
            }
        )
        records = records.merge(bg_summary, on="LatentID", how="left")

        bg_rows = source_map_df[source_map_df["SourceRole"] == "BACKGROUND"].copy()
        if not bg_rows.empty:
            fam_weights = (
                bg_rows.groupby(["LatentID", "ControlFamily"], sort=False)["Weight"]
                .sum()
                .reset_index()
            )
            dominant = fam_weights.loc[
                fam_weights.groupby("LatentID")["Weight"].idxmax(),
                ["LatentID", "ControlFamily", "Weight"],
            ].rename(
                columns={
                    "ControlFamily": "DominantBackgroundFamily",
                    "Weight": "DominantBackgroundFamilyWeight",
                }
            )
            records = records.merge(dominant, on="LatentID", how="left")

    source_usage_audit_df = build_source_usage_audit(source_map_df)

    needed_samples: List[str] = []
    for t in tumor_types:
        needed_samples.extend(samples_by_type.get(t, []))
    needed_samples.extend(background_samples)
    seen = set()
    needed_samples = [
        s for s in needed_samples if not (s in seen or seen.add(s))
    ]

    row_missing_weight = infer_row_missing_weights(
        beta_csv,
        id_cols,
        needed_samples,
        int(args.chunk_rows),
        bool(args.estimate_locus_missing_from_reference),
    )

    for path in [out_beta, out_meta, out_params]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if out_manifest:
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
    if out_source_map:
        out_source_map.parent.mkdir(parents=True, exist_ok=True)
    if out_validation:
        out_validation.parent.mkdir(parents=True, exist_ok=True)
    if out_source_usage_audit:
        out_source_usage_audit.parent.mkdir(parents=True, exist_ok=True)
    if out_beta.exists():
        out_beta.unlink()

    sim_ids = records["Sample"].astype(str).tolist()
    dtype_map = {c: np.float32 for c in needed_samples}
    usecols = list(id_cols) + needed_samples
    first = True
    row_offset = 0
    observed_finite_count = np.zeros(len(sim_ids), dtype=np.int64)
    observed_total_count = np.zeros(len(sim_ids), dtype=np.int64)

    platform_to_indices = {
        p: np.where(
            records["Platform"].astype(str).str.upper().to_numpy() == p
        )[0]
        for p in sorted(set(records["Platform"].astype(str).str.upper()))
    }

    reader = pd.read_csv(
        beta_csv,
        usecols=usecols,
        dtype=dtype_map,
        na_values=NA_VALUES,
        chunksize=int(args.chunk_rows),
    )
    for chunk_id, chunk in enumerate(reader, start=1):
        id_part = chunk[list(id_cols)].copy()
        n_rows = int(chunk.shape[0])
        if bool(args.use_reference_variability_noise):
            X_sd = chunk[needed_samples].to_numpy(dtype=np.float32, copy=False)
            locus_variability = np.nanstd(X_sd, axis=1).astype(np.float32)
        else:
            locus_variability = None

        X_bg = chunk[background_samples].to_numpy(dtype=np.float32, copy=False)
        sim = np.full((n_rows, len(sim_ids)), np.nan, dtype=np.float32)

        for t, item in plan.items():
            rec_idx = item["record_indices"]
            X_t = chunk[item["tumor_samples"]].to_numpy(
                dtype=np.float32, copy=False
            )
            T_base = weighted_nan_average(X_t, item["W_tumor_base"])
            B_base = weighted_nan_average(X_bg, item["W_background_base"])
            col_index = item["latent_col_index"]
            T = T_base[:, col_index]
            B = B_base[:, col_index]
            f = records.iloc[rec_idx]["TumorFraction"].to_numpy(
                dtype=np.float32
            ).reshape(1, -1)
            mixed = f * T + (1.0 - f) * B
            sim[:, rec_idx] = mixed.astype(np.float32)

        if control_idx.size > 0:
            C_base = weighted_nan_average(X_bg, W_control_base)
            C = C_base[:, ctrl_col_index]
            sim[:, control_idx] = C.astype(np.float32)

        observed = np.full_like(sim, np.nan, dtype=np.float32)
        for platform, idx in platform_to_indices.items():
            if idx.size == 0:
                continue
            vals = sim[:, idx]
            rec_sub = records.iloc[idx]
            vals = apply_platform_observation(
                rng,
                vals,
                rec_sub,
                presets[platform],
                args,
                id_part=id_part,
                locus_variability=locus_variability,
            )
            vals = apply_missingness(
                rng, vals, rec_sub, row_missing_weight, row_offset, args
            )
            observed[:, idx] = vals

        observed_finite_count += np.isfinite(observed).sum(axis=0).astype(np.int64)
        observed_total_count += int(n_rows)

        out_df = pd.concat(
            [
                id_part.reset_index(drop=True),
                pd.DataFrame(observed, columns=sim_ids),
            ],
            axis=1,
        )
        out_df.to_csv(
            out_beta,
            mode="a",
            header=first,
            index=False,
            float_format="%.6f",
            na_rep="NA",
        )
        first = False
        row_offset += n_rows
        if args.progress_every > 0 and chunk_id % int(args.progress_every) == 0:
            print(
                f"[INFO] processed chunks={chunk_id}, rows={row_offset}",
                flush=True,
            )

    records["ObservedFeatureCount"] = observed_finite_count.astype(np.int64)
    records["ObservedMissingRate"] = (
        1.0 - observed_finite_count / np.maximum(observed_total_count, 1)
    ).astype(np.float32)

    meta_cols = [
        "Sample", "Types", "Platform", "Material", "Scenario",
        "TumorFraction", "TumorFractionBin", "TumorFractionComponent",
        "RecommendedTrainingWeight", "IsSynthetic", "SourceDomain",
        "LatentID", "PlatformDesign", "GeneratorSplit", "TechnicalBatch",
        "DominantBackgroundFamily", "DominantBackgroundFamilyWeight",
        "BackgroundSourceCount", "BackgroundEffectiveSourceCount",
        "BackgroundMaxSourceWeight", "ObservedFeatureCount", "ObservedMissingRate",
    ]
    meta_cols = [c for c in meta_cols if c in records.columns]
    records[meta_cols].to_csv(out_meta, sep="\t", index=False)
    records.to_csv(out_params, sep="\t", index=False)
    if out_source_map:
        source_map_df.to_csv(out_source_map, sep="\t", index=False)
    if out_source_usage_audit:
        source_usage_audit_df.to_csv(out_source_usage_audit, sep="\t", index=False)
    validation_report = {
        "filtering": filter_report,
        "governance": governance_report,
        "source_split_balance_all": source_split_balance_all,
        "source_pool_balance_selected": source_pool_balance_selected,
        "post_split_rows": int(len(meta)),
        "post_split_unique_donors": int(meta[args.donor_column].nunique()) if args.donor_column in meta.columns else None,
        "eligible_background_rows": int(len(background_meta)),
        "eligible_background_unique_donors": int(background_meta["_DonorID"].nunique()),
        "observed_missing_rate_summary": {
            "min": float(records["ObservedMissingRate"].min()),
            "median": float(records["ObservedMissingRate"].median()),
            "max": float(records["ObservedMissingRate"].max()),
        },
        "source_usage_audit_rows": int(len(source_usage_audit_df)),
    }
    if out_validation:
        out_validation.write_text(json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if out_manifest:
        family_counts = (
            background_meta.groupby(
                ["_ControlFamily", "_ControlSubtype", "_BackgroundRole"],
                dropna=False,
            )
            .size()
            .reset_index(name="n")
            .to_dict(orient="records")
        )
        manifest = {
            "script": Path(__file__).name,
            "generator_id": "mbmmc_cross_platform_in_silico_beta",
            "inputs": {
                "beta_csv": str(beta_csv),
                "meta": str(meta_path),
                "source_split_column": split_col,
                "source_split_value": split_value,
            },
            "outputs": {
                "out_beta_csv": str(out_beta),
                "out_meta": str(out_meta),
                "out_params": str(out_params),
                "out_source_map": str(out_source_map) if out_source_map else "",
                "out_source_usage_audit": str(out_source_usage_audit) if out_source_usage_audit else "",
            },
            "n_features_streamed": int(row_offset),
            "n_simulated_samples": int(len(records)),
            "n_latent_biological_profiles": int(records["LatentID"].nunique()),
            "tumor_types": tumor_types,
            "all_control_labels": background_labels,
            "eligible_background_source_count": int(len(background_samples)),
            "eligible_background_families": family_counts,
            "platform_presets": {k: asdict(v) for k, v in presets.items()},
            "metadata_validation": validation_report,
            "args": vars(args),
        }
        out_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("[DONE] in silico beta generation completed", flush=True)
    print(f"[OUT] beta_csv={out_beta}", flush=True)
    print(f"[OUT] meta={out_meta}", flush=True)
    print(f"[OUT] params={out_params}", flush=True)
    if out_source_map:
        print(f"[OUT] source_map={out_source_map}", flush=True)
    if out_manifest:
        print(f"[OUT] manifest={out_manifest}", flush=True)
    if out_source_usage_audit:
        print(f"[OUT] source_usage_audit={out_source_usage_audit}", flush=True)
    if out_validation:
        print(f"[OUT] validation_report={out_validation}", flush=True)


def add_platform_args(p: argparse.ArgumentParser, prefix: str) -> None:
    for name in ["sens_min", "sens_mean", "sens_max", "sens_kappa", "fp_min", "fp_mean", "fp_max", "fp_kappa",
                 "locus_sens_sd", "locus_fp_sd", "sample_logit_bias_sd", "gaussian_noise_kappa",
                 "depth_mean", "depth_cv", "depth_min", "depth_max",
                 "missing_min", "missing_mean", "missing_max", "missing_kappa"]:
        # Mean depth and depth CV are continuous calibration quantities; only hard
        # truncation bounds are integers.
        numeric_type = int if name in {"depth_min", "depth_max"} else float
        p.add_argument(f"--{prefix}_{name}", type=numeric_type, default=None)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate source-audited, hierarchically controlled, cross-platform "
            "in silico methylation beta matrices."
        )
    )
    p.add_argument(
        "--version",
        action="version",
        version="MBMMC cross-platform simulator 1.0.0",
    )
    p.add_argument("--beta_csv", required=True, help="Reference beta CSV: first 3 probe columns + sample columns")
    p.add_argument("--meta", required=True, help="Metadata with required Sample and Types columns")
    p.add_argument("--out_beta_csv", required=True)
    p.add_argument("--out_meta", required=True)
    p.add_argument("--out_params", required=True)
    p.add_argument("--out_manifest", default="")
    p.add_argument("--out_source_map", default="", help="Recommended: exact latent-to-source weights TSV")
    p.add_argument("--out_source_usage_audit", default="", help="Optional TSV: per-source latent usage audit")
    p.add_argument("--out_validation_report", default="", help="Optional JSON metadata-governance report")
    p.add_argument("--require_formal_meta_schema", action="store_true",
                   help="Require the fixed 13-column formal metadata contract and role/type consistency.")
    p.add_argument("--require_explicit_platform_parameters", action="store_true",
                   help="Fail if a formal run relies on any hidden per-platform default.")

    # SourceSplit auditing. These options do not assign splits; they make split-pool imbalance visible.
    p.add_argument("--source_split_audit", default="warn", choices=["off", "warn", "error"])
    p.add_argument("--expected_source_splits", default="sim_train,sim_test,sim2")
    p.add_argument("--split_balance_columns", default="SourceDataset,BackgroundRole,ControlFamily,Types")
    p.add_argument("--split_balance_top_n", type=int, default=200)
    p.add_argument("--min_background_sources", type=int, default=8)
    p.add_argument("--min_anchor_sources", type=int, default=1)
    p.add_argument("--min_tumor_sources_per_type", type=int, default=3)

    # Source governance and semantic control metadata.
    p.add_argument("--source_split_column", default="SourceSplit")
    p.add_argument("--source_split_value", default="", help="Example: train; generate each split separately")
    p.add_argument(
        "--generator_split",
        default="unspecified",
        choices=["sim_train", "sim2", "sim_test", "train", "calibration", "test", "stress", "unspecified"],
        help="Semantic label written to output metadata; sim_train, sim2, and sim_test are preferred formal-run labels.",
    )
    p.add_argument("--source_dataset_column", default="SourceDataset")
    p.add_argument("--donor_column", default="DonorID")
    p.add_argument("--replicate_group_column", default="ReplicateGroup",
                   help="Formal split-isolation unit for technical/biological replicate groups.")
    p.add_argument("--control_family_column", default="ControlFamily")
    p.add_argument("--control_subtype_column", default="ControlSubtype")
    p.add_argument("--background_role_column", default="BackgroundRole")
    p.add_argument("--biological_state_column", default="BiologicalState")
    p.add_argument("--include_column", default="IncludeForSimulation")
    p.add_argument("--include_values", default="YES,TRUE,1,Y")
    p.add_argument("--qc_status_column", default="QCStatus")
    p.add_argument("--qc_pass_values", default="PASS,OK,INCLUDE")
    p.add_argument(
        "--replicate_group_columns",
        default="DonorID,ControlSubtype,TissueOrCellType,BiologicalState",
        help="Columns used only to detect replicate-like metadata groups.",
    )
    p.add_argument("--replicate_policy", default="warn", choices=["warn", "error", "allow"])
    p.add_argument("--background_roles",
        default="plasma_anchor,background_component",
        help=(
            "Eligible roles for cfDNA mixing. Normal CNS tissue should usually be "
            "negative_only and remain real training controls, not plasma background."
        ),
    )
    p.add_argument(
        "--tumor_roles", default="tumor_source",
        help="Comma-separated semantic roles permitted to provide tumor latent profiles. Formal runs should use tumor_source only.",
    )

    p.add_argument("--background_labels", default="", help="Comma-separated explicit control Types; otherwise inferred")
    p.add_argument(
        "--background_regex",
        default=r"^(CONTR([_-].*)?|CONTROL([_-].*)?|CTRL([_-].*)?|Normal([_-].*)?|NORMAL([_-].*)?|Healthy([_-].*)?|HEALTHY([_-].*)?)$",
    )
    p.add_argument("--background_match_columns", default="both", choices=["types", "sample", "both"])
    p.add_argument("--control_output_label", default="CONTROL")
    p.add_argument("--keep_control_subtypes", dest="keep_control_subtypes", action="store_true", default=False, help="Deprecated for mixed controls; Types remains control_output_label and biological composition is recorded in source_map.")
    p.add_argument("--collapse_control_subtypes", dest="keep_control_subtypes", action="store_false")
    p.add_argument("--include_types", default="")
    p.add_argument("--exclude_types", default="")

    # Hierarchical background and tumor source models.
    p.add_argument("--background_sampling_mode", default="hierarchical", choices=["hierarchical", "flat"])
    p.add_argument("--background_source_sampling", default="donor_unique", choices=["donor_unique", "donor_balanced", "dataset_donor_balanced", "uniform"],
                   help="donor_unique samples at most one source per donor within a family; dataset_donor_balanced first balances studies then donors; donor_balanced is retained as a compatibility mode.")
    p.add_argument("--background_anchor_family", default="CFDNA_HEALTHY")
    p.add_argument("--require_background_anchor", action="store_true")
    p.add_argument("--background_anchor_weight_min", type=float, default=0.65)
    p.add_argument("--background_anchor_weight_max", type=float, default=0.90)
    p.add_argument("--background_anchor_only_probability", type=float, default=0.0,
                   help="Probability of retaining a pure empirical anchor profile; otherwise an anchor-containing multi-family background is constructed.")
    p.add_argument("--background_min_families_per_latent", type=int, default=1)
    p.add_argument("--background_max_families_per_latent", type=int, default=3)
    p.add_argument("--background_family_concentration", type=float, default=12.0)
    p.add_argument(
        "--background_family_weight_map",
        default="CFDNA_HEALTHY:6,BLOOD_IMMUNE:3,ERYTHROID:1.5,ENDOTHELIAL:0.6,SOLID_TISSUE_LIVER:0.15",
        help="Family priors; unavailable families are ignored.",
    )
    p.add_argument("--tumor_mixture_mode", default="anchor", choices=["anchor", "dirichlet"])
    p.add_argument("--tumor_source_sampling", default="balanced_anchor",
                   choices=["uniform", "balanced_anchor"],
                   help="balanced_anchor equalizes anchor-source reuse across latents; recommended for formal runs.")
    p.add_argument("--tumor_anchor_weight_min", type=float, default=0.70)
    p.add_argument("--tumor_anchor_weight_max", type=float, default=1.00)
    p.add_argument("--tumor_dirichlet_alpha", type=float, default=0.35)
    p.add_argument("--background_dirichlet_alpha", type=float, default=0.80)
    p.add_argument(
        "--tumor_max_sources_per_latent",
        type=int,
        default=4,
        help="Recommended 1-4 for anchor-patient mixtures; 0 means all sources.",
    )
    p.add_argument(
        "--background_max_sources_per_latent",
        type=int,
        default=8,
        help="Total target source count across selected control families; 6-8 is recommended to avoid over-smoothing.",
    )

    # Simulation layout.
    p.add_argument("--platforms", default="TAPS,WGBS,ONT")
    p.add_argument("--platform_design", default="paired_latent", choices=["paired_latent", "independent"])
    p.add_argument("--n_per_type_per_platform", type=int, default=160)
    p.add_argument("--n_boundary_per_type_per_platform", type=int, default=120)
    p.add_argument("--n_control_per_platform", type=int, default=600)
    p.add_argument("--material", default="cfDNA")
    p.add_argument("--tf_mode", default="clinical_mixture", choices=["clinical_mixture", "stratified_bins", "uniform", "beta_low", "boundary", "fixed"])
    p.add_argument(
        "--tf_component_weight_map", default="",
        help="Optional weights for clinical_mixture components, e.g. ultra_low:0.18,low:0.32,boundary:0.25,mid:0.18,high:0.07. Ignored by stratified_bins.",
    )
    p.add_argument("--min_tf", type=float, default=0.001)
    p.add_argument("--max_tf", type=float, default=0.85)
    p.add_argument("--fixed_tf", type=float, default=0.10)
    p.add_argument("--decision_tf", type=float, default=0.10)

    # Measurement and missingness.
    p.add_argument("--measurement_model", default="beta_binomial", choices=["none", "gaussian", "binomial", "beta_binomial"])
    p.add_argument("--beta_binomial_kappa", type=float, default=80.0, help="Global compatibility fallback; formal runs should set platform-specific values below.")
    p.add_argument("--locus_depth_sigma", type=float, default=0.75, help="Global compatibility fallback; formal runs should set platform-specific values below.")
    p.add_argument("--taps_beta_binomial_kappa", type=float, default=None)
    p.add_argument("--wgbs_beta_binomial_kappa", type=float, default=None)
    p.add_argument("--ont_beta_binomial_kappa", type=float, default=None)
    p.add_argument("--taps_locus_depth_sigma", type=float, default=None)
    p.add_argument("--wgbs_locus_depth_sigma", type=float, default=None)
    p.add_argument("--ont_locus_depth_sigma", type=float, default=None)
    p.add_argument("--depth_lambda_clip", type=float, default=1000.0)
    p.add_argument("--minimum_observed_depth", type=int, default=1)
    p.add_argument("--zero_depth_policy", default="missing", choices=["missing", "floor", "resample"])
    p.add_argument("--zero_depth_resample_rounds", type=int, default=4)
    p.add_argument("--gaussian_variance_mode", default="sample_depth", choices=["sample_depth", "fixed_kappa"])
    p.add_argument(
        "--missing_mode",
        default="measurement_only",
        choices=["none", "measurement_only", "mcar", "mcar_beta", "locus_weighted", "feature_subset", "hybrid"],
        help="measurement_only keeps only finite-depth missingness; hybrid adds platform missingness.",
    )
    p.add_argument("--estimate_locus_missing_from_reference", action="store_true")
    p.add_argument("--max_locus_missing_weight", type=float, default=8.0)

    # Platform locus and batch effects.
    p.add_argument("--deterministic_locus_effects", dest="deterministic_locus_effects", action="store_true", default=True)
    p.add_argument("--stochastic_locus_effects", dest="deterministic_locus_effects", action="store_false")
    p.add_argument("--locus_effect_seed", type=int, default=None, help="Set independently across train/test stress generators")
    p.add_argument("--technical_batches_per_platform", type=int, default=6)
    p.add_argument("--batch_logit_bias_sd", type=float, default=0.025)
    p.add_argument("--use_reference_variability_noise", action="store_true")
    p.add_argument("--reference_variability_noise_scale", type=float, default=0.25)
    p.add_argument("--reference_variability_z_clip", type=float, default=4.0)
    p.add_argument("--reference_variability_max_multiplier", type=float, default=2.0)

    add_platform_args(p, "taps")
    add_platform_args(p, "wgbs")
    add_platform_args(p, "ont")

    # Backward-compatible aliases.
    p.add_argument("--taps_conv_min", type=float, default=None)
    p.add_argument("--taps_conv_mean", type=float, default=None)
    p.add_argument("--taps_conv_max", type=float, default=None)
    p.add_argument("--taps_conv_kappa", type=float, default=None)
    p.add_argument("--taps_unmodified_false_positive", type=float, default=None)
    p.add_argument("--wgbs_methylated_retention_min", type=float, default=None)
    p.add_argument("--wgbs_methylated_retention_mean", type=float, default=None)
    p.add_argument("--wgbs_methylated_retention_max", type=float, default=None)
    p.add_argument("--wgbs_methylated_retention_kappa", type=float, default=None)
    p.add_argument("--wgbs_incomplete_conversion_min", type=float, default=None)
    p.add_argument("--wgbs_incomplete_conversion_mean", type=float, default=None)
    p.add_argument("--wgbs_incomplete_conversion_max", type=float, default=None)
    p.add_argument("--wgbs_incomplete_conversion_kappa", type=float, default=None)

    # Training weights.
    p.add_argument("--sim_training_weight", type=float, default=0.35)
    p.add_argument("--control_training_weight", type=float, default=0.35)
    p.add_argument("--artifact_training_weight", type=float, default=0.20)
    p.add_argument("--low_tf_training_weight", type=float, default=0.10)
    p.add_argument("--boundary_tf_training_weight", type=float, default=0.45)
    p.add_argument("--high_tf_training_weight", type=float, default=0.35)
    p.add_argument("--platform_training_weight_map", default="")

    p.add_argument("--chunk_rows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_every", type=int, default=20)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    generate(args)


if __name__ == "__main__":
    main()
