from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd

@dataclass
class ReferenceData:
    X_beta: np.ndarray            # shape: (n_samples, n_features), float32 with NaN
    y: np.ndarray                 # shape: (n_samples,)
    samples: List[str]            # length n_samples
    features: List[str]           # length n_features (chr_pos)
    chr_arr: Optional[np.ndarray] # length n_features
    pos_arr: Optional[np.ndarray] # length n_features
    platform: Optional[np.ndarray] = None  # length n_samples
    material: Optional[np.ndarray] = None  # length n_samples


def _read_table_auto(path: str) -> pd.DataFrame:
    # try tab, comma
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass
    return pd.read_csv(path)


def load_reference(ref_csv: str, meta_path: str, exclude_types: Optional[List[str]] = None) -> ReferenceData:
    """Load the reference methylation CSV and meta file.

    Reference CSV (wide):
      probe_id, chr, pos, sample1, sample2, ... (β values; may contain NaN)

    meta.txt (tab or csv):
      Required: Sample, Types
      Optional: Platform, Material

    Returns:
      ReferenceData with X_beta shaped (n_samples, n_features) aligned to meta Sample order.
    """
    if not os.path.exists(ref_csv):
        raise FileNotFoundError(ref_csv)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    df = pd.read_csv(ref_csv,na_values=['NA', '', 'NaN', 'N/A', 'null']) ##增加了缺失值类型
    if df.shape[1] < 4:
        raise ValueError("Reference CSV must have at least 4 columns: probe_id, chr, pos, and >=1 sample")

    # first 3 columns are probe_id, chr, pos (names not strictly required, order required)
    df = df.dropna(subset=[df.columns[1], df.columns[2]])  # 删除chr和pos有缺失的行
    chr_col = df.iloc[:, 1].astype(str).to_numpy(copy=False)
    pos_col = df.iloc[:, 2].to_numpy(copy=False)
    features = [f"{c}_{int(p)}" for c, p in zip(chr_col.tolist(), pos_col.tolist())]

    sample_cols = list(df.columns[3:])
    sample_cols = [str(s).strip() for s in sample_cols] #新增
    beta_vals = df.iloc[:, 3:].to_numpy(dtype=np.float32, copy=False)  # shape (n_features, n_samples)
    X_beta = beta_vals.T  # shape (n_samples, n_features)

    # meta
    meta = pd.read_csv(meta_path, sep=None, engine="python")
    if "Sample" not in meta.columns or "Types" not in meta.columns:
        raise ValueError("meta.txt must contain columns: Sample and Types")

    meta["Sample"] = meta["Sample"].astype(str).str.strip()
    meta["Types"] = meta["Types"].astype(str).str.strip()
    if meta["Sample"].duplicated().any():   #新增
        dup = meta.loc[meta["Sample"].duplicated(), "Sample"].tolist()
        raise ValueError(f"Duplicate Sample in meta: {dup[:10]} (total={len(dup)})")
    # optional batch columns
    if "Platform" in meta.columns:
        meta["Platform"] = meta["Platform"].astype(str).str.strip().str.upper()
        # Normalize common spellings (e.g., 450k -> 450K)
        meta["Platform"] = meta["Platform"].replace({"450K": "450K"})
    if "Material" in meta.columns:
        meta["Material"] = meta["Material"].astype(str).str.strip()
        meta["Material"] = meta["Material"].str.replace("Fronzen", "Frozen", regex=False)
        _m = meta["Material"].str.lower()
        meta["Material"] = _m.map({"frozen": "Frozen", "ffpe": "FFPE", "unknown": "Unknown"}).fillna(meta["Material"])
    meta_samples_all_raw = meta["Sample"].tolist()
    missing_in_meta_raw = [s for s in sample_cols if s not in set(meta_samples_all_raw)]
    missing_in_csv_raw = [s for s in set(meta_samples_all_raw) if s not in sample_cols]
    if missing_in_meta_raw or missing_in_csv_raw:
        raise ValueError(f"reference CSV and meta with diff sample when without exclude labels!"
                f"missing_in_meta={missing_in_meta_raw[:10]} (total={len(missing_in_meta_raw)}), "
                f"missing_in_csv={missing_in_csv_raw[:10]} (total={len(missing_in_csv_raw)})"
                )
    # exclude labels if requested
    if exclude_types:
        exclude_set = set([str(x) for x in exclude_types])
        meta = meta[~meta["Types"].isin(exclude_set)].copy()

    # Validate sample names
    meta_samples_all = meta["Sample"].tolist()
#    missing_in_meta = [s for s in sample_cols if s not in set(meta_samples_all)]
#    if missing_in_meta:
#        raise ValueError(
#            "Some sample columns in reference CSV are missing in meta.txt. "
#            f"Examples: {missing_in_meta[:10]} (total={len(missing_in_meta)})"
#        )

    # Keep only samples that are present in the CSV, and in meta order
    meta_samples = [s for s in meta_samples_all if s in sample_cols]
    if len(meta_samples) == 0:
        raise ValueError("No overlapping samples between meta.txt and reference CSV")

    # Reorder X_beta columns to match meta_samples order
    col_index = {name: i for i, name in enumerate(sample_cols)}
    idx = np.array([col_index[s] for s in meta_samples], dtype=int)
    X_beta = X_beta[idx, :]

    y = meta.set_index("Sample").loc[meta_samples]["Types"].to_numpy()

    platform = None
    material = None
    if "Platform" in meta.columns:
        platform = meta.set_index("Sample").loc[meta_samples]["Platform"].to_numpy()
    if "Material" in meta.columns:
        material = meta.set_index("Sample").loc[meta_samples]["Material"].to_numpy()

    return ReferenceData(
        X_beta=X_beta,
        y=y,
        samples=meta_samples,
        features=features,
        chr_arr=chr_col,
        pos_arr=pos_col,
        platform=platform,
        material=material,
    )


def _sniff_sep(path: str) -> str:
    """Heuristically detect delimiter for wide single-sample file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    # count delimiters in the header line
    tab_n = line.count("	")
    comma_n = line.count(",")
    if tab_n > comma_n:
        return "	"
    return ","


def load_wide_sample(sample_wide_csv: str) -> Tuple[str, Dict[str, float]]:
    """Load a single-sample wide file (tab- or comma-delimited).
    Expected: first column is Sample, remaining columns are chr_pos features.
    Returns (sample_name, dict(feature->value)).
    Missing values can be NA/NaN/empty and will be ignored (treated as missing).
    """
    sep = _sniff_sep(sample_wide_csv)
    df = pd.read_csv(sample_wide_csv, sep=sep, na_values=["NA","Na","na","N/A","", "nan", "NaN"], keep_default_na=True)
    if df.shape[0] != 1:
        raise ValueError(f"Wide sample file must contain exactly 1 row, got {df.shape[0]}")
    if df.shape[1] < 2:
        raise ValueError("Wide sample file must have >=2 columns: Sample + features")

    sample_name = str(df.iloc[0, 0])
    feature_cols = list(df.columns[1:])
    vals = df.iloc[0, 1:].to_numpy()

    feat_map: Dict[str, float] = {}
    for c, v in zip(feature_cols, vals):
        try:
            if pd.isna(v):
                continue
            feat_map[str(c)] = float(v)
        except Exception:
            continue
    return sample_name, feat_map


def align_wide_to_features(feat_map: Dict[str, float], feature_list: List[str], default_value: float = 0.0) -> np.ndarray:
    """Align a sparse dict of feature values to a full feature list."""
    x = np.full((len(feature_list),), default_value, dtype=np.float32)
    for i, f in enumerate(feature_list):
        if f in feat_map:
            x[i] = float(feat_map[f])
    return x
