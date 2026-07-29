from __future__ import annotations
import json, random, os, sys, platform
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

def infer_sep(path: str | Path) -> str:
    p = str(path)
    if p.endswith(".tsv") or p.endswith(".txt"):
        return "\t"
    return ","

def read_table(path: str | Path, sample_id_col: str | None = None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=infer_sep(path))
    if sample_id_col and sample_id_col in df.columns:
        df = df.set_index(sample_id_col)
    elif sample_id_col and sample_id_col not in df.columns:
        # allow already-indexed CSV exported with first unnamed column
        if df.columns[0].lower().startswith("unnamed"):
            df = df.set_index(df.columns[0])
        else:
            raise ValueError(f"sample_id_col={sample_id_col!r} not found in {path}")
    return df

def write_json(obj: Dict[str, Any], path: str | Path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

def software_versions() -> Dict[str, str]:
    out = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn
        out["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import torch
        out["torch"] = str(torch.__version__)
        out["cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        pass
    return out

def load_matrix_and_labels(matrix_path, label_path, sample_id_col="sample_id", label_col="label"):
    X = read_table(matrix_path, sample_id_col=sample_id_col)
    labels = read_table(label_path, sample_id_col=sample_id_col)
    if label_col not in labels.columns:
        raise ValueError(f"Label column {label_col!r} not found in {label_path}")
    common = X.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between matrix and label files.")
    X = X.loc[common]
    y = labels.loc[common, label_col].astype(str)
    # keep numeric methylation features only
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y, labels.loc[common]
