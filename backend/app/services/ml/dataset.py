import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_meta() -> dict:
    meta_path = DATA_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found. Run download_datasets.py first.")
    with open(meta_path) as f:
        return json.load(f)


def get_all_datasets() -> list[dict]:
    meta = _load_meta()
    result = []
    for k, v in meta.items():
        path = DATA_DIR / f"{k}.parquet"
        if path.exists():
            result.append({"id": k, **v})
    return result


def load_dataset(dataset_id: str) -> pd.DataFrame:
    # Check main data dir first, then uploads
    path = DATA_DIR / f"{dataset_id}.parquet"
    if not path.exists():
        path = DATA_DIR / "uploads" / f"{dataset_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found.")
    return pd.read_parquet(path)


def profile_dataset(dataset_id: str) -> dict:
    df   = load_dataset(dataset_id)
    meta = _load_meta()

    if dataset_id not in meta:
        raise ValueError(f"No metadata for dataset '{dataset_id}'")

    target_col = meta[dataset_id]["target"]
    task       = meta[dataset_id]["task"]

    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )

    profile = {
        "shape":   {"rows": len(df), "cols": len(df.columns)},
        "target":  target_col,
        "task":    task,
        "columns": [],
        "missing_summary": {},
        "duplicate_rows":  int(df.duplicated().sum()),
    }

    for col in df.columns:
        col_info = {
            "name":          col,
            "dtype":         str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_pct":   round(float(df[col].isnull().mean() * 100), 2),
            "unique_count":  int(df[col].nunique()),
            "is_target":     col == target_col,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["type"] = "numeric"
            clean = df[col].dropna()

            if len(clean) > 0:
                col_info["stats"] = {
                    "mean":     round(float(clean.mean()),   4),
                    "median":   round(float(clean.median()), 4),
                    "std":      round(float(clean.std()),    4),
                    "min":      round(float(clean.min()),    4),
                    "max":      round(float(clean.max()),    4),
                    "skewness": round(float(clean.skew()),   4),
                }
                # Histogram — handle edge case of all-same values
                try:
                    counts, edges = np.histogram(clean, bins=min(20, len(clean.unique())))
                    col_info["histogram"] = {
                        "counts": counts.tolist(),
                        "edges":  [round(float(e), 4) for e in edges.tolist()],
                    }
                except Exception:
                    col_info["histogram"] = None
            else:
                col_info["stats"]     = {}
                col_info["histogram"] = None

        else:
            col_info["type"] = "categorical"
            vc = df[col].value_counts()
            col_info["stats"] = {
                "top_values": {
                    str(k): int(v)
                    for k, v in vc.head(10).items()
                },
            }

        profile["columns"].append(col_info)

    # Missing summary
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    profile["missing_summary"] = {
        "total_missing_cells":  int(df.isnull().sum().sum()),
        "columns_with_missing": missing_cols,
        "pct_complete":         round(float((1 - df.isnull().mean().mean()) * 100), 2),
    }

    return profile