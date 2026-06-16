import pandas as pd
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_all_datasets() -> list[dict]:
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    return [{"id": k, **v} for k, v in meta.items()]


def load_dataset(dataset_id: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    return pd.read_parquet(path)


def profile_dataset(dataset_id: str) -> dict:
    df = load_dataset(dataset_id)
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)

    target_col = meta[dataset_id]["target"]
    profile = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "target": target_col,
        "task": meta[dataset_id]["task"],
        "columns": [],
        "missing_summary": {},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
            "is_target": col == target_col,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["type"] = "numeric"
            col_info["stats"] = {
                "mean":   round(float(df[col].mean()), 4) if not df[col].isnull().all() else None,
                "median": round(float(df[col].median()), 4) if not df[col].isnull().all() else None,
                "std":    round(float(df[col].std()), 4) if not df[col].isnull().all() else None,
                "min":    round(float(df[col].min()), 4) if not df[col].isnull().all() else None,
                "max":    round(float(df[col].max()), 4) if not df[col].isnull().all() else None,
                "skewness": round(float(df[col].skew()), 4) if not df[col].isnull().all() else None,
            }
            # histogram bins
            clean = df[col].dropna()
            if len(clean) > 0:
                counts, edges = np.histogram(clean, bins=20)
                col_info["histogram"] = {
                    "counts": counts.tolist(),
                    "edges": [round(e, 4) for e in edges.tolist()],
                }
        else:
            col_info["type"] = "categorical"
            col_info["stats"] = {
                "top_values": df[col].value_counts().head(10).to_dict(),
            }

        profile["columns"].append(col_info)

    # overall missing summary
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    profile["missing_summary"] = {
        "total_missing_cells": int(df.isnull().sum().sum()),
        "columns_with_missing": missing_cols,
        "pct_complete": round((1 - df.isnull().mean().mean()) * 100, 2),
    }

    return profile