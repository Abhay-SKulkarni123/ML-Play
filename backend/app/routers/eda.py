import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.services.ml.dataset import load_dataset

router = APIRouter(prefix="/eda", tags=["eda"])

META_PATH = Path(__file__).parent.parent / "data" / "meta.json"


def _load_meta() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


def _validate_dataset(dataset_id: str) -> pd.DataFrame:
    try:
        return load_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found.",
        )


@router.get("/{dataset_id}/distributions")
async def get_distributions(dataset_id: str):
    df     = _validate_dataset(dataset_id)
    result = {}

    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue

        try:
            n_bins = min(20, max(5, len(clean.unique())))
            counts, edges = np.histogram(clean, bins=n_bins)
            histogram = {
                "counts": counts.tolist(),
                "edges":  [round(float(e), 4) for e in edges.tolist()],
            }
        except Exception:
            histogram = None

        result[col] = {
            "type":      "numeric",
            "histogram": histogram,
            "stats": {
                "mean":   round(float(clean.mean()),   4),
                "median": round(float(clean.median()), 4),
                "std":    round(float(clean.std()),    4),
                "min":    round(float(clean.min()),    4),
                "max":    round(float(clean.max()),    4),
                "skew":   round(float(clean.skew()),   4),
            },
        }

    for col in df.select_dtypes(exclude="number").columns:
        vc = df[col].value_counts().head(12)
        result[col] = {
            "type": "categorical",
            "bar": {
                "labels": [str(x) for x in vc.index.tolist()],
                "counts": vc.values.tolist(),
            },
        }

    return result


@router.get("/{dataset_id}/correlation")
async def get_correlation(dataset_id: str):
    df      = _validate_dataset(dataset_id)
    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        return {"columns": [], "matrix": []}

    # Drop columns that are all-null
    numeric = numeric.dropna(axis=1, how="all")

    try:
        corr = numeric.corr().round(3)
        # Replace NaN correlations with 0 (happens with zero-variance columns)
        corr = corr.fillna(0)
        return {
            "columns": corr.columns.tolist(),
            "matrix":  corr.values.tolist(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Correlation computation failed: {str(e)}",
        )


@router.get("/{dataset_id}/target-analysis")
async def get_target_analysis(dataset_id: str):
    df   = _validate_dataset(dataset_id)
    meta = _load_meta()

    if dataset_id not in meta:
        raise HTTPException(
            status_code=404,
            detail=f"No metadata found for dataset '{dataset_id}'.",
        )

    target = meta[dataset_id]["target"]
    task   = meta[dataset_id]["task"]

    if target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target}' not found in dataset. "
                f"Available: {df.columns.tolist()}"
            ),
        )

    result: dict = {"target": target, "task": task}
    y = df[target].dropna()

    if task == "classification":
        vc = y.value_counts()
        result["distribution"] = {
            "labels": [str(x) for x in vc.index.tolist()],
            "counts": vc.values.tolist(),
        }
        if len(vc) >= 2:
            balance_ratio = round(float(vc.min() / vc.max()), 3)
            result["class_balance"]  = balance_ratio
            result["is_imbalanced"]  = balance_ratio < 0.5
            result["n_classes"]      = int(len(vc))
            result["majority_class"] = str(vc.index[0])
            result["minority_class"] = str(vc.index[-1])
        else:
            result["class_balance"] = 1.0
            result["is_imbalanced"] = False
            result["n_classes"]     = int(len(vc))

    else:
        # Regression target
        try:
            n_bins = min(20, max(5, len(y.unique())))
            counts, edges = np.histogram(y, bins=n_bins)
            result["histogram"] = {
                "counts": counts.tolist(),
                "edges":  [round(float(e), 4) for e in edges.tolist()],
            }
        except Exception:
            result["histogram"] = None

        result["stats"] = {
            "mean":   round(float(y.mean()),   4),
            "median": round(float(y.median()), 4),
            "std":    round(float(y.std()),    4),
            "min":    round(float(y.min()),    4),
            "max":    round(float(y.max()),    4),
            "skew":   round(float(y.skew()),   4),
        }

    return result


@router.get("/{dataset_id}/pairwise")
async def get_pairwise(dataset_id: str, col_a: str, col_b: str):
    """
    Returns scatter plot data between two numeric columns.
    Used by the bottom chart bar scatter view.
    """
    df = _validate_dataset(dataset_id)

    for col in [col_a, col_b]:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found.",
            )
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' is not numeric.",
            )

    # Sample max 500 points for performance
    sample = df[[col_a, col_b]].dropna()
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)

    return {
        "col_a":  col_a,
        "col_b":  col_b,
        "points": [
            {"x": round(float(row[col_a]), 4), "y": round(float(row[col_b]), 4)}
            for _, row in sample.iterrows()
        ],
    }