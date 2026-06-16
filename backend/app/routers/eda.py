from fastapi import APIRouter, HTTPException
from app.services.ml.dataset import load_dataset, profile_dataset
import pandas as pd
import numpy as np
import json
from pathlib import Path

router = APIRouter(prefix="/eda", tags=["eda"])

META_PATH = Path(__file__).parent.parent / "data" / "meta.json"

def _load_meta():
    with open(META_PATH) as f:
        return json.load(f)


@router.get("/{dataset_id}/distributions")
async def get_distributions(dataset_id: str):
    try:
        df = load_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = {}
    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        counts, edges = np.histogram(clean, bins=20)
        result[col] = {
            "type": "numeric",
            "histogram": {
                "counts": counts.tolist(),
                "edges": [round(float(e), 4) for e in edges.tolist()],
            },
            "stats": {
                "mean":   round(float(clean.mean()), 4),
                "median": round(float(clean.median()), 4),
                "std":    round(float(clean.std()), 4),
                "min":    round(float(clean.min()), 4),
                "max":    round(float(clean.max()), 4),
                "skew":   round(float(clean.skew()), 4),
            }
        }

    for col in df.select_dtypes(exclude="number").columns:
        vc = df[col].value_counts().head(12)
        result[col] = {
            "type": "categorical",
            "bar": {
                "labels": vc.index.tolist(),
                "counts": vc.values.tolist(),
            }
        }

    return result


@router.get("/{dataset_id}/correlation")
async def get_correlation(dataset_id: str):
    try:
        df = load_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"columns": [], "matrix": []}

    corr = numeric.corr().round(3)
    return {
        "columns": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    }


@router.get("/{dataset_id}/target-analysis")
async def get_target_analysis(dataset_id: str):
    try:
        df = load_dataset(dataset_id)
        meta = _load_meta()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    target = meta[dataset_id]["target"]
    task = meta[dataset_id]["task"]

    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found")

    result = {"target": target, "task": task}

    if task == "classification":
        vc = df[target].value_counts()
        result["distribution"] = {
            "labels": [str(x) for x in vc.index.tolist()],
            "counts": vc.values.tolist(),
        }
        result["class_balance"] = round(float(vc.min() / vc.max()), 3)
        result["is_imbalanced"] = result["class_balance"] < 0.5
    else:
        clean = df[target].dropna()
        counts, edges = np.histogram(clean, bins=20)
        result["histogram"] = {
            "counts": counts.tolist(),
            "edges": [round(float(e), 4) for e in edges.tolist()],
        }
        result["stats"] = {
            "mean":   round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std":    round(float(clean.std()), 4),
            "min":    round(float(clean.min()), 4),
            "max":    round(float(clean.max()), 4),
        }

    return result