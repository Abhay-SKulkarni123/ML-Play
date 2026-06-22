import io
import json
import uuid as uuid_lib
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.ml.dataset import load_dataset

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"]
)

DATA_DIR = Path(__file__).parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

META_PATH = DATA_DIR / "meta.json"


@router.get("/")
async def list_datasets():
    if not META_PATH.exists():
        return []

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    return [
        {"id": dataset_id, **dataset_info}
        for dataset_id, dataset_info in meta.items()
    ]


@router.get("/{dataset_id}/profile")
async def get_dataset_profile(dataset_id: str):
    if not META_PATH.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    if dataset_id not in meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = DATA_DIR / f"{dataset_id}.parquet"

    if not dataset_path.exists():
        upload_path = UPLOAD_DIR / f"{dataset_id}.parquet"

        if upload_path.exists():
            dataset_path = upload_path
        else:
            raise HTTPException(
                status_code=404,
                detail="Dataset file not found"
            )

    df = pd.read_parquet(dataset_path)

    profile = {
        "shape": {
            "rows": len(df),
            "cols": len(df.columns)
        },
        "target": meta[dataset_id]["target"],
        "task": meta[dataset_id]["task"],
        "columns": [],
        "missing_summary": {
            "total_missing_cells": int(df.isna().sum().sum()),
            "columns_with_missing": [
                col for col in df.columns
                if df[col].isna().sum() > 0
            ],
            "pct_complete": round(
                100
                * (
                    1
                    - df.isna().sum().sum()
                    / (len(df) * len(df.columns))
                ),
                2,
            ),
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        profile["columns"].append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "type": (
                    "numeric"
                    if pd.api.types.is_numeric_dtype(df[col])
                    else "categorical"
                ),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(
                    100 * df[col].isna().mean(),
                    2,
                ),
                "unique_count": int(df[col].nunique()),
                "is_target": col == meta[dataset_id]["target"],
                "stats": {},
            }
        )

    return profile


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    dataset_name: str = Form("My Dataset"),
):
    if not file.filename.endswith(
        (".csv", ".xlsx", ".xls")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files supported.",
        )

    content = await file.read()

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Max 50MB.",
        )

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse file: {e}",
        )

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found.",
        )

    task = (
        "classification"
        if df[target_col].dtype == object
        or df[target_col].nunique() <= 20
        else "regression"
    )

    dataset_id = f"upload_{uuid_lib.uuid4().hex[:8]}"

    save_path = UPLOAD_DIR / f"{dataset_id}.parquet"
    df.to_parquet(save_path, index=False)

    meta = {}

    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            meta = json.load(f)

    meta[dataset_id] = {
        "name": dataset_name,
        "target": target_col,
        "task": task,
        "rows": len(df),
        "cols": len(df.columns),
        "uploaded": True,
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "target": target_col,
        "task": task,
        "rows": len(df),
        "cols": len(df.columns),
    }

@router.get("/{dataset_id}/sample")
async def get_sample(dataset_id: str, n: int = 200):
    """Return n real rows for scatter plot visualization."""
    try:
        df = load_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    sample = df.select_dtypes(include="number").sample(
        min(n, len(df)), random_state=42
    ).fillna(0)
    
    return {
        "rows": sample.to_dict(orient="records"),
        "columns": sample.columns.tolist(),
        "n": len(sample),
    }