import io
from pathlib import Path
import uuid as uuid_lib
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload") # type: ignore
async def upload_dataset(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    dataset_name: str = Form("My Dataset"),
):
    import pandas as pd
    import json
    from pathlib import Path

    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files supported.")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found. Available: {df.columns.tolist()}"
        )

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Dataset too small. Need at least 20 rows.")

    # Detect task type
    y = df[target_col]
    task = "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"

    # Save with unique ID
    dataset_id = f"upload_{str(uuid_lib.uuid4())[:8]}"
    save_path = UPLOAD_DIR / f"{dataset_id}.parquet"
    df.to_parquet(save_path, index=False)

    # Register in meta.json
    meta_path = Path(__file__).parent.parent / "data" / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    meta[dataset_id] = {
        "name":   dataset_name,
        "target": target_col,
        "task":   task,
        "rows":   len(df),
        "cols":   len(df.columns),
        "uploaded": True,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "dataset_id": dataset_id,
        "name":       dataset_name,
        "target":     target_col,
        "task":       task,
        "rows":       len(df),
        "cols":       len(df.columns),
    }