# from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse
# import pandas as pd
# import numpy as np
# import json
# import io
# import uuid
# from pathlib import Path
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import f1_score, r2_score, accuracy_score
# import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# router = APIRouter(prefix="/automl", tags=["automl"])

# UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
# UPLOAD_DIR.mkdir(exist_ok=True)

# automl_results: dict = {}


# def _detect_task(y: pd.Series) -> str:
#     if y.dtype == object or y.nunique() <= 20:
#         return "classification"
#     return "regression"


# def _run_automl(job_id: str, df: pd.DataFrame, target_col: str):
#     try:
#         automl_results[job_id] = {"status": "running", "progress": 0, "log": []}

#         def log(msg: str, progress: int):
#             automl_results[job_id]["log"].append(msg)
#             automl_results[job_id]["progress"] = progress

#         log("Detecting task type...", 5)
#         y = df[target_col]
#         X_raw = df.drop(columns=[target_col])
#         task = _detect_task(y)
#         automl_results[job_id]["task"] = task

#         log(f"Task detected: {task}", 10)

#         # Drop high-missing columns
#         log("Handling missing values...", 15)
#         high_missing = [c for c in X_raw.columns if X_raw[c].isnull().mean() > 0.5]
#         X_raw.drop(columns=high_missing, inplace=True)

#         # Encode categoricals
#         log("Encoding categorical columns...", 25)
#         cat_cols = X_raw.select_dtypes(exclude="number").columns.tolist()
#         for col in cat_cols:
#             le = LabelEncoder()
#             X_raw[col] = le.fit_transform(X_raw[col].astype(str))

#         if task == "classification":
#             le_y = LabelEncoder()
#             y = pd.Series(le_y.fit_transform(y.astype(str)))

#         # Impute
#         log("Imputing remaining missing values...", 35)
#         imputer = SimpleImputer(strategy="median")
#         X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

#         # Split
#         log("Splitting data...", 45)
#         stratify = y if task == "classification" else None
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=stratify
#         )

#         cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if task == "classification" else KFold(n_splits=3, shuffle=True, random_state=42)
#         scoring = "f1_weighted" if task == "classification" else "r2"

#         log("Running Optuna hyperparameter search...", 55)

#         def objective(trial):
#             params = {
#                 "n_estimators": trial.suggest_int("n_estimators", 50, 200),
#                 "max_depth": trial.suggest_int("max_depth", 2, 15),
#                 "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
#             }
#             model = RandomForestClassifier(**params, random_state=42) if task == "classification" else RandomForestRegressor(**params, random_state=42)
#             scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
#             return scores.mean()

#         study = optuna.create_study(direction="maximize")
#         study.optimize(objective, n_trials=20, timeout=45)

#         log("Training best model...", 80)
#         best_params = study.best_params
#         best_model = RandomForestClassifier(**best_params, random_state=42) if task == "classification" else RandomForestRegressor(**best_params, random_state=42)
#         best_model.fit(X_train, y_train)
#         y_pred = best_model.predict(X_test)

#         metrics = {}
#         if task == "classification":
#             metrics["accuracy"]    = round(float(accuracy_score(y_test, y_pred)), 4)
#             metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
#         else:
#             metrics["r2"]  = round(float(r2_score(y_test, y_pred)), 4)
#             metrics["rmse"] = round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 4)

#         cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring)
#         metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
#         metrics["cv_std"]  = round(float(cv_scores.std()), 4)

#         fi = {}
#         if hasattr(best_model, "feature_importances_"):
#             fi = dict(zip(X.columns, best_model.feature_importances_))
#             fi = {k: round(float(v), 6) for k, v in sorted(fi.items(), key=lambda x: -x[1])}

#         log("Done!", 100)

#         pipeline_summary = {
#             "missing_cols_dropped": high_missing,
#             "categorical_cols_encoded": cat_cols,
#             "imputation": "median",
#             "model": "RandomForest",
#             "best_params": best_params,
#             "n_trials": len(study.trials),
#             "best_trial_score": round(study.best_value, 4),
#         }

#         automl_results[job_id] = {
#             "status": "done",
#             "progress": 100,
#             "task": task,
#             "metrics": metrics,
#             "feature_importance": fi,
#             "pipeline_summary": pipeline_summary,
#             "shape": {"rows": len(df), "cols": len(df.columns)},
#             "log": automl_results[job_id]["log"],
#             "n_features": len(X.columns),
#             "train_size": len(X_train),
#             "test_size": len(X_test),
#         }

#     except Exception as e:
#         automl_results[job_id] = {
#             "status": "error",
#             "error": str(e),
#             "progress": 0,
#             "log": automl_results.get(job_id, {}).get("log", []),
#         }


# @router.post("/run")
# async def run_automl(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     target_col: str = "target",
# ):
#     if not file.filename.endswith((".csv", ".xlsx", ".xls")):
#         raise HTTPException(status_code=400, detail="Only CSV and Excel files supported")

#     content = await file.read()
#     if len(content) > 50 * 1024 * 1024:
#         raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

#     try:
#         if file.filename.endswith(".csv"):
#             df = pd.read_csv(io.BytesIO(content))
#         else:
#             df = pd.read_excel(io.BytesIO(content))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

#     if target_col not in df.columns:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}"
#         )

#     if len(df) < 50:
#         raise HTTPException(status_code=400, detail="Dataset too small. Need at least 50 rows.")

#     job_id = str(uuid.uuid4())
#     background_tasks.add_task(_run_automl, job_id, df, target_col)

#     return {"job_id": job_id, "status": "started", "columns": df.columns.tolist()}


# @router.post("/run/{dataset_id}")
# async def run_automl_for_dataset(
#     dataset_id: str,
#     background_tasks: BackgroundTasks,
# ):
#     """
#     Run AutoML on a pre-loaded dataset (no file upload needed).
#     """
#     from app.services.ml.dataset import load_dataset
#     from pathlib import Path
    
#     DATA_DIR = Path(__file__).parent.parent / "data"
#     dataset_path = DATA_DIR / f"{dataset_id}.parquet"
#     upload_path = DATA_DIR / "uploads" / f"{dataset_id}.parquet"
    
#     if dataset_path.exists():
#         df = pd.read_parquet(dataset_path)
#     elif upload_path.exists():
#         df = pd.read_parquet(upload_path)
#     else:
#         raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    
#     # Load metadata to get target column
#     META_PATH = DATA_DIR / "meta.json"
#     if not META_PATH.exists():
#         raise HTTPException(status_code=404, detail="Dataset metadata not found.")
    
#     with open(META_PATH) as f:
#         meta = json.load(f)
    
#     if dataset_id not in meta:
#         raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not in metadata.")
    
#     target_col = meta[dataset_id]["target"]
    
#     if len(df) < 50:
#         raise HTTPException(status_code=400, detail="Dataset too small. Need at least 50 rows.")
    
#     job_id = str(uuid.uuid4())
#     background_tasks.add_task(_run_automl, job_id, df, target_col)
    
#     return {
#         "job_id": job_id,
#         "status": "started",
#         "dataset_id": dataset_id,
#         "dataset_name": meta[dataset_id].get("name", dataset_id),
#         "target": target_col,
#         "columns": df.columns.tolist(),
#     }


# @router.get("/status/{job_id}")
# async def get_status(job_id: str):
#     if job_id not in automl_results:
#         return {"status": "pending", "progress": 0, "log": []}
#     return automl_results[job_id]

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import pandas as pd
import numpy as np
import json
import io
import uuid
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, r2_score, accuracy_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

router = APIRouter(prefix="/automl", tags=["automl"])

META_PATH = Path(__file__).parent.parent / "data" / "meta.json"
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory job store
automl_results: dict = {}


def _detect_task(y: pd.Series) -> str:
    return "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"


def _run_automl_job(job_id: str, df: pd.DataFrame, target_col: str):
    """Blocking AutoML pipeline — runs in background thread."""
    try:
        automl_results[job_id] = {"status": "running", "progress": 0, "log": []}

        def log(msg: str, pct: int):
            automl_results[job_id]["log"].append(msg)
            automl_results[job_id]["progress"] = pct

        log("Detecting task type...", 5)
        y = df[target_col].copy()
        X_raw = df.drop(columns=[target_col]).copy()
        task = _detect_task(y)
        automl_results[job_id]["task"] = task

        log(f"Task: {task}. Cleaning data...", 15)

        # Drop columns with >50% missing
        high_missing = [c for c in X_raw.columns if X_raw[c].isnull().mean() > 0.5]
        if high_missing:
            X_raw = X_raw.drop(columns=high_missing)
            log(f"Dropped {len(high_missing)} high-missing columns.", 20)

        # Encode categoricals
        log("Encoding categorical columns...", 25)
        cat_cols = X_raw.select_dtypes(exclude="number").columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))

        if task == "classification":
            le_y = LabelEncoder()
            y = pd.Series(le_y.fit_transform(y.astype(str)), index=y.index)

        # Impute
        log("Imputing missing values...", 35)
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

        # Scale
        log("Scaling features...", 40)
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split
        log("Splitting data...", 45)
        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        cv = (StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
              if task == "classification"
              else KFold(n_splits=3, shuffle=True, random_state=42))
        scoring = "f1_weighted" if task == "classification" else "r2"

        log("Running Optuna search (20 trials)...", 55)

        def objective(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 15),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            )
            m = (RandomForestClassifier(**p, random_state=42)
                 if task == "classification"
                 else RandomForestRegressor(**p, random_state=42))
            return float(cross_val_score(m, X_train, y_train, cv=cv, scoring=scoring).mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, timeout=45)

        log("Training best model...", 80)
        best_params = study.best_params
        best_model = (RandomForestClassifier(**best_params, random_state=42)
                      if task == "classification"
                      else RandomForestRegressor(**best_params, random_state=42))
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        metrics: dict = {}
        if task == "classification":
            metrics["accuracy"]    = round(float(accuracy_score(y_test, y_pred)), 4)
            metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        else:
            metrics["r2"]   = round(float(r2_score(y_test, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(np.mean((np.array(y_test) - np.array(y_pred)) ** 2))), 4)

        cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring)
        metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_std"]  = round(float(cv_scores.std()), 4)

        fi: dict = {}
        if hasattr(best_model, "feature_importances_"):
            fi = dict(zip(X.columns, best_model.feature_importances_.tolist()))
            fi = {k: round(float(v), 6) for k, v in sorted(fi.items(), key=lambda x: -x[1])}

        log("Done!", 100)

        automl_results[job_id] = {
            "status": "done",
            "progress": 100,
            "task": task,
            "metrics": metrics,
            "feature_importance": fi,
            "pipeline_summary": {
                "missing_cols_dropped":    high_missing,
                "categorical_cols_encoded": cat_cols,
                "imputation":              "median",
                "scaling":                 "standard",
                "model":                   "RandomForest",
                "best_params":             best_params,
                "n_trials":                len(study.trials),
                "best_trial_score":        round(study.best_value, 4),
            },
            "shape":      {"rows": len(df), "cols": len(df.columns)},
            "log":        automl_results[job_id]["log"],
            "n_features": len(X.columns),
            "train_size": len(X_train),
            "test_size":  len(X_test),
        }

    except Exception as e:
        automl_results[job_id] = {
            "status":   "error",
            "error":    str(e),
            "progress": 0,
            "log":      automl_results.get(job_id, {}).get("log", []),
        }


# ─── ENDPOINT 1: file upload (AutoML tab) ─────────────────────────────────────

@router.post("/run")
async def run_automl_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_col: str = "target",
):
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files supported.")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    try:
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith(".csv") \
             else pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}"
        )
    if len(df) < 50:
        raise HTTPException(status_code=400, detail="Dataset too small. Need at least 50 rows.")

    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_automl_job, job_id, df, target_col)
    return {"job_id": job_id, "status": "started", "columns": df.columns.tolist()}


# ─── ENDPOINT 2: dataset ID (preloaded + uploaded datasets on landing page) ────

@router.post("/run-for-dataset")
async def run_automl_for_dataset(
    background_tasks: BackgroundTasks,
    dataset_id: str,
    target_col: str = "",
):
    """Run AutoML on an already-registered dataset — no file upload needed."""
    meta_path = Path(__file__).parent.parent / "data" / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    if dataset_id not in meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    dataset_meta = meta[dataset_id]
    resolved_target = target_col or dataset_meta.get("target", "")
    if not resolved_target:
        raise HTTPException(status_code=400, detail="No target column specified.")

    # Load from disk
    data_dir = Path(__file__).parent.parent / "data"
    parquet_path = data_dir / f"{dataset_id}.parquet"
    if not parquet_path.exists():
        parquet_path = data_dir / "uploads" / f"{dataset_id}.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk.")

    df = pd.read_parquet(parquet_path)
    if resolved_target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{resolved_target}' not in dataset. Available: {df.columns.tolist()}"
        )

    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_automl_job, job_id, df, resolved_target)
    return {
        "job_id":   job_id,
        "status":   "started",
        "dataset":  dataset_meta.get("name", dataset_id),
        "target":   resolved_target,
        "columns":  df.columns.tolist(),
    }


# ─── ENDPOINT 3: poll status ──────────────────────────────────────────────────

@router.get("/status/{job_id}")
async def get_automl_status(job_id: str):
    if job_id not in automl_results:
        return {"status": "pending", "progress": 0, "log": []}
    return automl_results[job_id]