import uuid
import json
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
    LabelEncoder, OrdinalEncoder,
)

from app.core.database import get_db
from app.models.session import MLSession, StepResult as StepResultModel, TrainingRun
from app.schemas.session import (
    SessionCreate, SessionResponse,
    StepRequest, StepResponse,
    TrainRequest, TrainResponse,
)
from app.services.ml.dataset import load_dataset
from app.services.ml import preprocessing
from app.services.ml.training import train_model
from app.services.ml.types import PipelineStepResult
from app.services.ai.explainer import get_explanation
from app.services.export.code_gen import generate_pipeline_code

router = APIRouter(prefix="/sessions", tags=["sessions"])

META_PATH = Path(__file__).parent.parent / "data" / "meta.json"

# Thread pool for running blocking ML work off the async event loop
_ml_executor = ThreadPoolExecutor(max_workers=4)

# Resource limits
MAX_TRAINING_SECONDS = 120
MAX_DATASET_ROWS     = 100_000

# Models that have defined Optuna search spaces
TUNABLE_MODELS = ["random_forest", "xgboost"]


def _load_meta() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


# ─── SESSION ENDPOINTS ────────────────────────────────────────────────────────

@router.post("/", response_model=SessionResponse)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    meta = _load_meta()
    if body.dataset_id not in meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_id}' not found.")

    session = MLSession(
        dataset_id=body.dataset_id,
        task_type=meta[body.dataset_id]["task"],
        current_step=1,
        pipeline_state={},
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await _get_session_or_404(session_id, db)


# ─── STEP ENDPOINTS ───────────────────────────────────────────────────────────

@router.post("/{session_id}/steps/missing", response_model=StepResponse)
async def step_missing(
    session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)
):
    return await _run_step(
        session_id, "missing_values", body, db,
        lambda df, t, p, m: preprocessing.handle_missing(df, t, p),
    )


@router.post("/{session_id}/steps/outliers", response_model=StepResponse)
async def step_outliers(
    session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)
):
    return await _run_step(
        session_id, "outliers", body, db,
        lambda df, t, p, m: preprocessing.handle_outliers(df, t, p),
    )


@router.post("/{session_id}/steps/features", response_model=StepResponse)
async def step_features(
    session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)
):
    return await _run_step(
        session_id, "feature_engineering", body, db,
        lambda df, t, p, m: preprocessing.handle_feature_engineering(df, t, p, m["target"]),
    )


@router.post("/{session_id}/steps/encoding", response_model=StepResponse)
async def step_encoding(
    session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)
):
    return await _run_step(
        session_id, "encoding", body, db,
        lambda df, t, p, m: preprocessing.handle_encoding(df, t, p, m["target"]),
    )


@router.post("/{session_id}/steps/selection", response_model=StepResponse)
async def step_selection(
    session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)
):
    return await _run_step(
        session_id, "feature_selection", body, db,
        lambda df, t, p, m: preprocessing.handle_feature_selection(df, t, p, m["target"]),
    )


@router.post("/{session_id}/steps/scaling", response_model=StepResponse)
async def step_scaling(
    session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)
):
    return await _run_step(
        session_id, "scaling", body, db,
        lambda df, t, p, m: preprocessing.handle_scaling(df, t, p, m["target"]),
    )

@router.post("/{session_id}/steps/pca", response_model=StepResponse)
async def step_pca(
    session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)
):
    return await _run_step(
        session_id, "pca", body, db,
        lambda df, t, p, m: preprocessing.handle_pca(df, t, p, m["target"]),
    )


# ─── TRAIN ENDPOINT ───────────────────────────────────────────────────────────

@router.post("/{session_id}/train", response_model=TrainResponse)
async def train(
    session_id: uuid.UUID,
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_or_404(session_id, db)
    meta = _load_meta()
    dataset_meta = meta[session.dataset_id]
    target_col = dataset_meta["target"]
    task = session.task_type

    # Load dataset FIRST — before any validation that references df
    try:
        df = load_dataset(session.dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    # Now validate
    if len(df) > MAX_DATASET_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has {len(df)} rows. Maximum allowed: {MAX_DATASET_ROWS}.",
        )

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found in dataset.",
        )

    if task == "classification":
        class_counts = df[target_col].value_counts()
        if len(class_counts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Classification requires at least 2 classes in the target column.",
            )
        min_class_size = int(class_counts.min())
        if min_class_size < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Smallest class has only {min_class_size} sample(s). "
                    "Need at least 2 per class for cross-validation. "
                    "Try using drop_rows less aggressively or choose a different dataset."
                ),
            )

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _ml_executor,
                lambda: train_model(
                    df,
                    target_col,
                    session.task_type,
                    body.model_name,
                    body.params,
                    body.test_size,
                    pipeline_state=dict(session.pipeline_state),
                ),
            ),
            timeout=MAX_TRAINING_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=(
                f"Training timed out after {MAX_TRAINING_SECONDS}s. "
                "Try a simpler model or reduce dataset size."
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Training failed: {str(e)}. "
                "Ensure encoding was applied before training."
            ),
        )

    run = TrainingRun(
        session_id=session_id,
        model_name=body.model_name,
        params=body.params,
        metrics=result["metrics"],
        feature_importance=result["feature_importance"],
        pipeline_config=dict(session.pipeline_state),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    return TrainResponse(
        run_id=run.id,
        model=result["model"],
        metrics=result["metrics"],
        feature_importance=result["feature_importance"],
        train_size=result["train_size"],
        test_size=result["test_size"],
        n_features=result["n_features"],
    )

# ─── TUNE ENDPOINT ────────────────────────────────────────────────────────────

@router.post("/{session_id}/tune", response_model=TrainResponse)
async def tune(
    session_id: uuid.UUID,
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

    session = await _get_session_or_404(session_id, db)
    meta = _load_meta()
    dataset_meta = meta[session.dataset_id]
    target_col = dataset_meta["target"]
    task = session.task_type
    model_name = body.model_name

    df = load_dataset(session.dataset_id)

    from app.services.ml.training import _apply_pipeline_split_aware
    from sklearn.model_selection import train_test_split

    X_raw = df.drop(columns=[target_col])
    y = df[target_col]
    stratify = y if task == "classification" else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=body.test_size, random_state=42, stratify=stratify
    )
    X_train, _ = _apply_pipeline_split_aware(
        X_train_raw.copy(), X_test_raw.copy(), y_train,
        dict(session.pipeline_state), target_col,
    )
    X_train = X_train.select_dtypes(include="number")
    y_train_aligned = y_train.loc[X_train.index] if hasattr(y_train, 'loc') else y_train

    cv = (
        StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        if task == "classification"
        else KFold(n_splits=3, shuffle=True, random_state=42)
    )
    scoring = "f1_weighted" if task == "classification" else "r2"

    # ── Search spaces per model ───────────────────────────────────────────────
    def get_model(trial, name: str):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from xgboost import XGBClassifier, XGBRegressor
        from lightgbm import LGBMClassifier, LGBMRegressor

        if name == "random_forest":
            p = {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
                "max_depth":         trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "random_state": 42,
            }
            return RandomForestClassifier(**p) if task == "classification" else RandomForestRegressor(**p)

        elif name == "xgboost":
            p = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                "max_depth":     trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42, "verbosity": 0,
            }
            if task == "classification":
                return XGBClassifier(**p, eval_metric="logloss")
            return XGBRegressor(**p)

        elif name == "lightgbm":
            p = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                "max_depth":     trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves":    trial.suggest_int("num_leaves", 20, 100),
                "random_state": 42, "verbose": -1,
            }
            return LGBMClassifier(**p) if task == "classification" else LGBMRegressor(**p)

        elif name == "logistic_regression":
            p = {
                "C":        trial.suggest_float("C", 0.01, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
            }
            return LogisticRegression(**p)

        elif name == "decision_tree":
            p = {
                "max_depth":         trial.suggest_int("max_depth", 1, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
            }
            return DecisionTreeClassifier(**p)

        elif name == "knn":
            p = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p":           trial.suggest_int("p", 1, 2),
            }
            return KNeighborsClassifier(**p)

        elif name == "gradient_boosting":
            p = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 200),
                "max_depth":     trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "random_state": 42,
            }
            return GradientBoostingClassifier(**p)

        elif name == "ridge":
            p = {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}
            return Ridge(**p)

        elif name == "lasso":
            p = {"alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True)}
            return Lasso(**p)

        else:
            # Fallback to random forest
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth":    trial.suggest_int("max_depth", 2, 15),
                "random_state": 42,
            }
            return RandomForestClassifier(**p) if task == "classification" else RandomForestRegressor(**p)

    def objective(trial) -> float:
        model = get_model(trial, model_name)
        scores = cross_val_score(model, X_train, y_train_aligned, cv=cv, scoring=scoring)
        return float(scores.mean())

    loop = asyncio.get_event_loop()
    try:
        def _run_study():
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20, timeout=60)
            return study

        study = await asyncio.wait_for(
            loop.run_in_executor(_ml_executor, _run_study),
            timeout=90,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Hyperparameter search timed out after 90s.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tuning failed: {str(e)}")

    best_params = study.best_params
    trials_data = [
        {"number": t.number, "value": round(t.value, 4), "params": t.params}
        for t in study.trials if t.value is not None
    ]

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _ml_executor,
                lambda: train_model(
                    df, target_col, task, model_name,
                    best_params, body.test_size,
                    pipeline_state=dict(session.pipeline_state),
                ),
            ),
            timeout=MAX_TRAINING_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"Final training timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Final training failed: {str(e)}")

    final_metrics = {
        **result["metrics"],
        "best_trial_score": round(study.best_value, 4),
        "n_trials": len(study.trials),
        "trials": trials_data,
    }

    run = TrainingRun(
        session_id=session_id,
        model_name=model_name,
        params=best_params,
        metrics=final_metrics,
        feature_importance=result["feature_importance"],
        pipeline_config=dict(session.pipeline_state),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    return TrainResponse(
        run_id=run.id,
        model=model_name,
        metrics=final_metrics,
        feature_importance=result["feature_importance"],
        train_size=result["train_size"],
        test_size=result["test_size"],
        n_features=result["n_features"],
    )


# ─── EXPORT ENDPOINT ──────────────────────────────────────────────────────────

@router.get("/{session_id}/export")
async def export_code(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await _get_session_or_404(session_id, db)

    result = await db.execute(
        select(TrainingRun)
        .where(TrainingRun.session_id == session_id)
        .order_by(desc(TrainingRun.created_at))
        .limit(1)
    )
    last_run = result.scalar_one_or_none()

    session_data = {
        "dataset_id": session.dataset_id,
        "task_type":  session.task_type,
        "pipeline_state": {
            **session.pipeline_state,
            **({"model": {"name": last_run.model_name}} if last_run else {}),
        },
    }

    code = generate_pipeline_code(session_data)
    return PlainTextResponse(content=code, media_type="text/plain")


# ─── RUNS ENDPOINT ────────────────────────────────────────────────────────────

@router.get("/{session_id}/runs")
async def get_runs(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingRun)
        .where(TrainingRun.session_id == session_id)
        .order_by(desc(TrainingRun.created_at))
    )
    runs = result.scalars().all()
    return [
        {
            "id":                str(r.id),
            "model_name":        r.model_name,
            "params":            r.params,
            "metrics":           r.metrics,
            "feature_importance":r.feature_importance,
            "created_at":        r.created_at.isoformat(),
        }
        for r in runs
    ]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

async def _get_session_or_404(
    session_id: uuid.UUID, db: AsyncSession
) -> MLSession:
    result = await db.execute(
        select(MLSession).where(MLSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


async def _run_step(
    session_id: uuid.UUID,
    step_name: str,
    body: StepRequest,
    db: AsyncSession,
    fn,
) -> StepResponse:
    session = await _get_session_or_404(session_id, db)
    meta = _load_meta()
    dataset_meta = meta[session.dataset_id]

    # If step already applied, return the existing result from DB
    if step_name in session.pipeline_state:
        existing = await db.execute(
            select(StepResultModel)
            .where(
                StepResultModel.session_id == session_id,
                StepResultModel.step_name == step_name,
            )
            .order_by(desc(StepResultModel.created_at))
            .limit(1)
        )
        existing_rec = existing.scalar_one_or_none()
        if existing_rec:
            return StepResponse(
                step=existing_rec.step_name,
                technique=existing_rec.technique,
                params=existing_rec.params,
                stats=existing_rec.stats,
                warnings=existing_rec.warnings,
                ai_explanation=existing_rec.ai_explanation or "",
            )
        raise HTTPException(
            status_code=409,
            detail=f"Step '{step_name}' already applied. Start a new session to change it.",
        )

    try:
        df = load_dataset(session.dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Validate dataset is usable
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="Dataset is empty.")
    if len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="Dataset has no columns.")

    loop = asyncio.get_event_loop()
    try:
        result: PipelineStepResult = await loop.run_in_executor(
            _ml_executor,
            lambda: fn(df, body.technique, body.params, dataset_meta),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "step":       step_name,
                "technique":  body.technique,
                "error":      str(e),
                "suggestion": (
                    "Check that your dataset has the expected "
                    "column types for this technique."
                ),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "step":      step_name,
                "technique": body.technique,
                "error":     str(e),
            },
        )

    # AI explanation — non-blocking, falls back gracefully
    try:
        explanation, recommendation = await get_explanation(result, dataset_meta["name"])
    except Exception:
        explanation = _fallback_explanation(result)
        recommendation = ""

    step_rec = StepResultModel(
        session_id=session_id,
        step_number=session.current_step,
        step_name=step_name,
        technique=body.technique,
        params=body.params,
        stats=result.stats,
        warnings=result.warnings,
        ai_explanation=explanation,
    )
    db.add(step_rec)

    pipeline_state = dict(session.pipeline_state)
    pipeline_state[step_name] = {
        "technique": body.technique,
        "params":    body.params,
    }
    session.pipeline_state = pipeline_state
    session.current_step   = session.current_step + 1

    await db.commit()

    return StepResponse(
        step=step_name,
        technique=body.technique,
        params=body.params,
        stats=result.stats,
        warnings=result.warnings,
        ai_explanation=explanation,
        ai_recommendation=recommendation,
    )


def _fallback_explanation(result: PipelineStepResult) -> str:
    """Rule-based fallback when LLM is unavailable."""
    stats = result.stats
    w     = " ".join(result.warnings) if result.warnings else ""

    if result.step == "missing_values":
        before = stats.get("missing_before", 0)
        after  = stats.get("missing_after", 0)
        return (
            f"Applied {result.technique} imputation. "
            f"Missing values reduced from {before} to {after}. {w}"
        ).strip()

    if result.step == "outliers":
        total      = stats.get("total_outliers_found", 0)
        rows_after = stats.get("rows_after", "?")
        return (
            f"Applied {result.technique}. "
            f"{total} outliers found. "
            f"Dataset now has {rows_after} rows. {w}"
        ).strip()

    if result.step == "encoding":
        cols = stats.get("encoded_columns", [])
        new  = stats.get("new_cols_created", 0)
        return (
            f"Encoded {len(cols)} categorical columns using {result.technique}. "
            f"{new} new columns created. {w}"
        ).strip()

    if result.step == "scaling":
        n = stats.get("n_columns_scaled", 0)
        return (
            f"Applied {result.technique} to {n} numeric columns. {w}"
        ).strip()

    if result.step == "feature_engineering":
        new = stats.get("new_features_created", 0)
        return (
            f"Created {new} new features using {result.technique}. {w}"
        ).strip()

    if result.step == "feature_selection":
        dropped = stats.get("n_dropped", 0)
        return (
            f"Dropped {dropped} features using {result.technique}. {w}"
        ).strip()

    return f"Applied {result.technique} to step '{result.step}'. {w}".strip()

class PredictRequest(BaseModel):
    input_data: dict


@router.post("/{session_id}/predict")
async def predict(
    session_id: uuid.UUID,
    body: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_or_404(session_id, db)
    meta = _load_meta()
    dataset_meta = meta[session.dataset_id]
    target_col = dataset_meta["target"]
    task = session.task_type

    result = await db.execute(
        select(TrainingRun)
        .where(TrainingRun.session_id == session_id)
        .order_by(desc(TrainingRun.created_at))
        .limit(1)
    )
    last_run = result.scalar_one_or_none()
    if not last_run:
        raise HTTPException(status_code=400, detail="Train a model first before predicting.")

    df = load_dataset(session.dataset_id)
    X_full = df.drop(columns=[target_col])
    y_full = df[target_col]

    from app.services.ml.training import _apply_pipeline_split_aware
    from sklearn.model_selection import train_test_split

    stratify = y_full if task == "classification" else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=stratify
    )
    X_train, _, y_train = _apply_pipeline_split_aware(
        X_train_raw.copy(), X_test_raw.copy(), y_train,
        dict(session.pipeline_state), target_col,
    )
    X_train = X_train.select_dtypes(include="number")

    input_df = pd.DataFrame([body.input_data])
    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = X_train[col].median() if col in X_train.columns else 0
    input_df = input_df[X_train.columns]

    from app.services.ml.training import _build_classifier, _build_regressor
    model = (_build_classifier(last_run.model_name, last_run.params)
             if task == "classification"
             else _build_regressor(last_run.model_name, last_run.params))
    model.fit(X_train, y_train)
    pred = model.predict(input_df)[0]

    confidence = None
    if task == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        confidence = round(float(max(proba)), 4)

    return {
        "prediction": float(pred) if isinstance(pred, (int, float, np.integer, np.floating)) else str(pred),
        "confidence": confidence,
        "model_used": last_run.model_name,
    }