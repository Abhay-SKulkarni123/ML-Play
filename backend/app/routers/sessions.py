import uuid
import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
    LabelEncoder, OrdinalEncoder
)

from app.core.database import get_db
from app.models.session import MLSession, StepResult as StepResultModel, TrainingRun
from app.schemas.session import (
    SessionCreate, SessionResponse,
    StepRequest, StepResponse,
    TrainRequest, TrainResponse
)
from app.services.ml.dataset import load_dataset
from app.services.ml import preprocessing
from app.services.ml.training import train_model
from app.services.ml.types import PipelineStepResult
from app.services.ai.explainer import get_explanation
from app.services.export.code_gen import generate_pipeline_code

router = APIRouter(prefix="/sessions", tags=["sessions"])

META_PATH = Path(__file__).parent.parent / "data" / "meta.json"


def _load_meta() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


# ─── SESSION ENDPOINTS ────────────────────────────────────────────────────────

@router.post("/", response_model=SessionResponse)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    meta = _load_meta()
    if body.dataset_id not in meta:
        raise HTTPException(status_code=404, detail="Dataset not found")
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
async def step_missing(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "missing_values", body, db,
        lambda df, t, p, m: preprocessing.handle_missing(df, t, p))


@router.post("/{session_id}/steps/outliers", response_model=StepResponse)
async def step_outliers(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "outliers", body, db,
        lambda df, t, p, m: preprocessing.handle_outliers(df, t, p))


@router.post("/{session_id}/steps/features", response_model=StepResponse)
async def step_features(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "feature_engineering", body, db,
        lambda df, t, p, m: preprocessing.handle_feature_engineering(df, t, p, m["target"]))


@router.post("/{session_id}/steps/encoding", response_model=StepResponse)
async def step_encoding(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "encoding", body, db,
        lambda df, t, p, m: preprocessing.handle_encoding(df, t, p, m["target"]))


@router.post("/{session_id}/steps/selection", response_model=StepResponse)
async def step_selection(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "feature_selection", body, db,
        lambda df, t, p, m: preprocessing.handle_feature_selection(df, t, p, m["target"]))


@router.post("/{session_id}/steps/scaling", response_model=StepResponse)
async def step_scaling(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "scaling", body, db,
        lambda df, t, p, m: preprocessing.handle_scaling(df, t, p, m["target"]))


# ─── TRAIN ENDPOINT ───────────────────────────────────────────────────────────

@router.post("/{session_id}/train", response_model=TrainResponse)
async def train(session_id: uuid.UUID, body: TrainRequest, db: AsyncSession = Depends(get_db)):
    session = await _get_session_or_404(session_id, db)
    meta = _load_meta()
    dataset_meta = meta[session.dataset_id]
    target_col = dataset_meta["target"]

    df = _replay_pipeline(session)
    result = train_model(df, target_col, session.task_type, body.model_name, body.params, body.test_size)

    run = TrainingRun(
        session_id=session_id,
        model_name=body.model_name,
        params=body.params,
        metrics=result["metrics"],
        feature_importance=result["feature_importance"],
        pipeline_config=session.pipeline_state,
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
        "task_type": session.task_type,
        "pipeline_state": {
            **session.pipeline_state,
            **({"model": {"name": last_run.model_name}} if last_run else {}),
        },
    }

    code = generate_pipeline_code(session_data)
    return PlainTextResponse(content=code, media_type="text/plain")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

async def _get_session_or_404(session_id: uuid.UUID, db: AsyncSession) -> MLSession:
    result = await db.execute(select(MLSession).where(MLSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def _run_step(session_id, step_name, body, db, fn) -> StepResponse:
    session = await _get_session_or_404(session_id, db)
    meta = _load_meta()
    dataset_meta = meta[session.dataset_id]

    df = load_dataset(session.dataset_id)
    result: PipelineStepResult = fn(df, body.technique, body.params, dataset_meta)
    explanation = await get_explanation(result, dataset_meta["name"])

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
    pipeline_state[step_name] = {"technique": body.technique, "params": body.params}
    session.pipeline_state = pipeline_state
    session.current_step = session.current_step + 1

    await db.commit()

    return StepResponse(
        step=step_name,
        technique=body.technique,
        params=body.params,
        stats=result.stats,
        warnings=result.warnings,
        ai_explanation=explanation,
    )


def _replay_pipeline(session: MLSession) -> pd.DataFrame:
    df = load_dataset(session.dataset_id)
    meta = _load_meta()
    target = meta[session.dataset_id]["target"]
    state = session.pipeline_state

    step_order = [
        "missing_values",
        "outliers",
        "feature_engineering",
        "encoding",
        "feature_selection",
        "scaling",
    ]

    for step_name in step_order:
        if step_name in state:
            s = state[step_name]
            df = _transform(df, step_name, s["technique"], s["params"], target)

    return df


def _transform(df: pd.DataFrame, step_name: str, technique: str, params: dict, target: str) -> pd.DataFrame:
    df = df.copy()

    if step_name == "missing_values":
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        if technique == "mean":
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "missing", inplace=True)
        elif technique == "median":
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "missing", inplace=True)
        elif technique == "mode":
            for col in df.columns:
                if not df[col].mode().empty:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        elif technique == "knn":
            imp = KNNImputer(n_neighbors=params.get("n_neighbors", 5))
            df[numeric_cols] = imp.fit_transform(df[numeric_cols])
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "missing", inplace=True)
        elif technique == "mice":
            imp = IterativeImputer(max_iter=10, random_state=42)
            df[numeric_cols] = imp.fit_transform(df[numeric_cols])
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "missing", inplace=True)
        elif technique == "constant":
            df.fillna(params.get("fill_value", 0), inplace=True)
        elif technique == "drop_rows":
            df.dropna(inplace=True)
        elif technique == "drop_cols":
            threshold = params.get("threshold", 0.5)
            cols_to_drop = [c for c in df.columns if df[c].isnull().mean() > threshold]
            df.drop(columns=cols_to_drop, inplace=True)

    elif step_name == "outliers":
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if technique == "iqr_cap":
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        elif technique == "zscore_remove":
            threshold = params.get("threshold", 3.0)
            mask = pd.Series([True] * len(df), index=df.index)
            for col in numeric_cols:
                z = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask = mask & (z <= threshold)
            df = df[mask]
        elif technique == "log_transform":
            for col in numeric_cols:
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])

    elif step_name == "feature_engineering":
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]

        if technique == "polynomial":
            cols_to_use = numeric_cols[:5]
            for i, col1 in enumerate(cols_to_use):
                for col2 in cols_to_use[i:]:
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        elif technique == "interaction":
            cols_to_use = numeric_cols[:6]
            for i in range(len(cols_to_use)):
                for j in range(i + 1, len(cols_to_use)):
                    col1, col2 = cols_to_use[i], cols_to_use[j]
                    df[f"{col1}_times_{col2}"] = df[col1] * df[col2]
        elif technique == "log_features":
            for col in numeric_cols:
                if (df[col] > 0).all():
                    df[f"log_{col}"] = np.log1p(df[col])
        elif technique == "ratio":
            cols_to_use = numeric_cols[:4]
            for i in range(len(cols_to_use)):
                for j in range(len(cols_to_use)):
                    if i != j and (df[cols_to_use[j]] != 0).all():
                        df[f"{cols_to_use[i]}_div_{cols_to_use[j]}"] = df[cols_to_use[i]] / df[cols_to_use[j]]
        elif technique == "binning":
            n_bins = params.get("n_bins", 5)
            for col in numeric_cols[:5]:
                df[f"{col}_binned"] = pd.cut(df[col], bins=n_bins, labels=False, duplicates="drop")
        elif technique == "sqrt_features":
            for col in numeric_cols:
                if (df[col] >= 0).all():
                    df[f"sqrt_{col}"] = np.sqrt(df[col])

    elif step_name == "encoding":
        cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c != target]

        if technique == "onehot":
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        elif technique == "label":
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        elif technique == "ordinal":
            enc = OrdinalEncoder()
            df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
        elif technique == "frequency":
            for col in cat_cols:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map)
        elif technique == "target":
            for col in cat_cols:
                means = df.groupby(col)[target].mean().to_dict()
                df[col] = df[col].map(means)

    elif step_name == "feature_selection":
        X = df.drop(columns=[target]).select_dtypes(include="number")
        y = df[target]

        if technique == "variance_threshold":
            from sklearn.feature_selection import VarianceThreshold
            threshold = params.get("threshold", 0.01)
            sel = VarianceThreshold(threshold=threshold)
            sel.fit(X)
            low_var = X.columns[~sel.get_support()].tolist()
            df.drop(columns=low_var, inplace=True)
        elif technique == "correlation":
            threshold = params.get("threshold", 0.95)
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            df.drop(columns=to_drop, inplace=True)
        elif technique == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
            k = params.get("k", min(10, len(X.columns)))
            score_fn = mutual_info_classif if len(y.unique()) < 20 else mutual_info_regression
            sel = SelectKBest(score_fn, k=k)
            sel.fit(X, y)
            kept = X.columns[sel.get_support()].tolist()
            non_numeric = [c for c in df.columns if c not in X.columns and c != target]
            keep_cols = kept + [target] + non_numeric
            df = df[[c for c in keep_cols if c in df.columns]]

    elif step_name == "scaling":
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
        scalers = {
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "robust":   RobustScaler(),
            "maxabs":   MaxAbsScaler(),
            "quantile": QuantileTransformer(output_distribution="uniform", random_state=42),
            "power":    PowerTransformer(method="yeo-johnson"),
        }
        if technique in scalers:
            df[numeric_cols] = scalers[technique].fit_transform(df[numeric_cols])

    return df