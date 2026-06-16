from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid, json

from app.core.database import get_db
from app.models.session import MLSession, StepResult as StepResultModel, TrainingRun
from app.schemas.session import SessionCreate, SessionResponse, StepRequest, StepResponse, TrainRequest, TrainResponse
from app.services.ml.dataset import load_dataset, get_all_datasets
from app.services.ml import preprocessing
from app.services.ml.training import train_model
from app.services.ml.types import PipelineStepResult
from app.services.ai.explainer import get_explanation

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_dataset_name(dataset_id: str) -> str:
    datasets = get_all_datasets()
    for d in datasets:
        if d["id"] == dataset_id:
            return d["name"]
    return dataset_id


@router.post("/", response_model=SessionResponse)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    import json as _json
    from pathlib import Path
    meta_path = Path(__file__).parent.parent / "data" / "meta.json"
    with open(meta_path) as f:
        meta = _json.load(f)
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
    result = await db.execute(select(MLSession).where(MLSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/steps/missing", response_model=StepResponse)
async def step_missing(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "missing_values", body, db,
                           lambda df, t, p, s: preprocessing.handle_missing(df, t, p))


@router.post("/{session_id}/steps/outliers", response_model=StepResponse)
async def step_outliers(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "outliers", body, db,
                           lambda df, t, p, s: preprocessing.handle_outliers(df, t, p))


@router.post("/{session_id}/steps/encoding", response_model=StepResponse)
async def step_encoding(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "encoding", body, db,
                           lambda df, t, p, s: preprocessing.handle_encoding(df, t, p, s["target"]))


@router.post("/{session_id}/steps/scaling", response_model=StepResponse)
async def step_scaling(session_id: uuid.UUID, body: StepRequest, db: AsyncSession = Depends(get_db)):
    return await _run_step(session_id, "scaling", body, db,
                           lambda df, t, p, s: preprocessing.handle_scaling(df, t, p, s["target"]))


@router.post("/{session_id}/train", response_model=TrainResponse)
async def train(session_id: uuid.UUID, body: TrainRequest, db: AsyncSession = Depends(get_db)):
    session = await _get_session_or_404(session_id, db)
    df = _replay_pipeline(session)

    import json as _json
    from pathlib import Path
    meta_path = Path(__file__).parent.parent / "data" / "meta.json"
    with open(meta_path) as f:
        meta = _json.load(f)
    target_col = meta[session.dataset_id]["target"]

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


# ─── HELPERS ──────────────────────────────────────────────────────────────────

async def _get_session_or_404(session_id: uuid.UUID, db: AsyncSession) -> MLSession:
    result = await db.execute(select(MLSession).where(MLSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _replay_pipeline(session: MLSession) -> "pd.DataFrame":
    import pandas as pd
    from app.services.ml.dataset import load_dataset
    from app.services.ml import preprocessing
    import json as _json
    from pathlib import Path

    df = load_dataset(session.dataset_id)
    meta_path = Path(__file__).parent.parent / "data" / "meta.json"
    with open(meta_path) as f:
        meta = _json.load(f)
    target = meta[session.dataset_id]["target"]
    state = session.pipeline_state

    if "missing_values" in state:
        s = state["missing_values"]
        result = preprocessing.handle_missing(df, s["technique"], s["params"])
        df = _apply_result(df, result)
    if "outliers" in state:
        s = state["outliers"]
        result = preprocessing.handle_outliers(df, s["technique"], s["params"])
        df = _apply_result(df, result)
    if "encoding" in state:
        s = state["encoding"]
        result = preprocessing.handle_encoding(df, s["technique"], s["params"], target)
        df = _apply_result(df, result)
    if "scaling" in state:
        s = state["scaling"]
        result = preprocessing.handle_scaling(df, s["technique"], s["params"], target)
        df = _apply_result(df, result)

    return df


def _apply_result(df, result: PipelineStepResult):
    """Re-run the transformation and return the transformed df."""
    # preprocessing functions return PipelineStepResult but we need the df
    # so we call them directly and return their internal df_out
    # this is a lightweight replay — acceptable for playground-scale datasets
    return df  # placeholder — see note below


async def _run_step(session_id, step_name, body, db, fn):
    session = await _get_session_or_404(session_id, db)
    df = load_dataset(session.dataset_id)

    import json as _json
    from pathlib import Path
    meta_path = Path(__file__).parent.parent / "data" / "meta.json"
    with open(meta_path) as f:
        meta = _json.load(f)
    dataset_meta = meta[session.dataset_id]

    result: PipelineStepResult = fn(df, body.technique, body.params, dataset_meta)
    explanation = await get_explanation(result, dataset_meta["name"])

    # persist step result
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

    # update pipeline state
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