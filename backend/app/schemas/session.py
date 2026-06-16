from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Any


class SessionCreate(BaseModel):
    dataset_id: str


class SessionResponse(BaseModel):
    id: UUID
    dataset_id: str
    task_type: str
    current_step: int
    pipeline_state: dict
    created_at: datetime

    class Config:
        from_attributes = True


class StepRequest(BaseModel):
    technique: str
    params: dict = {}


class StepResponse(BaseModel):
    step: str
    technique: str
    params: dict = {}
    stats: dict = {}
    warnings: list[str] = []
    ai_explanation: str = ""
    ai_recommendation: str = ""
    metrics_delta: dict = {}


class TrainRequest(BaseModel):
    model_name: str
    params: dict = {}
    test_size: float = 0.2


class TrainResponse(BaseModel):
    run_id: UUID
    model: str
    metrics: dict
    feature_importance: dict
    train_size: int
    test_size: int
    n_features: int