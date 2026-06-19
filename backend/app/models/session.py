import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, JSON, ForeignKey, Integer, Float, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class MLSession(Base):
    __tablename__ = "ml_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(100))
    task_type: Mapped[str] = mapped_column(String(50), default="unknown")
    current_step: Mapped[int] = mapped_column(Integer, default=1)
    pipeline_state: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    step_results: Mapped[list["StepResult"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    runs: Mapped[list["TrainingRun"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class StepResult(Base):
    __tablename__ = "step_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_sessions.id"))
    step_number: Mapped[int] = mapped_column(Integer)
    step_name: Mapped[str] = mapped_column(String(100))
    technique: Mapped[str] = mapped_column(String(100))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    ai_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["MLSession"] = relationship(back_populates="step_results")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_sessions.id"))
    model_name: Mapped[str] = mapped_column(String(100))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_importance: Mapped[dict] = mapped_column(JSON, default=dict)
    shap_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pipeline_config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["MLSession"] = relationship(back_populates="runs")