"""
Model persistence — save and load trained models to/from disk.
Uses joblib for efficient sklearn model serialization.
"""
import os
import joblib
import hashlib
import json
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(run_id: str) -> Path:
    return MODELS_DIR / f"{run_id}.joblib"


def _meta_path(run_id: str) -> Path:
    return MODELS_DIR / f"{run_id}_meta.json"


def save_model(run_id: str, model: Any, metadata: dict) -> str:
    """
    Persist a trained model and its metadata.
    Returns the absolute path where the model was saved.
    """
    try:
        joblib.dump(model, _model_path(run_id))
        with open(_meta_path(run_id), "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return str(_model_path(run_id))
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")


def load_model(run_id: str) -> tuple[Any, dict]:
    """
    Load a persisted model and its metadata.
    Returns (model, metadata) or raises if not found.
    """
    model_path = _model_path(run_id)
    meta_path = _meta_path(run_id)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for run_id={run_id}")
    try:
        model = joblib.load(model_path)
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        return model, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def delete_model(run_id: str) -> None:
    """Remove model files from disk."""
    for p in [_model_path(run_id), _meta_path(run_id)]:
        if p.exists():
            p.unlink()


def model_exists(run_id: str) -> bool:
    return _model_path(run_id).exists()


def list_models() -> list[dict]:
    """List all persisted models with their metadata."""
    models = []
    for meta_file in MODELS_DIR.glob("*_meta.json"):
        run_id = meta_file.stem.replace("_meta", "")
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            models.append({
                "run_id": run_id,
                "model_path": str(_model_path(run_id)),
                "size_mb": round(_model_path(run_id).stat().st_size / (1024 * 1024), 2),
                **meta,
            })
        except Exception:
            continue
    return sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)