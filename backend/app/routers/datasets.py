from fastapi import APIRouter, HTTPException
from app.services.ml.dataset import get_all_datasets, profile_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/")
async def list_datasets():
    return get_all_datasets()


@router.get("/{dataset_id}/profile")
async def get_profile(dataset_id: str):
    try:
        return profile_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")