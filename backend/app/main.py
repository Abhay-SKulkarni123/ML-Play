from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import datasets, sessions, eda, automl
from app.core.database import engine, Base
from app.models.session import MLSession, StepResult as StepResultModel, TrainingRun

app = FastAPI(
    title="ML Playground",
    description="Interactive ML lifecycle playground with AI explanations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://YOUR_CUSTOM_DOMAIN.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(sessions.router)
app.include_router(eda.router)
app.include_router(automl.router)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ml-playground"}
