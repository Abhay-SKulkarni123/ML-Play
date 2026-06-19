from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import datasets, sessions, eda, automl

app = FastAPI(title="ML Playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(sessions.router)
app.include_router(eda.router)
app.include_router(automl.router)

@app.get("/health")
async def health():
    return {"status": "ok"}