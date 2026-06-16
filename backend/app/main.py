from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import datasets, sessions, eda

app = FastAPI(
    title="ML Playground",
    description="Interactive ML lifecycle playground with AI explanations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(sessions.router)
app.include_router(eda.router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ml-playground"}