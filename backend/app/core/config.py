import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR}/ml_playground.db")

settings = type("Settings", (), {"DATABASE_URL": DATABASE_URL})()
