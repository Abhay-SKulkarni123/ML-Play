# ML Playground

An interactive, step-by-step machine learning lifecycle tool built with FastAPI, Next.js, and PostgreSQL. Walk through the full ML pipeline from raw data to trained model, with AI explanations at every decision point.

**[Live Demo →](YOUR_RAILWAY_URL)**

---

## What it does

ML Playground guides you through 13 steps of the ML lifecycle on any tabular dataset:

1. **Data Profile** — shape, types, missing values, duplicates
2. **EDA** — distributions, correlation heatmap, target analysis, 3D scatter plots
3. **Missing Values** — 10 techniques (mean, median, KNN, MICE, indicator, constant, mode, random sample, drop rows/cols)
4. **Outlier Treatment** — IQR capping, z-score removal, percentile capping, log transform
5. **Feature Engineering** — polynomial, interaction, log, reciprocal, sqrt, ratio, binning, date decomposition
6. **Encoding** — one-hot, label, ordinal, frequency, target mean
7. **Feature Selection** — variance threshold, correlation filter, mutual information
8. **Dimensionality Reduction** — PCA auto (95% variance) or fixed components
9. **Scaling** — standard, min-max, robust, max-abs, quantile, power (Yeo-Johnson)
10. **Model Training** — 15 algorithms with individual hyperparameter panels and live effect previews
11. **Hyperparameter Tuning** — Bayesian search (Optuna, 60 trials) or Grid Search
12. **Explainability** — feature importance bar chart and radar view
13. **Experiment Comparison** — compare any two training runs side by side

**Also included:**
- AutoML mode — automatic pipeline with Optuna tuning, available for all datasets
- Prediction step — enter feature values and get a live prediction from your trained model
- Python code export — download the full preprocessing and training pipeline as a script
- AI explanations — every step explained by Claude or GPT with a next-step recommendation
- Upload your own CSV or Excel dataset

---

## Architecture

┌─────────────────────────────────────────────────────────┐

│                    Next.js Frontend                      │

│  Landing page · 13-step playground · AutoML tab         │

│  Recharts visualizations · Real-time polling             │

└──────────────────────┬──────────────────────────────────┘

│ HTTP / REST

┌──────────────────────▼──────────────────────────────────┐

│                   FastAPI Backend                        │

│  /sessions  /datasets  /eda  /automl                    │

│  Async routes · ThreadPoolExecutor for ML workloads      │

└──────────┬───────────────────────┬───────────────────────┘

│                       │

┌──────────▼──────┐    ┌──────────▼──────────────────────┐

│   PostgreSQL     │    │         ML Pipeline              │

│   Sessions       │    │  scikit-learn · XGBoost          │

│   Step results   │    │  LightGBM · CatBoost · Optuna    │

│   Training runs  │    │  Split-aware preprocessing       │

└─────────────────┘    └──────────────────────────────────┘

### Key technical decisions

**Split-aware preprocessing pipeline** — all preprocessing steps fit on the training set only and transform the test set separately. This prevents data leakage at every step, which is the most common mistake in beginner ML pipelines.

**Async ML training** — sklearn training is synchronous and CPU-bound. The backend runs it in a `ThreadPoolExecutor` so the FastAPI async event loop stays unblocked. Training has a 120-second timeout with a clean error response.

**Experiment tracking** — every training run is persisted to PostgreSQL with its parameters, metrics, and feature importances. Any two runs can be compared side by side.

**AI explanations with fallback** — the app tries Anthropic Claude, then OpenAI, then falls back to deterministic rule-based explanations. The UI never crashes due to missing API keys.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Backend | FastAPI, Python 3.10+, Uvicorn |
| Database | PostgreSQL 15, SQLAlchemy (async), Alembic |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| AI | Anthropic Claude API (with OpenAI fallback) |

---

## Running locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 15

### Backend

```bash
cd backend
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your DATABASE_URL and optionally API keys

alembic upgrade head

uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000

npm run dev
```

Open `http://localhost:3000`.

---

## Environment variables

### `backend/.env`
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ml_playground

ANTHROPIC_API_KEY=            # optional — enables real AI explanations

OPENAI_API_KEY=               # optional — fallback AI provider

LLM_PROVIDER=anthropic        # anthropic | openai | none

### `frontend/.env.local`
NEXT_PUBLIC_API_URL=http://localhost:8000

---

## Deploying

### Backend → Railway

```bash
npm install -g @railway/cli
railway login
cd backend
railway init
railway add postgresql
railway up
```

Set `DATABASE_URL` and optional API keys in the Railway dashboard environment variables. Railway auto-detects Python and installs from `requirements.txt`.

### Frontend → Vercel

```bash
npm install -g vercel
cd frontend
vercel
```

When prompted for environment variables, set `NEXT_PUBLIC_API_URL` to your Railway backend URL.

After deploying, run migrations on Railway:

```bash
railway run alembic upgrade head
```

---

## Project structure
ml-playground/

├── README.md

├── .gitignore

├── backend/

│   ├── .env.example

│   ├── requirements.txt

│   ├── Procfile

│   ├── railway.json

│   ├── alembic.ini

│   ├── migrations/

│   └── app/

│       ├── main.py

│       ├── core/           # database.py, config.py

│       ├── models/         # SQLAlchemy models

│       ├── routers/        # sessions, datasets, eda, automl

│       ├── schemas/        # Pydantic schemas

│       └── services/

│           ├── ml/         # preprocessing, training, dataset

│           ├── ai/         # explainer

│           └── export/     # code_gen

└── frontend/

├── app/

│   ├── page.tsx             # Landing page + AutoML tab

│   └── playground/

│       └── [sessionId]/

│           └── page.tsx     # 13-step playground

└── lib/

└── api.ts               # Typed API client

---

## Resume bullets

- Built an end-to-end interactive ML lifecycle platform (FastAPI, Next.js, PostgreSQL) with a split-aware preprocessing pipeline that prevents data leakage across 13 configurable steps supporting 15 algorithms
- Implemented async ML training with ThreadPoolExecutor, Optuna Bayesian hyperparameter search (60 trials), full experiment tracking with run comparison, and Python pipeline code export
- Integrated Anthropic Claude AI explanations with OpenAI fallback and deterministic rule-based fallback, plus AutoML background jobs with real-time progress polling via interval-based status endpoint

---

## License

MIT
