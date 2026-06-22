# ML Playground

An interactive, step-by-step machine learning lifecycle tool built with FastAPI, Next.js, and PostgreSQL. Walk through the full ML pipeline — from raw data to trained model — with AI explanations at every decision point.

**[Live Demo →](YOUR_RAILWAY_URL)**

![ML Playground Screenshot](docs/screenshot.png)

---

## What it does

ML Playground guides you through 13 steps of the ML lifecycle on any tabular dataset:

1. **Data Profile** — shape, types, missing values, duplicates
2. **EDA** — distributions, correlation heatmap, target analysis, scatter plots
3. **Missing Values** — 10 imputation techniques (mean, median, KNN, MICE, drop rows/cols, indicator, constant, mode, random sample)
4. **Outlier Treatment** — IQR capping, z-score removal, percentile capping, log transform
5. **Feature Engineering** — polynomial, interaction, log, reciprocal, sqrt, ratio, binning, date decomposition
6. **Encoding** — one-hot, label, ordinal, frequency, target mean
7. **Feature Selection** — variance threshold, correlation filter, mutual information
8. **Dimensionality Reduction** — PCA auto (95% variance) or fixed components
9. **Scaling** — standard, min-max, robust, max-abs, quantile, power (Yeo-Johnson)
10. **Model Training** — 15 algorithms with individual hyperparameter panels
11. **Hyperparameter Tuning** — Bayesian search (Optuna, 20 trials) or Grid Search
12. **Explainability** — feature importance bar chart and radar view
13. **Experiment Comparison** — compare any two training runs side by side

**Plus:**
- AutoML mode — automatic pipeline + Optuna tuning for one-click results
- Prediction step — enter feature values and get a live prediction from your trained model
- Python code export — download the full preprocessing + training pipeline as a runnable script
- AI explanations — every applied step explained by Claude/GPT with a next-step recommendation
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

**Split-aware preprocessing pipeline** — all preprocessing steps (scaling, encoding, imputation) fit on the training set only and transform the test set. This prevents data leakage at every step, which is the most common mistake in beginner ML pipelines.

**Async ML training** — sklearn training is synchronous and CPU-bound. The backend runs it in a `ThreadPoolExecutor` so the FastAPI async loop stays unblocked. Training has a 120-second timeout with a clean error message.

**Experiment tracking** — every training run is persisted to PostgreSQL with its parameters, metrics, and feature importances. Any two runs can be compared side by side.

**AI explanations with fallback** — the app tries Anthropic's Claude, then OpenAI, then falls back to deterministic rule-based explanations. The UI never crashes due to missing API keys.

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
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your DB URL and API keys

# Run migrations
alembic upgrade head

# Start server
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

### Backend `.env`
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ml_playground

ANTHROPIC_API_KEY=sk-ant-...        # optional — enables real AI explanations

OPENAI_API_KEY=sk-...               # optional — fallback AI

LLM_PROVIDER=anthropic              # anthropic | openai | none

### Frontend `.env.local`
NEXT_PUBLIC_API_URL=http://localhost:8000

---

## Deploying

### Backend → Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway add postgresql
railway up
```

Set environment variables in Railway dashboard. Railway auto-detects the Python project and installs from `requirements.txt`.

### Frontend → Vercel

```bash
npm install -g vercel
vercel
# Follow prompts
# Set NEXT_PUBLIC_API_URL to your Railway backend URL
```

---

## Project structure---

## Deploying

### Backend → Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway add postgresql
railway up
```

Set environment variables in Railway dashboard. Railway auto-detects the Python project and installs from `requirements.txt`.

### Frontend → Vercel

```bash
npm install -g vercel
vercel
# Follow prompts
# Set NEXT_PUBLIC_API_URL to your Railway backend URL
```

---

## Project structure
ml-playground/

├── backend/

│   ├── app/

│   │   ├── core/          # DB connection, settings

│   │   ├── models/        # SQLAlchemy models

│   │   ├── routers/       # FastAPI route handlers

│   │   │   ├── sessions.py    # Pipeline steps, training, tuning

│   │   │   ├── datasets.py    # Dataset loading, profiling

│   │   │   ├── eda.py         # Distributions, correlation

│   │   │   └── automl.py      # Background AutoML jobs

│   │   ├── schemas/       # Pydantic request/response models

│   │   └── services/

│   │       ├── ml/        # preprocessing.py, training.py

│   │       ├── ai/        # explainer.py

│   │       └── export/    # code_gen.py

│   └── requirements.txt

└── frontend/

├── app/

│   ├── page.tsx           # Landing page + AutoML tab

│   └── playground/

│       └── [sessionId]/

│           └── page.tsx   # 13-step playground

└── lib/

└── api.ts             # Typed API client

## License

MIT