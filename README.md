# GreenGen — Generative AI-Powered Carbon Footprint Optimization Platform

GreenGen is an end-to-end sustainability decision-support platform that ingests external CSV or Excel datasets, automatically profiles environmental and operational variables, trains and benchmarks four regression algorithms, runs constrained numerical carbon-reduction optimization (SciPy SLSQP), answers questions via a grounded Retrieval-Augmented Generation (RAG) assistant, and exports structured sustainability reports.

## 1. Architecture Overview

```text
greengen/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI application & CORS configuration
│   │   ├── api/
│   │   │   ├── dataset_routes.py        # Upload, profile, preview, preprocess & cleaned CSV download
│   │   │   ├── prediction_routes.py     # 4-model training, evaluation, prediction & time-series forecasting
│   │   │   ├── optimization_routes.py   # SciPy SLSQP constrained carbon optimization & what-if scenarios
│   │   │   ├── assistant_routes.py      # RAG vector retrieval + grounded AI sustainability assistant
│   │   │   └── report_routes.py         # Evidence-based recommendations, editable action plan & PDF/CSV/XLSX reports
│   │   ├── services/
│   │   │   ├── data_loader.py           # File size/extension validation & safe CSV/Excel parsing
│   │   │   ├── feature_mapping.py       # Semantic detection of 14 environmental & operational concepts
│   │   │   ├── data_profiler.py         # Missing values, IQR outliers, summary statistics & dynamic KPIs
│   │   │   ├── preprocessing.py         # Imputation, deduplication, outlier clipping & emission-factor calculation
│   │   │   ├── model_training.py        # Random Forest, Bayesian Ridge, XGBoost & Gaussian Process + zero-leakage pipeline
│   │   │   ├── model_evaluation.py      # Held-out MAE, RMSE, R², and conditional MAPE (guarded near zero)
│   │   │   ├── forecasting.py           # Chronological time-series forecasting with √h uncertainty scaling
│   │   │   ├── optimization.py          # SciPy SLSQP optimizer with bounds, budget & production constraints
│   │   │   ├── rag_service.py           # Chunked knowledge base vector search + LLM/deterministic fallback
│   │   │   ├── recommendation_service.py# Data-supported recommendations & 4-horizon action plan generator
│   │   │   └── report_service.py        # 14-section PDF, CSV, and Excel (.xlsx) sustainability report generator
│   │   ├── schemas/models.py            # Pydantic request/response validation schemas
│   │   ├── database/db.py               # SQLite persistence for datasets, model runs, optimizations & plans
│   │   └── knowledge_base/
│   │       └── sustainability_docs.json # Verified GHG Protocol Scopes 1-3, IPCC EFs, ISO 50001, SBTi, CSRD/GRI, UN SDGs
│   ├── tests/
│   │   ├── synthetic_dev_dataset.csv    # Clearly labeled synthetic dataset for development & testing only
│   │   └── test_greengen.py             # Automated test suite covering all 15 validation requirements
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx                      # Complete 8-module React + TypeScript sustainability analytics interface
│   │   ├── services/api.ts              # Axios REST API client
│   │   └── types/index.ts               # TypeScript domain models
│   ├── package.json
│   └── vite.config.ts
├── README.md
└── .gitignore
```

## 2. Setup and Exact Terminal Commands

### Backend Setup (Python 3.11+ & FastAPI)

```bash
cd greengen/backend

# Optional: Create and activate a virtual environment (or use your existing Python 3.11 environment)
# python3 -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt

# Copy environment configuration template
cp .env.example .env

# Run the automated backend test suite (15 test criteria across 12 test methods)
python3 tests/test_greengen.py

# Start the FastAPI server on http://127.0.0.1:8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### Frontend Setup (React + Vite + TypeScript + Tailwind CSS)

```bash
cd greengen/frontend

# Install frontend dependencies (when running in an online environment)
npm install

# Start the Vite development server on http://127.0.0.1:5173
npm run dev

# Build for production
npm run build
```

## 3. Environment Configuration & LLM Fallback Mode

GreenGen keeps all credentials in environment variables (`backend/.env`):
- `GREENGEN_DB_PATH`: Path to the SQLite database file (defaults to `./greengen.db`).
- `OPENAI_API_KEY` or `GREENGEN_LLM_API_KEY`: Optional API key for LLM narrative synthesis.
- **Grounded Fallback Mode**: When no external LLM API key is configured, the RAG Assistant automatically operates in **Grounded Deterministic RAG Fallback Mode**, combining cosine-similarity passage retrieval from `sustainability_docs.json` with exact Pearson correlations, KPIs, model metrics, and optimization results from the currently active dataset. It explicitly labels this mode in the UI and API response.

## 4. REST API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/datasets` | List all uploaded datasets and synthetic status flags |
| `POST` | `/api/datasets/upload` | Upload an external `.csv` or `.xlsx` dataset (up to 25 MB) |
| `GET` | `/api/datasets/{dataset_id}/profile` | Retrieve data quality profile, semantic column mapping, and filtered KPIs |
| `GET` | `/api/datasets/{dataset_id}/preview` | Preview first $N$ rows of the dataset |
| `POST` | `/api/datasets/{dataset_id}/preprocess` | Impute missing values, drop duplicates, clip outliers, or apply emission factors |
| `POST` | `/api/datasets/{dataset_id}/train` | Train Random Forest, Bayesian Ridge, XGBoost, and Gaussian Process models |
| `GET` | `/api/datasets/{dataset_id}/models` | Retrieve held-out evaluation metrics (`MAE`, `RMSE`, `R²`, `MAPE`) and feature importances |
| `POST` | `/api/datasets/{dataset_id}/predict` | Generate single-point prediction with 95% confidence interval and OOD checks |
| `POST` | `/api/datasets/{dataset_id}/forecast` | Generate multi-period chronological forecast (when temporal data exists) |
| `POST` | `/api/datasets/{dataset_id}/optimize` | Run constrained SciPy SLSQP carbon-reduction optimization |
| `POST` | `/api/datasets/{dataset_id}/scenarios` | Compare interactive what-if decarbonization scenarios |
| `POST` | `/api/datasets/{dataset_id}/assistant/chat` | Query the RAG sustainability assistant with cited knowledge base sources |
| `GET` | `/api/datasets/{dataset_id}/recommendations` | Generate dataset-grounded recommendations and 4-horizon action plan |
| `POST` | `/api/datasets/{dataset_id}/action-plan` | Save edits to the decarbonization action plan |
| `POST` | `/api/datasets/{dataset_id}/reports` | Generate and download 14-section sustainability report (`pdf`, `csv`, or `xlsx`) |

## 5. Methodological Safeguards and Limitations

1. **Zero Fabrication**: No KPIs, model metrics, emission reductions, or financial savings are ever invented. Before a dataset is uploaded, the dashboard displays an empty state prompting the user to upload their data or load a clearly labeled synthetic demo dataset.
2. **Zero Data Leakage**: Numerical scaling, median/mean imputation, and categorical encoding are fit strictly on the training split. Chronological splitting is supported for time-series datasets.
3. **Conditional MAPE**: Mean Absolute Percentage Error (MAPE) is automatically omitted with an explanation whenever held-out target values contain zero or near-zero ($<0.001$) numbers.
4. **Constrained Local Optimization**: SciPy SLSQP solves for a local constrained minimum within user-specified variable bounds and training distributions; engineering feasibility must always be verified on site.
5. **Non-Certified Reporting**: Generated PDF, CSV, and Excel reports are analytical decision-support deliverables and are explicitly labeled as non-certified (not third-party audited under ISO 14064 or CSRD assurance).
