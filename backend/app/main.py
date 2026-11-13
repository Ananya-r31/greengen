"""GreenGen FastAPI Backend Entry Point."""
import os
import platform
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import (
    assistant_routes,
    dataset_routes,
    optimization_routes,
    prediction_routes,
    report_routes,
)
from app.database.db import init_db

app = FastAPI(
    title="GreenGen — Generative AI-Powered Carbon Footprint Optimization Platform",
    description="Enterprise Sustainability Decision-Support, Multi-Model ML Prediction, Constrained SLSQP Optimization, and Grounded RAG Assistant API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(dataset_routes.router)
app.include_router(prediction_routes.router)
app.include_router(optimization_routes.router)
app.include_router(assistant_routes.router)
app.include_router(report_routes.router)


@app.get("/api/system/info")
def get_system_info():
    llm_configured = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GREENGEN_LLM_API_KEY"))
    return {
        "platform": "GreenGen Sustainability Decision-Support Platform",
        "version": "1.0.0",
        "python_version": platform.python_version(),
        "supported_models": [
            "Random Forest Regressor",
            "Bayesian Ridge Regression",
            "XGBoost Regressor",
            "Gaussian Process Regression",
        ],
        "optimizer": "SciPy SLSQP (Sequential Least Squares Programming) with bound & operational constraints",
        "rag_mode": "LLM API Connected" if llm_configured else "Grounded Deterministic RAG Fallback Mode (Active)",
        "supported_report_formats": ["pdf", "csv", "xlsx"],
        "max_upload_size_mb": 25,
    }
