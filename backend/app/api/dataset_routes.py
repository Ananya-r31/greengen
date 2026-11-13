"""Dataset Management & Overview KPI API routes."""
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from app.database.db import (
    get_dataset_record,
    get_latest_model_run,
    get_latest_optimization_run,
    list_dataset_records,
    save_dataset_record,
    update_dataset_cleaned,
)
from app.schemas.models import PreprocessRequest
from app.services.data_loader import DatasetLoadError, load_dataframe, validate_and_save_upload
from app.services.data_profiler import compute_dashboard_kpis, profile_dataset
from app.services.feature_mapping import detect_column_concepts
from app.services.preprocessing import preprocess_dataset

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


@router.get("")
def list_datasets():
    records = list_dataset_records()
    return {
        "count": len(records),
        "datasets": [
            {
                "dataset_id": r["dataset_id"],
                "filename": r["filename"],
                "row_count": r["row_count"],
                "column_count": r["column_count"],
                "is_synthetic": r["is_synthetic"],
                "primary_target": r["mapping"].get("primary_target"),
                "created_at": r["created_at"],
            }
            for r in records
        ],
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    is_synthetic: bool = Query(default=False, description="Flag if dataset is synthetic dev data"),
):
    try:
        content = await file.read()
        dataset_id, saved_path, df = validate_and_save_upload(file.filename or "dataset.csv", content)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Automatically flag synthetic if filename or organization column contains SYNTHETIC
    syn_flag = bool(is_synthetic or "synthetic" in (file.filename or "").lower())
    mapping = detect_column_concepts(df)
    profile = profile_dataset(df, mapping)
    save_dataset_record(
        dataset_id=dataset_id,
        filename=file.filename or "dataset.csv",
        file_path=saved_path,
        row_count=len(df),
        column_count=len(df.columns),
        profile=profile,
        mapping=mapping,
        is_synthetic=syn_flag,
    )
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "is_synthetic": syn_flag,
        "row_count": len(df),
        "column_count": len(df.columns),
        "profile": profile,
        "mapping": mapping,
    }


@router.get("/{dataset_id}/profile")
def get_profile(
    dataset_id: str,
    year: Optional[str] = None,
    country: Optional[str] = None,
    sector: Optional[str] = None,
    organization: Optional[str] = None,
):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    mapping = rec["mapping"]

    # Map filter query params to actual detected columns
    c2c = mapping.get("concept_to_columns", {})
    filters = {}
    if year and c2c.get("temporal"):
        filters[c2c["temporal"][0]] = year
    if country and c2c.get("country_region"):
        filters[c2c["country_region"][0]] = country
    if sector and c2c.get("industry_sector"):
        filters[c2c["industry_sector"][0]] = sector
    if organization and c2c.get("organization"):
        filters[c2c["organization"][0]] = organization

    kpis = compute_dashboard_kpis(df, mapping, filters=filters)
    latest_model = get_latest_model_run(dataset_id)
    latest_opt = get_latest_optimization_run(dataset_id)

    return {
        "dataset_id": dataset_id,
        "filename": rec["filename"],
        "is_synthetic": rec["is_synthetic"],
        "profile": rec["profile"],
        "mapping": mapping,
        "preprocessing_log": rec["preprocessing_log"],
        "kpis": kpis,
        "latest_model_summary": (
            {
                "run_id": latest_model["run_id"],
                "best_model_name": latest_model["best_model_name"],
                "target_column": latest_model["target_column"],
                "models_summary": latest_model["models_summary"],
            }
            if latest_model
            else None
        ),
        "latest_optimization_summary": (
            {
                "opt_id": latest_opt["opt_id"],
                "model_name": latest_opt["model_name"],
                "baseline_emissions": latest_opt["baseline_emissions"],
                "optimized_emissions": latest_opt["optimized_emissions"],
                "absolute_reduction": latest_opt["absolute_reduction"],
                "percentage_reduction": latest_opt["percentage_reduction"],
                "status": latest_opt["status"],
            }
            if latest_opt
            else None
        ),
    }


@router.get("/{dataset_id}/preview")
def get_preview(dataset_id: str, limit: int = Query(default=25, ge=1, le=200)):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    preview_rows = df.head(limit).where(df.notna(), None).to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "filename": rec["filename"],
        "total_rows": len(df),
        "columns": list(df.columns),
        "rows": preview_rows,
    }


@router.post("/{dataset_id}/preprocess")
def preprocess_endpoint(dataset_id: str, req: PreprocessRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    raw_df = load_dataframe(rec["file_path"])
    clean_df, cleaned_path, profile, mapping, logs = preprocess_dataset(
        dataset_id=dataset_id,
        df=raw_df,
        missing_numeric_strategy=req.missing_numeric_strategy,
        missing_categorical_strategy=req.missing_categorical_strategy,
        drop_duplicates=req.drop_duplicates,
        outlier_method=req.outlier_method,
        type_overrides=req.type_overrides,
        concept_overrides=req.concept_overrides,
        emission_factors=req.emission_factors,
    )
    update_dataset_cleaned(
        dataset_id=dataset_id,
        cleaned_file_path=cleaned_path,
        row_count=len(clean_df),
        column_count=len(clean_df.columns),
        profile=profile,
        mapping=mapping,
        preprocessing_log=logs,
    )
    return {
        "dataset_id": dataset_id,
        "row_count": len(clean_df),
        "column_count": len(clean_df.columns),
        "preprocessing_log": logs,
        "profile": profile,
        "mapping": mapping,
    }


@router.get("/{dataset_id}/download-cleaned")
def download_cleaned_dataset(dataset_id: str):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    path = rec["cleaned_file_path"] or rec["file_path"]
    return FileResponse(path, media_type="text/csv", filename=f"cleaned_{rec['filename']}")
