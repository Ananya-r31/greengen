"""ML Prediction, Model Comparison, and Forecasting API routes."""
from fastapi import APIRouter, HTTPException
from app.database.db import get_dataset_record, get_latest_model_run, list_model_runs, save_model_run
from app.schemas.models import ForecastRequest, PredictRequest, TrainRequest
from app.services.data_loader import load_dataframe
from app.services.forecasting import generate_emissions_forecast
from app.services.model_training import load_model_bundle, predict_single, train_and_evaluate_models

router = APIRouter(prefix="/api/datasets", tags=["ML Prediction & Forecasting"])


@router.post("/{dataset_id}/train")
def train_models_endpoint(dataset_id: str, req: TrainRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    try:
        res = train_and_evaluate_models(
            dataset_id=dataset_id,
            df=df,
            target_column=req.target_column,
            target_unit=req.target_unit,
            feature_columns=req.feature_columns,
            test_size=req.test_size,
            split_strategy=req.split_strategy,
            time_column=req.time_column,
            gpr_max_samples=req.gpr_max_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    save_model_run(
        run_id=res["run_id"],
        dataset_id=dataset_id,
        target_column=res["target_column"],
        target_unit=res["target_unit"],
        feature_columns=res["feature_columns"],
        split_strategy=res["split_strategy"],
        test_size=res["test_size"],
        models_summary=res["models"],
        best_model_name=res["best_model_name"],
        selection_criterion=res["selection_criterion"],
        artifact_path=res["artifact_path"],
    )
    return res


@router.get("/{dataset_id}/models")
def get_models_endpoint(dataset_id: str):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    runs = list_model_runs(dataset_id)
    return {
        "dataset_id": dataset_id,
        "runs_count": len(runs),
        "latest_run": runs[0] if runs else None,
        "all_runs": runs,
    }


@router.post("/{dataset_id}/predict")
def predict_endpoint(dataset_id: str, req: PredictRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    model_run = get_latest_model_run(dataset_id, target_column=req.target_column)
    if not model_run:
        raise HTTPException(
            status_code=400,
            detail="No trained models exist for this dataset/target. Please train models first via POST /api/datasets/{dataset_id}/train.",
        )
    bundle = load_model_bundle(model_run["artifact_path"])
    return predict_single(
        bundle=bundle,
        models_summary=model_run["models_summary"],
        input_values=req.input_values,
        model_key=req.model_name,
    )


@router.post("/{dataset_id}/forecast")
def forecast_endpoint(dataset_id: str, req: ForecastRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    model_run = get_latest_model_run(dataset_id, target_column=req.target_column)
    if not model_run:
        raise HTTPException(
            status_code=400,
            detail="No trained models exist for this dataset. Train models first before forecasting.",
        )
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    bundle = load_model_bundle(model_run["artifact_path"])
    try:
        return generate_emissions_forecast(
            df=df,
            mapping=rec["mapping"],
            bundle=bundle,
            models_summary=model_run["models_summary"],
            horizon_periods=req.horizon_periods,
            time_column=req.time_column,
            model_key=req.model_name,
            feature_growth_rates=req.feature_growth_rates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
