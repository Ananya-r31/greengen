"""Carbon Optimization and What-If Scenario API routes."""
import uuid
from fastapi import APIRouter, HTTPException
from app.database.db import get_dataset_record, get_latest_model_run, save_optimization_run
from app.schemas.models import OptimizeRequest, ScenariosRequest
from app.services.data_loader import load_dataframe
from app.services.model_training import load_model_bundle
from app.services.optimization import run_carbon_optimization, run_what_if_scenarios

router = APIRouter(prefix="/api/datasets", tags=["Carbon Optimization & Scenarios"])


@router.post("/{dataset_id}/optimize")
def optimize_endpoint(dataset_id: str, req: OptimizeRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    model_run = get_latest_model_run(dataset_id, target_column=req.target_column)
    if not model_run:
        raise HTTPException(
            status_code=400,
            detail="No trained models exist for this dataset. Train models before running optimization.",
        )
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    bundle = load_model_bundle(model_run["artifact_path"])
    try:
        res = run_carbon_optimization(
            df=df,
            mapping=rec["mapping"],
            bundle=bundle,
            models_summary=model_run["models_summary"],
            target_reduction_pct=req.target_reduction_pct,
            max_budget=req.max_budget,
            baseline_row_index=req.baseline_row_index,
            baseline_inputs=req.baseline_inputs,
            variable_bounds=req.variable_bounds,
            preserve_production_column=req.preserve_production_column,
            min_production_ratio=req.min_production_ratio,
            model_key=req.model_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    opt_id = f"opt_{uuid.uuid4().hex[:10]}"
    res["opt_id"] = opt_id
    save_optimization_run(
        opt_id=opt_id,
        dataset_id=dataset_id,
        run_id=model_run["run_id"],
        model_name=res["model_name"],
        target_reduction_pct=req.target_reduction_pct,
        baseline_emissions=res["baseline_emissions"],
        optimized_emissions=res["optimized_emissions"],
        absolute_reduction=res["absolute_reduction"],
        percentage_reduction=res["percentage_reduction"],
        status=res["status"],
        result_data=res,
    )
    return res


@router.post("/{dataset_id}/scenarios")
def scenarios_endpoint(dataset_id: str, req: ScenariosRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    model_run = get_latest_model_run(dataset_id, target_column=req.target_column)
    if not model_run:
        raise HTTPException(
            status_code=400,
            detail="No trained models exist for this dataset. Train models before running scenarios.",
        )
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    bundle = load_model_bundle(model_run["artifact_path"])
    scenarios_dicts = [
        {"name": s.name, "description": s.description, "adjustments_pct": s.adjustments_pct}
        for s in req.scenarios
    ]
    return run_what_if_scenarios(
        bundle=bundle,
        models_summary=model_run["models_summary"],
        df=df,
        scenarios=scenarios_dicts,
        baseline_inputs=req.baseline_inputs,
        model_key=req.model_name,
    )
