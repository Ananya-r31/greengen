"""Recommendations, Editable Action Plan, and Sustainability Report Download API routes."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.database.db import (
    get_action_plan,
    get_dataset_record,
    get_latest_model_run,
    get_latest_optimization_run,
    save_action_plan,
)
from app.schemas.models import ActionPlanRequest, ReportRequest
from app.services.data_loader import load_dataframe
from app.services.data_profiler import compute_dashboard_kpis
from app.services.recommendation_service import generate_default_action_plan, generate_recommendations
from app.services.report_service import generate_sustainability_report

router = APIRouter(prefix="/api/datasets", tags=["Recommendations, Action Plan & Reports"])


@router.get("/{dataset_id}/recommendations")
def get_recommendations_endpoint(dataset_id: str):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    kpis = compute_dashboard_kpis(df, rec["mapping"])
    latest_model = get_latest_model_run(dataset_id)
    latest_opt = get_latest_optimization_run(dataset_id)

    recs = generate_recommendations(
        df=df,
        profile=rec["profile"],
        mapping=rec["mapping"],
        kpis=kpis,
        latest_model_run=latest_model,
        latest_opt_run=latest_opt,
    )
    saved_plan = get_action_plan(dataset_id)
    if saved_plan:
        action_items = saved_plan["items"]
    else:
        action_items = generate_default_action_plan(df, rec["mapping"], kpis, latest_opt)
        save_action_plan(dataset_id, action_items)

    return {
        "dataset_id": dataset_id,
        "recommendations": recs["recommendations"],
        "action_plan": action_items,
    }


@router.post("/{dataset_id}/action-plan")
def update_action_plan_endpoint(dataset_id: str, req: ActionPlanRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    items_dicts = [item.dict() for item in req.items]
    saved = save_action_plan(dataset_id, items_dicts)
    return saved


@router.post("/{dataset_id}/reports")
def generate_report_endpoint(dataset_id: str, req: ReportRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    kpis = compute_dashboard_kpis(df, rec["mapping"])
    latest_model = get_latest_model_run(dataset_id)
    latest_opt = get_latest_optimization_run(dataset_id)
    recs_out = generate_recommendations(df, rec["profile"], rec["mapping"], kpis, latest_model, latest_opt)
    saved_plan = get_action_plan(dataset_id)
    action_items = saved_plan["items"] if saved_plan else generate_default_action_plan(df, rec["mapping"], kpis, latest_opt)

    report_path, mime_type = generate_sustainability_report(
        dataset_record=rec,
        df=df,
        profile=rec["profile"],
        mapping=rec["mapping"],
        kpis=kpis,
        latest_model_run=latest_model,
        latest_opt_run=latest_opt,
        recommendations=recs_out["recommendations"],
        action_plan=action_items,
        report_format=req.format,
        organization_name=req.organization_name or "Uploaded Dataset Organization",
    )
    ext = req.format.lower().strip()
    return FileResponse(
        report_path,
        media_type=mime_type,
        filename=f"GreenGen_Sustainability_Report_{dataset_id}.{ext}",
    )
