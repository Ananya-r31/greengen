"""RAG Sustainability AI Assistant API routes."""
from fastapi import APIRouter, HTTPException
from app.database.db import get_dataset_record, get_latest_model_run, get_latest_optimization_run
from app.schemas.models import AssistantChatRequest
from app.services.data_loader import load_dataframe
from app.services.data_profiler import compute_dashboard_kpis
from app.services.rag_service import answer_sustainability_question

router = APIRouter(prefix="/api/datasets", tags=["RAG & AI Assistant"])


@router.post("/{dataset_id}/assistant/chat")
def assistant_chat_endpoint(dataset_id: str, req: AssistantChatRequest):
    rec = get_dataset_record(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    df = load_dataframe(rec["cleaned_file_path"] or rec["file_path"])
    mapping = rec["mapping"]
    filters = {}
    if req.sector_filter and mapping.get("concept_to_columns", {}).get("industry_sector"):
        filters[mapping["concept_to_columns"]["industry_sector"][0]] = req.sector_filter

    kpis = compute_dashboard_kpis(df, mapping, filters=filters)
    latest_model = get_latest_model_run(dataset_id)
    latest_opt = get_latest_optimization_run(dataset_id)

    return answer_sustainability_question(
        question=req.question,
        df=df,
        mapping=mapping,
        kpis=kpis,
        latest_model_run=latest_model,
        latest_opt_run=latest_opt,
        sector_filter=req.sector_filter,
        top_k=req.top_k,
    )
