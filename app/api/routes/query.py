from fastapi import APIRouter

from app.pipelines.query_pipeline import QueryPipeline
from app.repositories.query_log_repository import QueryLogRepository
from app.schemas.query import QueryClickRequest, QueryRequest, QueryResponse

router = APIRouter()
pipeline = QueryPipeline()
query_log_repo = QueryLogRepository()


@router.post("", response_model=QueryResponse)
def ask(payload: QueryRequest) -> QueryResponse:
    return pipeline.run(payload)


@router.post("/click")
def save_click(payload: QueryClickRequest) -> dict[str, str]:
    click_id = query_log_repo.save_click_log(
        {
            "query_id": payload.query_id,
            "doc_id": payload.doc_id,
            "title": payload.title,
            "position": payload.position,
            "action": payload.action,
            "user_id": payload.user_id,
        }
    )
    return {"status": "ok", "click_id": str(click_id)}
