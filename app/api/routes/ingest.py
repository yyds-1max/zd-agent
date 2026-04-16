from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines.ingest_pipeline import IngestPipeline

router = APIRouter()
pipeline = IngestPipeline()


class IngestRequest(BaseModel):
    source_dir: str


@router.post("")
def ingest(payload: IngestRequest) -> dict[str, str]:
    return pipeline.run(payload.source_dir)
