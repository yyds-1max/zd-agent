from datetime import date, datetime

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    doc_id: str
    title: str
    source_type: str
    content_chunk: str
    summary: str | None = None
    department_scope: list[str] = Field(default_factory=list)
    role_scope: list[str] = Field(default_factory=list)
    project_scope: list[str] = Field(default_factory=list)
    version: str
    is_latest: bool
    effective_date: date | None = None
    updated_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
