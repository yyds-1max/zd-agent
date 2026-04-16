from datetime import date, datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    doc_id: str
    title: str
    source_type: str
    content_chunk: str = Field(validation_alias=AliasChoices("content_chunk", "chunk_text"))
    summary: str | None = None
    version: str
    is_latest: bool = True
    effective_date: date | None = None
    updated_at: datetime | None = None
    role_scope: list[str] = Field(default_factory=list)
    department_scope: list[str] = Field(default_factory=list)
    project_scope: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @property
    def chunk_text(self) -> str:
        """兼容旧字段名，内部统一使用 content_chunk。"""
        return self.content_chunk
