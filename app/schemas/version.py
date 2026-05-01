from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class VersionCheckResult(BaseModel):
    source_chunk_id: str | None = None
    source_section_title: str | None = None
    source_subsection_title: str | None = None
    source_doc_id: str
    source_title: str
    source_version: str | None = None
    source_is_latest: bool = False
    has_newer_version: bool = False
    latest_doc_id: str | None = None
    latest_title: str | None = None
    latest_version: str | None = None
    latest_published_at: date | None = None
    latest_accessible: bool = True
    latest_chunk_id: str | None = None
    latest_section_title: str | None = None
    latest_subsection_title: str | None = None
    latest_chunk_text: str | None = None
    latest_chunk_match_confidence: float = 0.0
    notice: str = ""


class VersionDiffResult(BaseModel):
    source_chunk_id: str
    source_doc_id: str
    source_version: str | None = None
    latest_chunk_id: str | None = None
    latest_doc_id: str | None = None
    latest_version: str | None = None
    change_type: str
    summary: str
    confidence: float = 0.0
    key_changes: list[str] = Field(default_factory=list)
