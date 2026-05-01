from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    doc_id: str
    title: str
    doc_type: str
    topic: str
    permission_level: str
    version: str | None = None
    status: str | None = None
    published_at: date | None = None
    updated_at: date | None = None
    is_latest: bool = False
    project_name: str | None = None
    source_path: str
    body: str
    sections: list[str] = Field(default_factory=list)

    def searchable_text(self) -> str:
        return "\n".join(
            [
                self.title,
                self.doc_type,
                self.topic,
                self.permission_level,
                self.body,
            ]
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "topic": self.topic,
            "permission_level": self.permission_level,
            "version": self.version or "",
            "status": self.status or "",
            "published_at": self.published_at.isoformat() if self.published_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
            "is_latest": self.is_latest,
            "project_name": self.project_name or "",
            "source_path": self.source_path,
        }


class KnowledgeChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    doc_title: str
    doc_type: str
    topic: str
    permission_level: str
    version: str | None = None
    status: str | None = None
    published_at: date | None = None
    updated_at: date | None = None
    is_latest: bool = False
    project_name: str | None = None
    source_path: str
    section_title: str | None = None
    subsection_title: str | None = None
    text: str

    def searchable_text(self) -> str:
        return "\n".join(
            [
                self.doc_title,
                self.doc_type,
                self.topic,
                self.permission_level,
                self.section_title or "",
                self.subsection_title or "",
                self.text,
            ]
        ).strip()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "title": self.doc_title,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "topic": self.topic,
            "permission_level": self.permission_level,
            "version": self.version or "",
            "status": self.status or "",
            "published_at": self.published_at.isoformat() if self.published_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
            "is_latest": self.is_latest,
            "project_name": self.project_name or "",
            "source_path": self.source_path,
            "section_title": self.section_title or "",
            "subsection_title": self.subsection_title or "",
        }


class RetrievedChunk(BaseModel):
    chunk: KnowledgeChunk
    snippet: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class Citation(BaseModel):
    chunk_id: str
    chunk_index: int
    doc_id: str
    title: str
    doc_type: str
    version: str | None = None
    permission_level: str
    published_at: date | None = None
    is_latest: bool = False
    section_title: str | None = None
    subsection_title: str | None = None
    score: float = 0.0
    snippet: str = ""
    chunk_text: str = ""
    source_path: str
