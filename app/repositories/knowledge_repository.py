from datetime import date, datetime
from typing import Any

from app.core.vector_store import ChromaVectorStore
from app.repositories.metadata_repository import MetadataRepository
from app.schemas.knowledge import KnowledgeChunk


class KnowledgeRepository:
    def __init__(self) -> None:
        self._store = ChromaVectorStore()
        self._metadata_repo = MetadataRepository()

    def save_chunks(self, items: list[dict]) -> int:
        chunks = [KnowledgeChunk.model_validate(item) for item in items]
        records: list[dict[str, Any]] = []
        metadata_rows: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            role_scope = chunk.role_scope or ["*"]
            department_scope = chunk.department_scope or ["*"]
            project_scope = chunk.project_scope or ["*"]
            chunk_id = f"{chunk.doc_id}:{idx}"
            record = {
                "id": chunk_id,
                "document": chunk.content_chunk,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source_type": chunk.source_type,
                    "summary": chunk.summary,
                    "version": chunk.version,
                    "is_latest": chunk.is_latest,
                    "effective_date": self._iso_date(chunk.effective_date),
                    "updated_at": self._iso_datetime(chunk.updated_at),
                    "tags": chunk.tags or ["未分类"],
                    "role_scope": role_scope,
                    "department_scope": department_scope,
                    "project_scope": project_scope,
                },
            }
            records.append(record)
            metadata_rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source_type": chunk.source_type,
                    "content_chunk": chunk.content_chunk,
                    "summary": chunk.summary,
                    "version": chunk.version,
                    "is_latest": chunk.is_latest,
                    "effective_date": self._iso_date(chunk.effective_date),
                    "updated_at": self._iso_datetime(chunk.updated_at),
                    "role_scope": role_scope,
                    "department_scope": department_scope,
                    "project_scope": project_scope,
                    "tags": chunk.tags or ["未分类"],
                }
            )

        saved = self._store.upsert(records)
        self._metadata_repo.upsert_rows(metadata_rows)
        return saved

    def query_chunks(
        self, question: str, where: dict[str, Any] | None = None, n_results: int = 8
    ) -> list[dict[str, Any]]:
        return self._store.query(question=question, where=where, n_results=n_results)

    @staticmethod
    def _iso_date(value: date | None) -> str | None:
        return value.isoformat() if value else None

    @staticmethod
    def _iso_datetime(value: datetime | None) -> str | None:
        return value.isoformat() if value else None
