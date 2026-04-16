import json

from app.schemas.citation import Citation
from app.schemas.user import UserProfile
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.permission_service import PermissionService
from app.services.rerank_service import RerankService


class RetrievalService:
    def __init__(self) -> None:
        self._repo = KnowledgeRepository()
        self._permission = PermissionService()
        self._rerank = RerankService()
        self._recall_k = 16
        self._top_k = 8

    def retrieve(self, question: str, user: UserProfile) -> list[Citation]:
        where = self._permission.build_vector_filter(user)
        try:
            rows = self._repo.query_chunks(question=question, where=where, n_results=self._recall_k)
        except RuntimeError:
            return []
        rows = self._rerank.rerank(query=question, rows=rows, top_n=self._top_k)
        citations: list[Citation] = []
        for row in rows:
            metadata = row.get("metadata", {})
            citations.append(
                Citation(
                    doc_id=metadata.get("doc_id", ""),
                    title=metadata.get("title", "未命名文档"),
                    source_type=metadata.get("source_type", "unknown"),
                    content_chunk=row.get("document", ""),
                    summary=metadata.get("summary"),
                    version=metadata.get("version", "v0"),
                    is_latest=bool(metadata.get("is_latest", True)),
                    effective_date=metadata.get("effective_date"),
                    updated_at=metadata.get("updated_at"),
                    role_scope=self._normalize_scope(metadata.get("role_scope", [])),
                    department_scope=self._normalize_scope(metadata.get("department_scope", [])),
                    project_scope=self._normalize_scope(metadata.get("project_scope", [])),
                    tags=self._normalize_scope(metadata.get("tags", [])),
                )
            )
        # 召回后再做一次校验，确保无权限数据不进入答案。
        return self._permission.filter_citations(citations, user)

    @staticmethod
    def _normalize_scope(raw: object) -> list[str]:
        if isinstance(raw, list):
            return [str(item) for item in raw]
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return [text]
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            return [text]
        return []
