from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import requests

from app.schemas.knowledge import KnowledgeChunk, KnowledgeDocument


class RerankService(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[KnowledgeDocument | KnowledgeChunk],
    ) -> dict[str, float]: ...


class DashScopeRerankService(RerankService):
    def __init__(
        self,
        api_key: str | None,
        rerank_url: str,
        model: str,
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key
        self.rerank_url = rerank_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        return bool(self.api_key)

    def rerank(
        self,
        query: str,
        documents: list[KnowledgeDocument | KnowledgeChunk],
    ) -> dict[str, float]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for remote rerank.")
        if not documents:
            return {}

        payload: dict[str, Any] = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": [document.searchable_text() for document in documents],
            },
            "parameters": {
                "return_documents": False,
            },
        }
        response = requests.post(
            self.rerank_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        results = response.json().get("output", {}).get("results", [])
        return {
            self._item_id(documents[item["index"]]): float(item["relevance_score"])
            for item in results
        }

    def _item_id(self, item: KnowledgeDocument | KnowledgeChunk) -> str:
        if isinstance(item, KnowledgeChunk):
            return item.chunk_id
        return item.doc_id
