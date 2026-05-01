from __future__ import annotations

from app.services.embedding_service import DashScopeEmbeddingService
from app.services.langchain_support import LANGCHAIN_AVAILABLE, LangChainEmbeddings


if LANGCHAIN_AVAILABLE:

    class DashScopeLangChainEmbeddings(LangChainEmbeddings):
        def __init__(self, service: DashScopeEmbeddingService):
            self.service = service

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.service.embed_texts(texts)

        def embed_query(self, text: str) -> list[float]:
            return self.service.embed_query(text)

else:

    class DashScopeLangChainEmbeddings:  # pragma: no cover - fallback shim
        def __init__(self, service: DashScopeEmbeddingService):
            self.service = service
