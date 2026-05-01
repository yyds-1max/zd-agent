from __future__ import annotations

from pathlib import Path

from app.schemas.knowledge import KnowledgeChunk, KnowledgeDocument
from app.services.langchain_embedding_adapter import DashScopeLangChainEmbeddings
from app.services.langchain_support import (
    LANGCHAIN_AVAILABLE,
    LANGCHAIN_IMPORT_ERROR,
    LangChainChroma,
)

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_service,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.available = chromadb is not None and embedding_service.is_available()
        self._status_reason = "ready"
        self._last_error_detail = ""
        self._client = None
        self._collection = None
        self._vector_store = None

        if chromadb is None:
            self._status_reason = "chromadb_not_installed"
            return
        if not embedding_service.is_available():
            self._status_reason = "embedding_service_unavailable"
            return
        if not LANGCHAIN_AVAILABLE:
            self._status_reason = (
                f"langchain_not_installed: {LANGCHAIN_IMPORT_ERROR or 'unknown import error'}"
            )
            self.available = False
            return

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._vector_store = LangChainChroma(
            collection_name=self.collection_name,
            embedding_function=DashScopeLangChainEmbeddings(self.embedding_service),
            persist_directory=str(self.persist_directory),
            client=self._client,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._collection = self._vector_store._collection

    @property
    def status_reason(self) -> str:
        return self._status_reason

    @property
    def last_error_detail(self) -> str:
        return self._last_error_detail

    def describe(self) -> str:
        if self.available:
            return f"chroma:{self.collection_name}"
        return f"fallback:{self._status_reason}"

    def reset_collection(self) -> None:
        if chromadb is None or self._client is None:
            raise RuntimeError(f"Chroma unavailable: {self._status_reason}")
        try:
            if self._vector_store is not None:
                self._vector_store.delete_collection()
            else:
                self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._vector_store = LangChainChroma(
            collection_name=self.collection_name,
            embedding_function=DashScopeLangChainEmbeddings(self.embedding_service),
            persist_directory=str(self.persist_directory),
            client=self._client,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._collection = self._vector_store._collection

    def count(self) -> int:
        if not self.available or self._vector_store is None:
            return 0
        try:
            return int(self._vector_store._collection.count())
        except Exception:
            return 0

    def sync_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        if not self.available or not chunks:
            return
        try:
            self._vector_store.add_documents(
                documents=[self._to_langchain_chunk(chunk) for chunk in chunks],
                ids=[chunk.chunk_id for chunk in chunks],
            )
        except Exception as exc:
            self.available = False
            self._status_reason = "sync_failed"
            self._last_error_detail = f"{type(exc).__name__}: {exc}"

    def sync_documents(self, documents: list[KnowledgeDocument]) -> None:
        chunks = [self._document_as_chunk(document) for document in documents]
        self.sync_chunks(chunks)

    def search(
        self,
        query_text: str,
        allowed_doc_ids: list[str],
        n_results: int,
    ) -> dict[str, float]:
        chunk_scores = self.search_chunk_scores(
            query_text=query_text,
            allowed_doc_ids=allowed_doc_ids,
            n_results=n_results,
        )
        aggregated: dict[str, float] = {}
        for chunk_id, score in chunk_scores.items():
            doc_id = chunk_id.split("::chunk::", 1)[0]
            aggregated[doc_id] = max(aggregated.get(doc_id, float("-inf")), score)
        return aggregated

    def search_chunk_scores(
        self,
        query_text: str,
        allowed_doc_ids: list[str],
        n_results: int,
    ) -> dict[str, float]:
        if not self.available or not allowed_doc_ids:
            return {}
        try:
            results = self._vector_store.similarity_search_with_relevance_scores(
                query=query_text,
                k=min(n_results, len(allowed_doc_ids)),
                filter={"doc_id": {"$in": allowed_doc_ids}},
            )
            return {
                document.metadata.get("chunk_id", document.metadata["doc_id"]): float(score)
                for document, score in results
            }
        except Exception as exc:
            self.available = False
            self._status_reason = "query_failed"
            self._last_error_detail = f"{type(exc).__name__}: {exc}"
            return {}

    def _to_langchain_document(self, document: KnowledgeDocument):
        return self._langchain_document_class()(
            page_content=document.searchable_text(),
            metadata=document.to_metadata(),
        )

    def _to_langchain_chunk(self, chunk: KnowledgeChunk):
        return self._langchain_document_class()(
            page_content=chunk.searchable_text(),
            metadata=chunk.to_metadata(),
        )

    def _document_as_chunk(self, document: KnowledgeDocument) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=f"{document.doc_id}::chunk::000",
            doc_id=document.doc_id,
            chunk_index=0,
            doc_title=document.title,
            doc_type=document.doc_type,
            topic=document.topic,
            permission_level=document.permission_level,
            version=document.version,
            status=document.status,
            published_at=document.published_at,
            updated_at=document.updated_at,
            is_latest=document.is_latest,
            project_name=document.project_name,
            source_path=document.source_path,
            section_title=None,
            subsection_title=None,
            text=document.body,
        )

    def _langchain_document_class(self):
        from app.services.langchain_support import LangChainDocument

        return LangChainDocument
