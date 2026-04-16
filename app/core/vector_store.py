from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.embedding_service import EmbeddingService

try:
    import chromadb
except ImportError:  # pragma: no cover - 依赖未安装时提示
    chromadb = None  # type: ignore[assignment]


class ChromaVectorStore:
    def __init__(self) -> None:
        self._embedding_service = EmbeddingService()
        self._collection = None

    def upsert(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        collection = self._get_collection()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for record in records:
            record_id = str(record["id"])
            document = str(record["document"])
            metadata = self._sanitize_metadata(record.get("metadata", {}))
            ids.append(record_id)
            documents.append(document)
            metadatas.append(metadata)

        embeddings = self._embedding_service.embed_texts(documents)
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(ids)

    def query(self, *, question: str, where: dict[str, Any] | None = None, n_results: int = 5) -> list[dict[str, Any]]:
        collection = self._get_collection()
        query_embedding = self._embedding_service.embed_query(question)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        rows: list[dict[str, Any]] = []
        for idx, row_id in enumerate(ids):
            rows.append(
                {
                    "id": row_id,
                    "document": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return rows

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if chromadb is None:
            raise RuntimeError("未安装 chromadb，请先执行 `pip install -r requirements.txt`。")

        Path(settings.vector_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.vector_dir)
        self._collection = client.get_or_create_collection(name=settings.chroma_collection)
        return self._collection

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, list):
                if not value:
                    continue
                sanitized[key] = value
                continue
            sanitized[key] = value
        return sanitized
