from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.vector_store import ChromaVectorStore
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.embedding_service import DashScopeEmbeddingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the Chroma vector index.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not drop the existing Chroma collection before upserting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    knowledge_repository = KnowledgeRepository(settings.fixtures_dir)
    documents = knowledge_repository.list_documents()
    chunks = knowledge_repository.list_chunks()
    embedding_service = DashScopeEmbeddingService(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_embedding_base_url,
        model=settings.dashscope_embedding_model,
        dimensions=settings.dashscope_embedding_dimensions,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_dir,
        collection_name=settings.chroma_collection_name,
        embedding_service=embedding_service,
    )

    if not vector_store.available:
        raise SystemExit(
            "Cannot rebuild Chroma index. "
            f"Backend unavailable: {vector_store.status_reason}. "
            "Make sure `chromadb` is installed and `DASHSCOPE_API_KEY` is configured."
        )

    if not args.keep_existing:
        print(f"Dropping existing Chroma collection: {settings.chroma_collection_name}")
        vector_store.reset_collection()

    print(
        "Rebuilding index with "
        f"{len(chunks)} chunks from {len(documents)} documents, "
        f"collection={settings.chroma_collection_name}, "
        f"path={settings.vector_store_dir}"
    )
    vector_store.sync_chunks(chunks)

    if not vector_store.available:
        raise SystemExit(
            "Index rebuild failed during sync. "
            f"Backend status: {vector_store.status_reason}. "
            f"Detail: {vector_store.last_error_detail or 'unknown error'}"
        )

    print(f"Rebuild complete. Indexed documents: {vector_store.count()}")


if __name__ == "__main__":
    main()
