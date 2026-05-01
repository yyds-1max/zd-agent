from __future__ import annotations

LANGCHAIN_IMPORT_ERROR: str | None = None

try:
    from langchain_chroma import Chroma as LangChainChroma
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
    from langchain_core.runnables import RunnableLambda, RunnableParallel

    LANGCHAIN_AVAILABLE = True
except ModuleNotFoundError as exc:
    LangChainChroma = None
    BM25Retriever = None
    LangChainDocument = None
    LangChainEmbeddings = object
    RunnableLambda = None
    RunnableParallel = None
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_IMPORT_ERROR = str(exc)
