import os

from app.core.config import settings

try:
    from langchain_community.embeddings import DashScopeEmbeddings
except ImportError:  # pragma: no cover - 依赖未安装时的兜底
    DashScopeEmbeddings = None  # type: ignore[assignment]


class EmbeddingService:
    def __init__(self) -> None:
        self.model = settings.embedding_model
        self._client = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        return client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        return client.embed_query(text)

    def _get_client(self):
        if self._client is not None:
            return self._client
        self._client = self._build_client()
        return self._client

    def _build_client(self):
        if DashScopeEmbeddings is None:
            raise RuntimeError("未安装 LangChain DashScope 依赖，请执行 `pip install -r requirements.txt`。")

        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未检测到 DashScope API Key，请设置 `DASHSCOPE_API_KEY` 环境变量。")

        return DashScopeEmbeddings(model=self.model, dashscope_api_key=api_key)
