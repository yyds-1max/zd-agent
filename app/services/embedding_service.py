from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import requests


class EmbeddingService(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class DashScopeEmbeddingService(EmbeddingService):
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        dimensions: int,
        batch_size: int = 10,
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        return bool(self.api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for remote embeddings.")
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            payload: dict[str, Any] = {
                "model": self.model,
                "input": chunk,
                "encoding_format": "float",
                "dimensions": self.dimensions,
            }
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text.strip()
                raise requests.HTTPError(
                    f"{exc}. Response body: {detail}",
                    response=response,
                ) from exc
            data = response.json().get("data", [])
            ordered = sorted(data, key=lambda item: item["index"])
            embeddings.extend(item["embedding"] for item in ordered)
        return embeddings
