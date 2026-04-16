import os
import re
from http import HTTPStatus
from typing import Any

from app.core.config import settings

try:
    from dashscope import TextReRank
except ImportError:  # pragma: no cover - 依赖缺失时走规则重排
    TextReRank = None  # type: ignore[assignment]


class RerankService:
    def __init__(self, enable_model_rerank: bool | None = None) -> None:
        if enable_model_rerank is None:
            self.enable_model_rerank = settings.enable_model_rerank
        else:
            self.enable_model_rerank = enable_model_rerank
        self.model_name = settings.rerank_model

    def rerank(self, query: str, rows: list[dict[str, Any]], top_n: int = 8) -> list[dict[str, Any]]:
        if not rows:
            return []

        model_scores = self._model_scores(query, rows)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            metadata = row.get("metadata", {})
            relevance = model_scores.get(idx, self._distance_to_relevance(row.get("distance")))
            latest = 1.0 if bool(metadata.get("is_latest", False)) else 0.0
            confirm = self._confirmability_score(row)
            score = 0.70 * relevance + 0.20 * latest + 0.10 * confirm
            ranked.append((score, row))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [row for _, row in ranked[:top_n]]

    def _model_scores(self, query: str, rows: list[dict[str, Any]]) -> dict[int, float]:
        if not self.enable_model_rerank:
            return {}
        if TextReRank is None:
            return {}

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return {}

        documents = [str(row.get("document", "")) for row in rows]
        try:
            response = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=len(documents),
                api_key=api_key,
            )
        except Exception:
            return {}

        if response.status_code != HTTPStatus.OK or not response.output:
            return {}

        scores: dict[int, float] = {}
        for result in response.output.results or []:
            index = int(result.get("index", -1))
            relevance_score = float(result.get("relevance_score", 0.0))
            if 0 <= index < len(rows):
                scores[index] = max(0.0, min(1.0, relevance_score))
        return scores

    @staticmethod
    def _distance_to_relevance(distance: Any) -> float:
        if distance is None:
            return 0.5
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.5
        return 1.0 / (1.0 + max(0.0, value))

    @staticmethod
    def _confirmability_score(row: dict[str, Any]) -> float:
        text = str(row.get("document", "")).strip()
        metadata = row.get("metadata", {})
        score = 0.0

        # 包含明确数字/日期/金额等，通常更可证实。
        if re.search(r"\d", text):
            score += 0.35

        # 包含制度性约束词，表示可以被文档直接核对。
        if any(token in text for token in ("必须", "需", "应", "不得", "标准", "生效", "以", "为准")):
            score += 0.35

        # 元数据完整时提高置信。
        if metadata.get("version"):
            score += 0.15
        if metadata.get("effective_date"):
            score += 0.15

        return min(1.0, score)
