import json
import os
import re
from typing import Any

from app.core.config import settings
from app.schemas.intent import QueryIntent
from app.schemas.user import UserProfile

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 依赖缺失时走规则兜底
    OpenAI = None  # type: ignore[assignment]


class IntentParserService:
    SUPPORTED_INTENTS = {"policy", "faq", "project", "chat_summary", "recommendation", "general"}

    def __init__(self) -> None:
        self._client = None

    def parse(self, question: str, user: UserProfile) -> QueryIntent:
        llm_intent = self._parse_with_llm(question, user)
        if llm_intent is not None:
            return llm_intent
        return self._parse_with_rules(question)

    def _parse_with_llm(self, question: str, user: UserProfile) -> QueryIntent | None:
        client = self._get_client()
        if client is None:
            return None

        prompt = (
            "你是企业知识问答系统的意图解析器。"
            "请仅输出 JSON，字段包括：intent_type, retrieval_query, keywords, need_latest。"
            "intent_type 仅允许 policy/faq/project/chat_summary/recommendation/general。"
        )
        user_input = {
            "question": question,
            "user_profile": {
                "user_id": user.user_id,
                "role": user.role,
                "department": user.department,
                "projects": user.projects,
            },
        }
        try:
            model_name = settings.chat_model.strip() or "qwen3-max"
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return self._normalize_llm_payload(question, payload)
        except Exception:
            return None

    def _get_client(self):
        if OpenAI is None:
            return None
        if self._client is not None:
            return self._client

        base_url = os.getenv("OPENAI_BASE_URL")
        api_key: str | None = None
        if base_url:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        elif os.getenv("DASHSCOPE_API_KEY"):
            api_key = os.getenv("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _normalize_llm_payload(self, question: str, payload: dict[str, Any]) -> QueryIntent:
        intent_type_raw = str(payload.get("intent_type", "general")).strip().lower()
        intent_type = intent_type_raw if intent_type_raw in self.SUPPORTED_INTENTS else "general"

        retrieval_query = str(payload.get("retrieval_query", "")).strip() or question.strip()

        keywords_raw = payload.get("keywords", [])
        keywords: list[str] = []
        if isinstance(keywords_raw, list):
            keywords = [str(item).strip() for item in keywords_raw if str(item).strip()]
        elif isinstance(keywords_raw, str):
            keywords = [part.strip() for part in re.split(r"[，,、\s]+", keywords_raw) if part.strip()]

        need_latest = bool(payload.get("need_latest", False))
        if not need_latest:
            need_latest = any(token in question for token in ("最新", "当前", "现行", "生效"))

        return QueryIntent(
            original_question=question,
            retrieval_query=retrieval_query,
            intent_type=intent_type,
            keywords=keywords[:8],
            need_latest=need_latest,
            source="llm",
        )

    def _parse_with_rules(self, question: str) -> QueryIntent:
        text = question.strip()
        lower_text = text.lower()
        intent_type = "general"
        if any(token in text for token in ("推荐", "订阅", "推送", "关注什么", "有什么更新")):
            intent_type = "recommendation"
        elif any(token in text for token in ("制度", "报销", "审批", "流程", "规范")):
            intent_type = "policy"
        elif "faq" in lower_text or "常见问题" in text:
            intent_type = "faq"
        elif any(token in text for token in ("项目", "里程碑", "交付", "周报")):
            intent_type = "project"
        elif any(token in text for token in ("结论", "会议纪要", "群聊")):
            intent_type = "chat_summary"

        keywords = [part for part in re.split(r"[，,。！？、\s]+", text) if 1 < len(part) <= 16][:8]
        need_latest = any(token in text for token in ("最新", "当前", "现行", "生效"))

        return QueryIntent(
            original_question=question,
            retrieval_query=text or question,
            intent_type=intent_type,
            keywords=keywords,
            need_latest=need_latest,
            source="rule",
        )
