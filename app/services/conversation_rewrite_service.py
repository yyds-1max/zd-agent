from __future__ import annotations

import json
import re

from app.schemas.conversation import ConversationRewriteResult, ConversationTurn
from app.schemas.llm import ConversationRewriteLLMOutput
from app.services.qwen_llm_service import QwenStructuredLLMService


class ConversationRewriteService:
    def __init__(
        self,
        llm_service: QwenStructuredLLMService | None = None,
        min_confidence: float = 0.72,
        max_history_turns: int = 6,
    ):
        self.llm_service = llm_service
        self.min_confidence = min_confidence
        self.max_history_turns = max_history_turns

    def rewrite(
        self,
        *,
        question: str,
        history: list[ConversationTurn],
    ) -> ConversationRewriteResult:
        llm_result = self._rewrite_with_llm(question=question, history=history)
        if llm_result is not None:
            return llm_result
        return self._rewrite_with_template(question=question, history=history)

    def _rewrite_with_llm(
        self,
        *,
        question: str,
        history: list[ConversationTurn],
    ) -> ConversationRewriteResult | None:
        if self.llm_service is None or not self.llm_service.is_available() or not history:
            return None

        history_payload = [
            {
                "role": turn.role,
                "content": self._truncate(turn.content, limit=420),
                "metadata": turn.metadata,
            }
            for turn in history[-self.max_history_turns :]
        ]
        try:
            output = self.llm_service.generate_structured(
                system_prompt=(
                    "你是企业知识助手的多轮查询改写器。"
                    "你的任务是把用户当前追问改写成可以独立检索的中文问题。"
                    "只能使用给定对话历史和当前问题，不要编造企业事实。"
                    "如果当前问题已经完整，不需要强行改写。"
                    "如果追问缺少必要指代对象且历史也无法确定，needs_clarification=true。"
                    "输出必须是严格 JSON。"
                ),
                user_prompt=(
                    f"最近对话历史:\n{json.dumps(history_payload, ensure_ascii=False, indent=2)}\n\n"
                    f"当前用户问题:\n{question}\n\n"
                    "请输出 JSON 字段：\n"
                    "- standalone_question: 改写后的独立问题，中文，适合企业知识库检索\n"
                    "- needs_clarification: 是否必须追问用户补充范围\n"
                    "- referenced_topic: 当前问题承接的主题，如制度名、项目名或 FAQ 主题；无法判断则为 null\n"
                    "- rewrite_reason: 简短说明如何利用历史\n"
                    "- confidence: 0 到 1\n"
                ),
                response_model=ConversationRewriteLLMOutput,
            )
        except Exception:
            return None

        standalone_question = output.standalone_question.strip()
        if output.confidence < self.min_confidence or not standalone_question:
            return None

        return ConversationRewriteResult(
            standalone_question=standalone_question,
            needs_clarification=output.needs_clarification,
            referenced_topic=output.referenced_topic,
            rewrite_reason=output.rewrite_reason,
            confidence=output.confidence,
            source="llm",
        )

    def _rewrite_with_template(
        self,
        *,
        question: str,
        history: list[ConversationTurn],
    ) -> ConversationRewriteResult:
        recent_user = next((turn for turn in reversed(history) if turn.role == "user"), None)
        recent_assistant = next(
            (turn for turn in reversed(history) if turn.role == "assistant"),
            None,
        )
        if recent_user is None:
            return ConversationRewriteResult(
                standalone_question=question,
                rewrite_reason="没有可用历史，保留原问题。",
                confidence=0.5,
                source="rule",
            )

        parts = [
            "请结合以下多轮对话上下文理解当前追问，并检索与当前追问最相关的企业知识。",
            f"上一轮用户问题：{recent_user.content}",
        ]
        if recent_assistant is not None:
            parts.append(f"上一轮助手回答摘要：{self._truncate(recent_assistant.content)}")
        parts.append(f"当前用户追问：{question}")
        return ConversationRewriteResult(
            standalone_question="\n".join(parts),
            referenced_topic=self._infer_referenced_topic(recent_user.content),
            rewrite_reason="规则回退：使用上一轮用户问题和助手回答摘要补全追问上下文。",
            confidence=0.68,
            source="rule",
        )

    def _infer_referenced_topic(self, text: str) -> str | None:
        compact = re.sub(r"\s+", "", text)
        for marker in ["制度", "项目", "FAQ", "报销", "差旅", "入职"]:
            if marker in compact:
                return marker
        return None

    def _truncate(self, text: str, limit: int = 260) -> str:
        content = re.sub(r"\s+", " ", text).strip()
        if len(content) <= limit:
            return content
        return content[:limit] + "..."
