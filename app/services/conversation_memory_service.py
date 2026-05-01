from __future__ import annotations

import re
from urllib.parse import quote

from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversation import ConversationTurn
from app.schemas.query import QueryResponse
from app.services.conversation_rewrite_service import ConversationRewriteService


class ConversationMemoryService:
    FOLLOW_UP_TOKENS = [
        "那",
        "那么",
        "这个",
        "这些",
        "它",
        "它们",
        "上述",
        "刚才",
        "上一条",
        "前面",
        "旧版呢",
        "新版呢",
        "有什么变化",
        "需要注意什么",
        "继续",
        "展开",
        "详细说",
    ]

    def __init__(
        self,
        repository: ConversationRepository,
        default_history_limit: int = 6,
        rewrite_service: ConversationRewriteService | None = None,
    ):
        self.repository = repository
        self.default_history_limit = default_history_limit
        self.rewrite_service = rewrite_service or ConversationRewriteService()

    def conversation_id_for(
        self,
        *,
        user_id: str,
        requested_conversation_id: str | None,
    ) -> str:
        if requested_conversation_id:
            return requested_conversation_id
        return f"user:{quote(user_id, safe='')}"

    def load_recent(
        self,
        *,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        return self.repository.list_recent(
            conversation_id=conversation_id,
            limit=limit or self.default_history_limit,
        )

    def contextualize_question(
        self,
        *,
        question: str,
        history: list[ConversationTurn],
    ) -> str:
        if not history or not self._looks_like_follow_up(question):
            return question
        rewrite = self.rewrite_service.rewrite(question=question, history=history)
        if rewrite.needs_clarification:
            return question
        return rewrite.standalone_question

    def record_exchange(
        self,
        *,
        conversation_id: str,
        user_question: str,
        response: QueryResponse,
    ) -> None:
        self.repository.append_exchange(
            conversation_id=conversation_id,
            user_turn=ConversationTurn(
                role="user",
                content=user_question,
            ),
            assistant_turn=ConversationTurn(
                role="assistant",
                content=response.answer,
                metadata={
                    "intent": response.intent.name,
                    "citation_count": str(len(response.citations)),
                    "citation_titles": "；".join(
                        dict.fromkeys(citation.title for citation in response.citations[:3])
                    ),
                    "version_notice": response.version_notice or "",
                },
            ),
        )

    def _looks_like_follow_up(self, question: str) -> bool:
        normalized = re.sub(r"\s+", "", question.strip())
        if not normalized:
            return False
        if any(token in normalized for token in self.FOLLOW_UP_TOKENS):
            return True
        return len(normalized) <= 8 and normalized.endswith(("呢", "吗", "？", "?"))
