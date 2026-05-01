from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.repositories.conversation_session_repository import ConversationSessionRepository


class ConversationSessionService:
    def __init__(self, repository: ConversationSessionRepository):
        self.repository = repository

    def active_conversation_id(self, *, channel_key: str) -> str:
        conversation_id = self.repository.get_active_conversation_id(
            channel_key=channel_key,
        )
        if conversation_id:
            return conversation_id
        conversation_id = self._new_conversation_id(channel_key)
        self.repository.set_active_conversation_id(
            channel_key=channel_key,
            conversation_id=conversation_id,
        )
        return conversation_id

    def start_new_conversation(self, *, channel_key: str) -> str:
        conversation_id = self._new_conversation_id(channel_key)
        self.repository.set_active_conversation_id(
            channel_key=channel_key,
            conversation_id=conversation_id,
        )
        return conversation_id

    def _new_conversation_id(self, channel_key: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        suffix = uuid4().hex[:8]
        safe_key = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in channel_key
        )[:48]
        return f"{safe_key}:conv:{timestamp}:{suffix}"
