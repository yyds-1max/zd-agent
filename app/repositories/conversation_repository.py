from __future__ import annotations

import json
import threading
from pathlib import Path

from app.schemas.conversation import ConversationTurn


class ConversationRepository:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._lock = threading.Lock()

    def list_recent(
        self,
        *,
        conversation_id: str,
        limit: int,
    ) -> list[ConversationTurn]:
        if limit <= 0:
            return []
        with self._lock:
            data = self._load()
            turns = [
                ConversationTurn.model_validate(item)
                for item in data.get(conversation_id, [])
            ]
        return turns[-limit:]

    def append_turn(
        self,
        *,
        conversation_id: str,
        turn: ConversationTurn,
    ) -> None:
        with self._lock:
            data = self._load()
            turns = data.setdefault(conversation_id, [])
            turns.append(turn.model_dump(mode="json"))
            data[conversation_id] = turns[-40:]
            self._save(data)

    def append_exchange(
        self,
        *,
        conversation_id: str,
        user_turn: ConversationTurn,
        assistant_turn: ConversationTurn,
    ) -> None:
        with self._lock:
            data = self._load()
            turns = data.setdefault(conversation_id, [])
            turns.append(user_turn.model_dump(mode="json"))
            turns.append(assistant_turn.model_dump(mode="json"))
            data[conversation_id] = turns[-40:]
            self._save(data)

    def _load(self) -> dict[str, list[dict]]:
        if not self.store_path.exists():
            return {}
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, list)
        }

    def _save(self, data: dict[str, list[dict]]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
