from __future__ import annotations

import json
import threading
from pathlib import Path


class ConversationSessionRepository:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._lock = threading.Lock()

    def get_active_conversation_id(self, *, channel_key: str) -> str | None:
        with self._lock:
            data = self._load()
        value = data.get(channel_key)
        return str(value) if value else None

    def set_active_conversation_id(
        self,
        *,
        channel_key: str,
        conversation_id: str,
    ) -> None:
        with self._lock:
            data = self._load()
            data[channel_key] = conversation_id
            self._save(data)

    def _load(self) -> dict[str, str]:
        if not self.store_path.exists():
            return {}
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items() if value}

    def _save(self, data: dict[str, str]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
