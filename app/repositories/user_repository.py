from __future__ import annotations

import json
from pathlib import Path

from app.schemas.user import DirectoryUser


class UserRepository:
    def __init__(self, directory_path: Path):
        self.directory_path = directory_path
        self._users = self._load_users()

    def _load_users(self) -> dict[str, DirectoryUser]:
        raw_users = json.loads(self.directory_path.read_text(encoding="utf-8"))
        return {
            item["user_id"]: DirectoryUser.model_validate(item)
            for item in raw_users
        }

    def get_by_user_id(self, user_id: str) -> DirectoryUser:
        if user_id not in self._users:
            available = ", ".join(sorted(self._users))
            raise KeyError(f"Unknown user_id `{user_id}`. Available demo users: {available}")
        return self._users[user_id]

    def find_by_user_id(self, user_id: str) -> DirectoryUser | None:
        return self._users.get(user_id)
