import json
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import settings


class QueryLogRepository:
    def __init__(self) -> None:
        self._db_path = Path(settings.metadata_db)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def save_query_log(self, row: dict[str, Any]) -> None:
        normalized = self._normalize_query_row(row)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    query_id,
                    question,
                    retrieval_query,
                    intent_type,
                    user_id,
                    role,
                    department,
                    projects,
                    citation_doc_ids,
                    citation_count,
                    version_hint
                ) VALUES (
                    :query_id,
                    :question,
                    :retrieval_query,
                    :intent_type,
                    :user_id,
                    :role,
                    :department,
                    :projects,
                    :citation_doc_ids,
                    :citation_count,
                    :version_hint
                )
                ON CONFLICT(query_id) DO UPDATE SET
                    question=excluded.question,
                    retrieval_query=excluded.retrieval_query,
                    intent_type=excluded.intent_type,
                    user_id=excluded.user_id,
                    role=excluded.role,
                    department=excluded.department,
                    projects=excluded.projects,
                    citation_doc_ids=excluded.citation_doc_ids,
                    citation_count=excluded.citation_count,
                    version_hint=excluded.version_hint
                """,
                normalized,
            )
            conn.commit()

    def save_click_log(self, row: dict[str, Any]) -> int:
        normalized = self._normalize_click_row(row)
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO query_click_logs (
                    query_id,
                    doc_id,
                    title,
                    position,
                    action,
                    user_id
                ) VALUES (
                    :query_id,
                    :doc_id,
                    :title,
                    :position,
                    :action,
                    :user_id
                )
                """,
                normalized,
            )
            conn.commit()
            click_id = cursor.lastrowid
        return int(click_id)

    def _ensure_tables(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    retrieval_query TEXT NOT NULL,
                    intent_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT NOT NULL,
                    projects TEXT NOT NULL,
                    citation_doc_ids TEXT NOT NULL,
                    citation_count INTEGER NOT NULL DEFAULT 0,
                    version_hint TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_logs_user_id
                ON query_logs(user_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_logs_created_at
                ON query_logs(created_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_click_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    title TEXT,
                    position INTEGER,
                    action TEXT NOT NULL,
                    user_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_click_logs_query_id
                ON query_click_logs(query_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_click_logs_created_at
                ON query_click_logs(created_at)
                """
            )
            conn.commit()

    @staticmethod
    def _normalize_query_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "query_id": str(row.get("query_id") or "").strip(),
            "question": str(row.get("question") or "").strip(),
            "retrieval_query": str(row.get("retrieval_query") or "").strip(),
            "intent_type": str(row.get("intent_type") or "general").strip() or "general",
            "user_id": str(row.get("user_id") or "anonymous").strip() or "anonymous",
            "role": str(row.get("role") or "employee").strip() or "employee",
            "department": str(row.get("department") or "unknown").strip() or "unknown",
            "projects": json.dumps(row.get("projects") or [], ensure_ascii=False),
            "citation_doc_ids": json.dumps(row.get("citation_doc_ids") or [], ensure_ascii=False),
            "citation_count": int(row.get("citation_count") or 0),
            "version_hint": (str(row.get("version_hint")).strip() if row.get("version_hint") else None),
        }

    @staticmethod
    def _normalize_click_row(row: dict[str, Any]) -> dict[str, Any]:
        position = row.get("position")
        return {
            "query_id": str(row.get("query_id") or "").strip(),
            "doc_id": str(row.get("doc_id") or "").strip(),
            "title": str(row.get("title") or "").strip() or None,
            "position": int(position) if position is not None else None,
            "action": str(row.get("action") or "open_citation").strip() or "open_citation",
            "user_id": str(row.get("user_id") or "").strip() or None,
        }
