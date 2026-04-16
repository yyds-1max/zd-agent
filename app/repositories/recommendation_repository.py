import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import settings


class RecommendationRepository:
    def __init__(self) -> None:
        self._db_path = Path(settings.metadata_db)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def list_latest_knowledge(self, limit: int = 300) -> list[dict[str, Any]]:
        sql = """
            SELECT
                doc_id,
                title,
                source_type,
                version,
                is_latest,
                effective_date,
                updated_at,
                MAX(summary) AS summary,
                MAX(tags) AS tags,
                MAX(role_scope) AS role_scope,
                MAX(department_scope) AS department_scope,
                MAX(project_scope) AS project_scope
            FROM knowledge_metadata
            WHERE is_latest = 1
            GROUP BY doc_id, title, source_type, version, is_latest, effective_date, updated_at
            ORDER BY COALESCE(updated_at, effective_date, '') DESC
            LIMIT ?
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(sql, (max(1, int(limit)),)).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def list_recent_queries(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        sql = """
            SELECT
                query_id,
                question,
                retrieval_query,
                intent_type,
                citation_doc_ids,
                created_at
            FROM query_logs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(sql, (user_id, max(1, int(limit)))).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def list_recent_clicks(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        sql = """
            SELECT
                c.id,
                c.query_id,
                c.doc_id,
                c.title,
                c.action,
                c.position,
                c.created_at,
                COALESCE(c.user_id, q.user_id) AS resolved_user_id
            FROM query_click_logs c
            LEFT JOIN query_logs q ON q.query_id = c.query_id
            WHERE COALESCE(c.user_id, q.user_id) = ?
            ORDER BY c.created_at DESC
            LIMIT ?
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(sql, (user_id, max(1, int(limit)))).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def save_push_log(self, item: dict[str, Any]) -> int:
        row = self._normalize_push(item)
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO recommendation_push_logs (
                    user_id,
                    channel,
                    item_count,
                    payload,
                    status,
                    trigger_reason
                ) VALUES (
                    :user_id,
                    :channel,
                    :item_count,
                    :payload,
                    :status,
                    :trigger_reason
                )
                """,
                row,
            )
            conn.commit()
            return int(cursor.lastrowid)

    def _ensure_tables(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_push_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    item_count INTEGER NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'triggered',
                    trigger_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_recommendation_push_logs_user_id
                ON recommendation_push_logs(user_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_recommendation_push_logs_created_at
                ON recommendation_push_logs(created_at)
                """
            )
            conn.commit()

    @staticmethod
    def _normalize_push(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": str(item.get("user_id") or "anonymous").strip() or "anonymous",
            "channel": str(item.get("channel") or "manual").strip() or "manual",
            "item_count": int(item.get("item_count") or 0),
            "payload": str(item.get("payload") or "[]"),
            "status": str(item.get("status") or "triggered").strip() or "triggered",
            "trigger_reason": str(item.get("trigger_reason") or "").strip() or None,
        }
