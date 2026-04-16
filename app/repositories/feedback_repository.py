import json
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import settings


class FeedbackRepository:
    def __init__(self) -> None:
        self._db_path = Path(settings.metadata_db)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def save_feedback(self, item: dict[str, Any]) -> int:
        normalized = self._normalize_feedback(item)
        snapshot = self._get_query_snapshot(str(normalized["query_id"]))
        row = {
            **normalized,
            "user_id": snapshot.get("user_id"),
            "question": snapshot.get("question"),
            "intent_type": snapshot.get("intent_type"),
        }
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_logs (
                    query_id,
                    helpful,
                    is_obsolete,
                    note,
                    user_id,
                    question,
                    intent_type
                ) VALUES (
                    :query_id,
                    :helpful,
                    :is_obsolete,
                    :note,
                    :user_id,
                    :question,
                    :intent_type
                )
                """,
                row,
            )
            conn.commit()
            return int(cursor.lastrowid)

    def upsert_governance_issue(self, item: dict[str, Any]) -> int:
        normalized = self._normalize_issue(item)
        query_snapshot = self._get_query_snapshot(str(normalized["query_id"]))
        first_doc_id = self._first_doc_id(query_snapshot.get("citation_doc_ids"))
        row = {
            **normalized,
            "user_id": normalized.get("user_id") or query_snapshot.get("user_id"),
            "question": normalized.get("question") or query_snapshot.get("question"),
            "doc_id": normalized.get("doc_id") or first_doc_id,
        }
        with sqlite3.connect(self._db_path) as conn:
            existing = conn.execute(
                "SELECT id, occurrence_count FROM governance_issues WHERE issue_key = ?",
                (row["issue_key"],),
            ).fetchone()
            if existing:
                issue_id = int(existing[0])
                occurrence_count = int(existing[1] or 0) + 1
                conn.execute(
                    """
                    UPDATE governance_issues
                    SET
                        feedback_id = :feedback_id,
                        detail = :detail,
                        note = :note,
                        user_id = :user_id,
                        question = :question,
                        doc_id = :doc_id,
                        severity = :severity,
                        status = :status,
                        occurrence_count = :occurrence_count,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE issue_key = :issue_key
                    """,
                    {**row, "occurrence_count": occurrence_count},
                )
                conn.commit()
                return issue_id

            cursor = conn.execute(
                """
                INSERT INTO governance_issues (
                    issue_key,
                    query_id,
                    feedback_id,
                    issue_type,
                    severity,
                    status,
                    title,
                    detail,
                    note,
                    user_id,
                    question,
                    doc_id
                ) VALUES (
                    :issue_key,
                    :query_id,
                    :feedback_id,
                    :issue_type,
                    :severity,
                    :status,
                    :title,
                    :detail,
                    :note,
                    :user_id,
                    :question,
                    :doc_id
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
                CREATE TABLE IF NOT EXISTS feedback_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    helpful INTEGER NOT NULL,
                    is_obsolete INTEGER NOT NULL DEFAULT 0,
                    note TEXT,
                    user_id TEXT,
                    question TEXT,
                    intent_type TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_logs_query_id
                ON feedback_logs(query_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_logs_created_at
                ON feedback_logs(created_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS governance_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue_key TEXT NOT NULL UNIQUE,
                    query_id TEXT NOT NULL,
                    feedback_id INTEGER,
                    issue_type TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'medium',
                    status TEXT NOT NULL DEFAULT 'open',
                    title TEXT NOT NULL,
                    detail TEXT,
                    note TEXT,
                    user_id TEXT,
                    question TEXT,
                    doc_id TEXT,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_governance_issues_issue_type
                ON governance_issues(issue_type)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_governance_issues_status
                ON governance_issues(status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_governance_issues_created_at
                ON governance_issues(created_at)
                """
            )
            conn.commit()

    def _get_query_snapshot(self, query_id: str) -> dict[str, Any]:
        if not query_id:
            return {}
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """
                    SELECT
                        query_id,
                        user_id,
                        question,
                        intent_type,
                        citation_doc_ids
                    FROM query_logs
                    WHERE query_id = ?
                    """,
                    (query_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                return {}
        return dict(row) if row else {}

    @staticmethod
    def _normalize_feedback(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "query_id": str(item.get("query_id") or "").strip(),
            "helpful": 1 if bool(item.get("helpful")) else 0,
            "is_obsolete": 1 if bool(item.get("is_obsolete")) else 0,
            "note": str(item.get("note") or "").strip() or None,
        }

    @staticmethod
    def _normalize_issue(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "issue_key": str(item.get("issue_key") or "").strip(),
            "query_id": str(item.get("query_id") or "").strip(),
            "feedback_id": int(item.get("feedback_id") or 0),
            "issue_type": str(item.get("issue_type") or "").strip(),
            "severity": str(item.get("severity") or "medium").strip() or "medium",
            "status": str(item.get("status") or "open").strip() or "open",
            "title": str(item.get("title") or "未命名治理问题").strip() or "未命名治理问题",
            "detail": str(item.get("detail") or "").strip() or None,
            "note": str(item.get("note") or "").strip() or None,
            "user_id": str(item.get("user_id") or "").strip() or None,
            "question": str(item.get("question") or "").strip() or None,
            "doc_id": str(item.get("doc_id") or "").strip() or None,
        }

    @staticmethod
    def _first_doc_id(raw: Any) -> str | None:
        if not raw:
            return None
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
            return values[0] if values else None
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, list):
                    values = [str(item).strip() for item in parsed if str(item).strip()]
                    return values[0] if values else None
            return text
        return None
