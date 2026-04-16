import json
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import settings


class MetadataRepository:
    def __init__(self) -> None:
        self._db_path = Path(settings.metadata_db)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def upsert_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executemany(
                """
                INSERT INTO knowledge_metadata (
                    chunk_id,
                    doc_id,
                    title,
                    source_type,
                    content_chunk,
                    summary,
                    version,
                    is_latest,
                    effective_date,
                    updated_at,
                    role_scope,
                    department_scope,
                    project_scope,
                    tags
                ) VALUES (
                    :chunk_id,
                    :doc_id,
                    :title,
                    :source_type,
                    :content_chunk,
                    :summary,
                    :version,
                    :is_latest,
                    :effective_date,
                    :updated_at,
                    :role_scope,
                    :department_scope,
                    :project_scope,
                    :tags
                )
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id=excluded.doc_id,
                    title=excluded.title,
                    source_type=excluded.source_type,
                    content_chunk=excluded.content_chunk,
                    summary=excluded.summary,
                    version=excluded.version,
                    is_latest=excluded.is_latest,
                    effective_date=excluded.effective_date,
                    updated_at=excluded.updated_at,
                    role_scope=excluded.role_scope,
                    department_scope=excluded.department_scope,
                    project_scope=excluded.project_scope,
                    tags=excluded.tags
                """,
                [self._normalize_row(row) for row in rows],
            )
            conn.commit()
        return len(rows)

    def _ensure_table(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content_chunk TEXT NOT NULL,
                    summary TEXT,
                    version TEXT NOT NULL,
                    is_latest INTEGER NOT NULL,
                    effective_date TEXT,
                    updated_at TEXT,
                    role_scope TEXT NOT NULL,
                    department_scope TEXT NOT NULL,
                    project_scope TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_metadata_doc_id
                ON knowledge_metadata(doc_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_metadata_source_type
                ON knowledge_metadata(source_type)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_metadata_version
                ON knowledge_metadata(version)
                """
            )
            conn.commit()

    def list_document_versions(self, source_type: str | None = None) -> list[dict[str, Any]]:
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
                MAX(content_chunk) AS content_chunk
            FROM knowledge_metadata
        """
        params: tuple[Any, ...] = ()
        if source_type:
            sql += " WHERE source_type = ?"
            params = (source_type,)
        sql += """
            GROUP BY doc_id, title, source_type, version, is_latest, effective_date, updated_at
        """

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        normalized["is_latest"] = 1 if bool(normalized.get("is_latest")) else 0
        normalized["role_scope"] = json.dumps(normalized.get("role_scope", []), ensure_ascii=False)
        normalized["department_scope"] = json.dumps(normalized.get("department_scope", []), ensure_ascii=False)
        normalized["project_scope"] = json.dumps(normalized.get("project_scope", []), ensure_ascii=False)
        normalized["tags"] = json.dumps(normalized.get("tags", []), ensure_ascii=False)
        return normalized
