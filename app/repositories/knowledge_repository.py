from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable

from app.schemas.knowledge import KnowledgeChunk, KnowledgeDocument

DOC_TYPE_MAP = {
    "制度文档": "policy",
    "财务细则": "finance_rule",
    "FAQ": "faq",
    "入职指南": "onboarding",
    "项目需求文档": "project_requirement",
    "项目周报": "project_weekly",
    "项目计划文档": "project_plan",
    "聊天结论文档": "chat_summary",
}


class KnowledgeRepository:
    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir
        self._documents = self._load_documents()
        self._documents_by_topic = self._group_by_topic(self._documents)
        self._chunks = self._build_chunks(self._documents)
        self._chunks_by_doc_id = self._group_chunks_by_doc_id(self._chunks)
        self._known_projects = sorted(
            {doc.project_name for doc in self._documents if doc.project_name}
        )

    def list_documents(self) -> list[KnowledgeDocument]:
        return list(self._documents)

    def get_document_by_id(self, doc_id: str) -> KnowledgeDocument:
        for document in self._documents:
            if document.doc_id == doc_id:
                return document
        raise KeyError(f"Unknown doc_id `{doc_id}`")

    def list_known_projects(self) -> list[str]:
        return list(self._known_projects)

    def list_chunks(self) -> list[KnowledgeChunk]:
        return list(self._chunks)

    def list_chunks_for_documents(
        self, documents: list[KnowledgeDocument]
    ) -> list[KnowledgeChunk]:
        doc_ids = {document.doc_id for document in documents}
        return [
            chunk
            for chunk in self._chunks
            if chunk.doc_id in doc_ids
        ]

    def list_chunks_for_document(self, doc_id: str) -> list[KnowledgeChunk]:
        return list(self._chunks_by_doc_id.get(doc_id, []))

    def find_by_topic(self, topic: str) -> list[KnowledgeDocument]:
        return list(self._documents_by_topic.get(topic, []))

    def _load_documents(self) -> list[KnowledgeDocument]:
        documents = [
            self._parse_document(path)
            for path in sorted(self.fixtures_dir.glob("*.txt"))
        ]
        self._mark_latest_versions(documents)
        return documents

    def _build_chunks(self, documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            chunks.extend(self._chunk_document(document))
        return chunks

    def _parse_document(self, path: Path) -> KnowledgeDocument:
        content = path.read_text(encoding="utf-8")
        metadata_text, _, body_text = content.partition("正文")

        metadata: dict[str, str] = {}
        for raw_line in metadata_text.splitlines():
            line = raw_line.strip()
            if "：" not in line:
                continue
            key, value = line.split("：", 1)
            metadata[key.strip()] = value.strip()

        body_lines = [line.strip() for line in body_text.splitlines() if line.strip()]
        title = body_lines[0] if body_lines else path.stem
        sections = body_lines[1:] if len(body_lines) > 1 else []

        published_at = self._parse_date(metadata.get("发布日期"))
        updated_at = self._parse_date(metadata.get("更新时间")) or published_at
        doc_type = DOC_TYPE_MAP.get(metadata.get("文档类型", ""), "general")
        topic = self._normalize_topic(title)
        project_name = self._extract_project_name(title)

        return KnowledgeDocument(
            doc_id=path.stem,
            title=title,
            doc_type=doc_type,
            topic=topic,
            permission_level=metadata.get("权限级别", "全员可见"),
            version=metadata.get("版本号"),
            status=metadata.get("状态"),
            published_at=published_at,
            updated_at=updated_at,
            is_latest=False,
            project_name=project_name,
            source_path=str(path),
            body="\n".join(body_lines),
            sections=sections,
        )

    def _group_chunks_by_doc_id(
        self, chunks: Iterable[KnowledgeChunk]
    ) -> dict[str, list[KnowledgeChunk]]:
        grouped: dict[str, list[KnowledgeChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        return grouped

    def _group_by_topic(
        self, documents: Iterable[KnowledgeDocument]
    ) -> dict[str, list[KnowledgeDocument]]:
        grouped: dict[str, list[KnowledgeDocument]] = {}
        for document in documents:
            grouped.setdefault(document.topic, []).append(document)
        return grouped

    def _mark_latest_versions(self, documents: list[KnowledgeDocument]) -> None:
        grouped = self._group_by_topic(documents)
        for same_topic_docs in grouped.values():
            if len(same_topic_docs) == 1:
                same_topic_docs[0].is_latest = "过期" not in (same_topic_docs[0].status or "")
                continue
            ranked = sorted(same_topic_docs, key=self._document_sort_key, reverse=True)
            for index, document in enumerate(ranked):
                document.is_latest = index == 0

    def _document_sort_key(self, document: KnowledgeDocument) -> tuple[int, tuple[int, ...], int]:
        status = document.status or ""
        if "当前生效" in status:
            status_rank = 3
        elif "生效" in status and "过期" not in status:
            status_rank = 2
        elif "过期" in status:
            status_rank = 0
        else:
            status_rank = 1

        version_numbers = tuple(int(item) for item in re.findall(r"\d+", document.version or ""))
        published_ordinal = document.published_at.toordinal() if document.published_at else 0
        return status_rank, version_numbers, published_ordinal

    def _parse_date(self, value: str | None) -> date | None:
        if not value:
            return None
        value = value.strip()
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    def _normalize_topic(self, title: str) -> str:
        topic = re.sub(r"（[^）]*V[\d.]+[^）]*）", "", title)
        topic = re.sub(r"\([^)]*V[\d.]+[^)]*\)", "", topic)
        topic = re.sub(r"（\d{4}\s*年[^）]*）", "", topic)
        topic = re.sub(r"\([^)]*\d{4}[^)]*\)", "", topic)
        topic = topic.replace("（内部口径版）", "")
        topic = topic.replace(" ", "")
        return topic.strip("：:")

    def _extract_project_name(self, title: str) -> str | None:
        match = re.search(r"项目[“\"]?([^”\" ]+)", title)
        return match.group(1) if match else None

    def _chunk_document(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        lines = [line for line in document.sections if line]
        if not lines:
            return [self._build_chunk(document, 0, None, None, document.body)]
        if document.doc_type == "faq":
            return self._chunk_faq_document(document, lines)

        section_blocks = self._split_into_sections(lines)
        chunks: list[KnowledgeChunk] = []
        chunk_index = 0
        for section_title, section_lines in section_blocks:
            sub_blocks = self._split_section_into_subblocks(section_lines)
            for subsection_title, block_lines in sub_blocks:
                for window in self._window_lines(block_lines):
                    text_lines = [section_title]
                    if subsection_title:
                        text_lines.append(subsection_title)
                    text_lines.extend(window)
                    chunks.append(
                        self._build_chunk(
                            document,
                            chunk_index,
                            section_title,
                            subsection_title,
                            "\n".join(text_lines),
                        )
                    )
                    chunk_index += 1

        return chunks or [self._build_chunk(document, 0, None, None, document.body)]

    def _chunk_faq_document(
        self, document: KnowledgeDocument, lines: list[str]
    ) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        chunk_index = 0
        current_question: str | None = None
        current_answer_lines: list[str] = []

        def flush() -> None:
            nonlocal chunk_index, current_question, current_answer_lines
            if not current_question:
                return
            text_lines = [current_question, *current_answer_lines]
            chunks.append(
                self._build_chunk(
                    document,
                    chunk_index,
                    "FAQ",
                    current_question,
                    "\n".join(text_lines),
                )
            )
            chunk_index += 1
            current_question = None
            current_answer_lines = []

        for line in lines:
            if re.match(r"^Q\d+[：:]", line):
                flush()
                current_question = line
                continue
            if current_question:
                current_answer_lines.append(line)
            else:
                chunks.append(
                    self._build_chunk(document, chunk_index, "FAQ", None, line)
                )
                chunk_index += 1

        flush()
        return chunks or [self._build_chunk(document, 0, "FAQ", None, document.body)]

    def _split_into_sections(self, lines: list[str]) -> list[tuple[str, list[str]]]:
        sections: list[tuple[str, list[str]]] = []
        current_title: str | None = None
        current_lines: list[str] = []
        for line in lines:
            if self._is_top_level_heading(line):
                if current_title is not None:
                    sections.append((current_title, current_lines))
                current_title = line
                current_lines = []
                continue
            if current_title is None:
                current_title = "导语"
            current_lines.append(line)
        if current_title is not None:
            sections.append((current_title, current_lines))
        return sections

    def _split_section_into_subblocks(
        self, lines: list[str]
    ) -> list[tuple[str | None, list[str]]]:
        if not lines:
            return [(None, [])]

        has_subheading = any(self._is_subheading(line) for line in lines)
        if not has_subheading:
            return [(None, lines)]

        blocks: list[tuple[str | None, list[str]]] = []
        current_subheading: str | None = None
        current_lines: list[str] = []
        for line in lines:
            if self._is_subheading(line):
                if current_subheading is not None or current_lines:
                    blocks.append((current_subheading, current_lines))
                current_subheading = line
                current_lines = []
                continue
            current_lines.append(line)

        if current_subheading is not None or current_lines:
            blocks.append((current_subheading, current_lines))
        return blocks

    def _window_lines(
        self,
        lines: list[str],
        max_chars: int = 220,
        max_lines: int = 5,
        overlap: int = 1,
    ) -> list[list[str]]:
        if not lines:
            return [[]]

        windows: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for line in lines:
            projected_chars = current_chars + len(line)
            if current and (projected_chars > max_chars or len(current) >= max_lines):
                windows.append(list(current))
                retained = current[-overlap:] if overlap > 0 else []
                current = list(retained)
                current_chars = sum(len(item) for item in current)
            current.append(line)
            current_chars += len(line)
        if current:
            windows.append(current)
        return windows

    def _build_chunk(
        self,
        document: KnowledgeDocument,
        chunk_index: int,
        section_title: str | None,
        subsection_title: str | None,
        text: str,
    ) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=f"{document.doc_id}::chunk::{chunk_index:03d}",
            doc_id=document.doc_id,
            chunk_index=chunk_index,
            doc_title=document.title,
            doc_type=document.doc_type,
            topic=document.topic,
            permission_level=document.permission_level,
            version=document.version,
            status=document.status,
            published_at=document.published_at,
            updated_at=document.updated_at,
            is_latest=document.is_latest,
            project_name=document.project_name,
            source_path=document.source_path,
            section_title=section_title,
            subsection_title=subsection_title,
            text=text.strip(),
        )

    def _is_top_level_heading(self, line: str) -> bool:
        return bool(re.match(r"^[一二三四五六七八九十]+、", line) or re.match(r"^\d+[.、]", line))

    def _is_subheading(self, line: str) -> bool:
        if self._is_top_level_heading(line):
            return False
        if re.match(r"^[QA]\d+[：:]", line):
            return False
        if any(token in line for token in ["：", "。", "；", "，", "\t"]):
            return False
        if re.search(r"\d{4}[-年]", line):
            return False
        return 1 <= len(line) <= 12
