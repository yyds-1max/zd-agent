from __future__ import annotations

import json
import re

from app.schemas.knowledge import KnowledgeChunk
from app.schemas.llm import VersionDiffLLMOutput
from app.schemas.version import VersionDiffResult
from app.services.qwen_llm_service import QwenStructuredLLMService

ALLOWED_CHANGE_TYPES = {"unchanged", "modified", "deleted", "added"}


class VersionDiffService:
    def __init__(self, llm_service: QwenStructuredLLMService | None = None):
        self.llm_service = llm_service

    def compare(
        self,
        *,
        old_chunk: KnowledgeChunk,
        new_chunk: KnowledgeChunk | None,
        question: str | None = None,
    ) -> VersionDiffResult:
        if self.llm_service is not None and self.llm_service.is_available():
            try:
                return self._compare_with_llm(
                    old_chunk=old_chunk,
                    new_chunk=new_chunk,
                    question=question,
                )
            except Exception:
                pass
        return self._compare_with_heuristic(old_chunk=old_chunk, new_chunk=new_chunk)

    def _compare_with_llm(
        self,
        *,
        old_chunk: KnowledgeChunk,
        new_chunk: KnowledgeChunk | None,
        question: str | None,
    ) -> VersionDiffResult:
        system_prompt = (
            "你是企业知识版本差异分析工具。"
            "请比较旧版知识片段与新版知识片段，输出严格 JSON。"
            "change_type 仅可为 unchanged、modified、deleted、added。"
            "summary 必须简洁明确，直接说明变化点。"
            "如果新版片段为空，优先判断为 deleted。"
        )
        user_prompt = (
            f"用户问题:\n{question or '无'}\n\n"
            f"旧版 chunk:\n{json.dumps(self._chunk_payload(old_chunk), ensure_ascii=False, indent=2)}\n\n"
            f"新版 chunk:\n{json.dumps(self._chunk_payload(new_chunk) if new_chunk else None, ensure_ascii=False, indent=2)}\n\n"
            "请输出 JSON，字段要求：\n"
            "- change_type: unchanged / modified / deleted / added\n"
            "- summary: 1-2 句，概括最重要变化\n"
            "- confidence: 0~1\n"
            "- key_changes: 最多 4 条变化点\n"
        )
        output = self.llm_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=VersionDiffLLMOutput,
        )
        change_type = output.change_type if output.change_type in ALLOWED_CHANGE_TYPES else "modified"
        return VersionDiffResult(
            source_chunk_id=old_chunk.chunk_id,
            source_doc_id=old_chunk.doc_id,
            source_version=old_chunk.version,
            latest_chunk_id=new_chunk.chunk_id if new_chunk else None,
            latest_doc_id=new_chunk.doc_id if new_chunk else None,
            latest_version=new_chunk.version if new_chunk else None,
            change_type=change_type,
            summary=output.summary.strip(),
            confidence=output.confidence,
            key_changes=output.key_changes[:4],
        )

    def _compare_with_heuristic(
        self,
        *,
        old_chunk: KnowledgeChunk,
        new_chunk: KnowledgeChunk | None,
    ) -> VersionDiffResult:
        old_text = self._normalize_text(old_chunk.text)
        new_text = self._normalize_text(new_chunk.text) if new_chunk else ""

        if new_chunk is None:
            change_type = "deleted"
            summary = "该旧版内容在新版文档中未找到稳定对应片段，可能已被删除或并入其他段落。"
            key_changes: list[str] = []
            confidence = 0.7
        elif old_text == new_text:
            change_type = "unchanged"
            summary = "该内容在新版中保持不变。"
            key_changes = []
            confidence = 0.98
        else:
            change_type = "modified"
            key_changes = self._extract_key_changes(old_chunk.text, new_chunk.text)
            summary = (
                f"该内容在新版中已调整。"
                f"{f' 关键变化：{key_changes[0]}' if key_changes else ''}"
            ).strip()
            confidence = 0.82

        return VersionDiffResult(
            source_chunk_id=old_chunk.chunk_id,
            source_doc_id=old_chunk.doc_id,
            source_version=old_chunk.version,
            latest_chunk_id=new_chunk.chunk_id if new_chunk else None,
            latest_doc_id=new_chunk.doc_id if new_chunk else None,
            latest_version=new_chunk.version if new_chunk else None,
            change_type=change_type,
            summary=summary,
            confidence=confidence,
            key_changes=key_changes if new_chunk is not None else [],
        )

    def _chunk_payload(self, chunk: KnowledgeChunk) -> dict[str, str | int | None]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "version": chunk.version,
            "section_title": chunk.section_title,
            "subsection_title": chunk.subsection_title,
            "text": chunk.text,
        }

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _extract_key_changes(self, old_text: str, new_text: str) -> list[str]:
        old_lines = [line.strip() for line in old_text.splitlines() if line.strip()]
        new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]
        changes: list[str] = []

        old_set = set(old_lines)
        new_set = set(new_lines)
        removed = [line for line in old_lines if line not in new_set]
        added = [line for line in new_lines if line not in old_set]

        for old_line, new_line in zip(removed, added):
            changes.append(f"`{old_line}` 调整为 `{new_line}`")
        if len(added) > len(removed):
            for line in added[len(removed) :]:
                changes.append(f"新增 `{line}`")
        elif len(removed) > len(added):
            for line in removed[len(added) :]:
                changes.append(f"删除 `{line}`")
        return changes[:4]
