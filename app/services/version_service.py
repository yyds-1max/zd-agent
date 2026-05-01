from __future__ import annotations

import re

from app.repositories.knowledge_repository import KnowledgeRepository
from app.schemas.knowledge import KnowledgeChunk, KnowledgeDocument, RetrievedChunk
from app.schemas.user import UserProfile
from app.schemas.version import VersionCheckResult
from app.services.permission_service import PermissionService


class VersionService:
    def __init__(
        self,
        knowledge_repository: KnowledgeRepository,
        permission_service: PermissionService,
    ):
        self.knowledge_repository = knowledge_repository
        self.permission_service = permission_service

    def check_versions(
        self, profile: UserProfile, retrieved_chunks: list[RetrievedChunk]
    ) -> list[VersionCheckResult]:
        results: list[VersionCheckResult] = []
        for retrieved in retrieved_chunks:
            chunk = retrieved.chunk
            document = self.document_for_chunk(retrieved.chunk)
            latest_document = self._latest_accessible_document(profile, document)
            has_newer = latest_document.doc_id != document.doc_id
            aligned_chunk, confidence = self._align_chunk_to_document(chunk, latest_document)
            if has_newer:
                if aligned_chunk is not None:
                    section_hint = self._format_chunk_hint(
                        aligned_chunk.section_title,
                        aligned_chunk.subsection_title,
                    )
                    notice = (
                        f"检测到 `{document.title}` 不是当前推荐版本，"
                        f"已在最新版 `{latest_document.title}` 中定位到对应内容"
                        f"{f'（{section_hint}）' if section_hint else ''}。"
                    )
                else:
                    notice = (
                        f"检测到 `{document.title}` 不是当前推荐版本，"
                        f"已定位到最新版 `{latest_document.title}`，但尚未找到稳定对应 chunk。"
                    )
            else:
                notice = f"`{document.title}` 已是当前推荐版本。"
            results.append(
                VersionCheckResult(
                    source_chunk_id=chunk.chunk_id,
                    source_section_title=chunk.section_title,
                    source_subsection_title=chunk.subsection_title,
                    source_doc_id=document.doc_id,
                    source_title=document.title,
                    source_version=document.version,
                    source_is_latest=document.is_latest,
                    has_newer_version=has_newer,
                    latest_doc_id=latest_document.doc_id,
                    latest_title=latest_document.title,
                    latest_version=latest_document.version,
                    latest_published_at=latest_document.published_at,
                    latest_accessible=True,
                    latest_chunk_id=aligned_chunk.chunk_id if aligned_chunk else None,
                    latest_section_title=aligned_chunk.section_title if aligned_chunk else None,
                    latest_subsection_title=aligned_chunk.subsection_title if aligned_chunk else None,
                    latest_chunk_text=aligned_chunk.text if aligned_chunk else None,
                    latest_chunk_match_confidence=confidence if aligned_chunk else 0.0,
                    notice=notice,
                )
            )
        return results

    def document_for_chunk(self, chunk: KnowledgeChunk) -> KnowledgeDocument:
        return self.knowledge_repository.get_document_by_id(chunk.doc_id)

    def latest_aligned_chunk_for_check(
        self, check: VersionCheckResult
    ) -> KnowledgeChunk | None:
        if not check.latest_chunk_id:
            return None
        latest_chunks = self.knowledge_repository.list_chunks_for_document(check.latest_doc_id or "")
        for chunk in latest_chunks:
            if chunk.chunk_id == check.latest_chunk_id:
                return chunk
        return None

    def previous_accessible_document(
        self,
        profile: UserProfile,
        document: KnowledgeDocument,
    ) -> KnowledgeDocument | None:
        candidates = [
            candidate
            for candidate in self.knowledge_repository.find_by_topic(document.topic)
            if candidate.doc_id != document.doc_id
            and self.permission_service.can_access(profile, candidate)
        ]
        if not candidates:
            return None
        ranked = sorted(candidates, key=self._document_sort_key, reverse=True)
        return ranked[0]

    def aligned_chunk_to_document(
        self,
        source_chunk: KnowledgeChunk,
        target_document: KnowledgeDocument,
    ) -> KnowledgeChunk | None:
        aligned_chunk, _ = self._align_chunk_to_document(source_chunk, target_document)
        return aligned_chunk

    def _latest_accessible_document(
        self, profile: UserProfile, document: KnowledgeDocument
    ) -> KnowledgeDocument:
        candidates = [
            candidate
            for candidate in self.knowledge_repository.find_by_topic(document.topic)
            if self.permission_service.can_access(profile, candidate)
        ]
        if not candidates:
            return document
        latest_candidates = [candidate for candidate in candidates if candidate.is_latest]
        if latest_candidates:
            return latest_candidates[0]
        return sorted(
            candidates,
            key=lambda item: (
                1 if item.is_latest else 0,
                item.published_at.toordinal() if item.published_at else 0,
            ),
            reverse=True,
        )[0]

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

    def _align_chunk_to_document(
        self,
        source_chunk: KnowledgeChunk,
        target_document: KnowledgeDocument,
    ) -> tuple[KnowledgeChunk | None, float]:
        target_chunks = self.knowledge_repository.list_chunks_for_document(target_document.doc_id)
        if not target_chunks:
            return None, 0.0
        if source_chunk.doc_id == target_document.doc_id:
            for chunk in target_chunks:
                if chunk.chunk_id == source_chunk.chunk_id:
                    return chunk, 1.0

        exact_matches = [
            chunk
            for chunk in target_chunks
            if chunk.section_title == source_chunk.section_title
            and chunk.subsection_title == source_chunk.subsection_title
        ]
        if exact_matches:
            return exact_matches[0], 0.98

        section_matches = [
            chunk
            for chunk in target_chunks
            if chunk.section_title == source_chunk.section_title
        ]
        if section_matches:
            ranked = self._rank_similar_chunks(source_chunk, section_matches)
            return ranked[0]

        ranked = self._rank_similar_chunks(source_chunk, target_chunks)
        return ranked[0]

    def _rank_similar_chunks(
        self,
        source_chunk: KnowledgeChunk,
        candidates: list[KnowledgeChunk],
    ) -> list[tuple[KnowledgeChunk, float]]:
        source_tokens = self._tokenize(source_chunk.text)
        ranked: list[tuple[KnowledgeChunk, float]] = []
        for candidate in candidates:
            target_tokens = self._tokenize(candidate.text)
            overlap = self._jaccard_similarity(source_tokens, target_tokens)
            heading_bonus = 0.0
            if candidate.subsection_title and candidate.subsection_title == source_chunk.subsection_title:
                heading_bonus += 0.12
            if candidate.section_title and candidate.section_title == source_chunk.section_title:
                heading_bonus += 0.08
            ranked.append((candidate, min(overlap + heading_bonus, 1.0)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked or [(candidates[0], 0.0)]

    def _jaccard_similarity(self, left: list[str], right: list[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    def _tokenize(self, text: str) -> list[str]:
        import re

        tokens: list[str] = []
        for chunk in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
            if len(chunk) <= 2:
                tokens.append(chunk)
                continue
            tokens.append(chunk)
            for index in range(len(chunk) - 1):
                tokens.append(chunk[index : index + 2])
        return tokens

    def _format_chunk_hint(
        self,
        section_title: str | None,
        subsection_title: str | None,
    ) -> str | None:
        parts = [part for part in [section_title, subsection_title] if part]
        if not parts:
            return None
        return " / ".join(parts)
