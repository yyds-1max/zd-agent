from __future__ import annotations

from app.repositories.knowledge_repository import KnowledgeRepository
from app.schemas.intent import IntentResult
from app.schemas.knowledge import KnowledgeChunk, KnowledgeDocument, RetrievedChunk
from app.schemas.user import UserProfile
from app.schemas.version import VersionCheckResult, VersionDiffResult
from app.services.feishu_contact_service import ContactService
from app.services.intent_parser_service import IntentParserService
from app.services.permission_service import PermissionService
from app.services.retrieval_service import RetrievalService
from app.services.user_profile_llm_service import UserProfileLLMService
from app.services.version_diff_service import VersionDiffService
from app.services.version_service import VersionService


class UserProfileTool:
    """Loads user context and lightweight intent hints without owning workflow decisions."""

    def __init__(
        self,
        feishu_contact_service: ContactService,
        intent_parser_service: IntentParserService,
        user_profile_llm_service: UserProfileLLMService | None = None,
    ):
        self.feishu_contact_service = feishu_contact_service
        self.intent_parser_service = intent_parser_service
        self.user_profile_llm_service = user_profile_llm_service

    def run(
        self,
        user_id: str,
        question: str,
        user_id_type: str = "open_id",
    ) -> tuple[UserProfile, IntentResult, str]:
        base_user = self.feishu_contact_service.get_user_profile(user_id, user_id_type)
        profile_source = "飞书 API" if base_user.source == "feishu_contact_api" else "本地飞书目录"
        trace_prefix = f"用户画像工具：已从{profile_source}补齐部门/职级/岗位，"
        if self.user_profile_llm_service is not None and self.user_profile_llm_service.is_available():
            try:
                intent = self.user_profile_llm_service.understand(question, base_user)
                trace = trace_prefix + "并使用 qwen3-max 完成项目抽取与意图识别。"
            except Exception:
                intent = self.intent_parser_service.parse(question)
                trace = trace_prefix + "qwen3-max 调用失败，已回退到规则抽取与意图识别。"
        else:
            intent = self.intent_parser_service.parse(question)
            trace = trace_prefix + "并结合规则抽取完成项目与意图识别。"
        profile = UserProfile(
            **base_user.model_dump(),
            project_mentions=intent.project_names,
            active_projects=sorted(
                set(base_user.projects) | set(base_user.managed_projects) | set(intent.project_names)
            ),
            intent_hint=intent.name,
        )
        return profile, intent, trace


class KnowledgeRetrievalTool:
    """Runs permission-aware retrieval and returns evidence candidates."""

    def __init__(
        self,
        knowledge_repository: KnowledgeRepository,
        permission_service: PermissionService,
        retrieval_service: RetrievalService,
    ):
        self.knowledge_repository = knowledge_repository
        self.permission_service = permission_service
        self.retrieval_service = retrieval_service

    def run(
        self,
        question: str,
        profile: UserProfile,
        intent: IntentResult,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], str]:
        accessible_documents, accessible_chunks = self._accessible_scope(profile)
        retrieved = self.retrieval_service.search(
            question=question,
            intent=intent,
            documents=accessible_documents,
            top_k=top_k,
            chunks=accessible_chunks,
        )
        trace = (
            f"检索工具：在 {len(accessible_documents)} 份可访问文档、"
            f"{len(accessible_chunks)} 个可访问 chunks 上执行混合检索，"
            f"返回 {len(retrieved)} 条候选。后端={self.retrieval_service.describe_backend()}。"
        )
        return retrieved, trace

    def run_supplemental(
        self,
        queries: list[str],
        existing_chunks: list[RetrievedChunk],
        profile: UserProfile,
        intent: IntentResult,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], str]:
        if not queries:
            return existing_chunks, "补充检索工具：没有补充 query，已跳过。"

        merged: list[RetrievedChunk] = list(existing_chunks)
        seen_chunk_ids = {item.chunk.chunk_id for item in merged}
        added = 0
        accessible_documents, accessible_chunks = self._accessible_scope(profile)
        for query in queries:
            retrieved = self.retrieval_service.search(
                question=query,
                intent=intent,
                documents=accessible_documents,
                top_k=top_k,
                chunks=accessible_chunks,
            )
            for item in retrieved:
                if item.chunk.chunk_id in seen_chunk_ids:
                    continue
                merged.append(item)
                seen_chunk_ids.add(item.chunk.chunk_id)
                added += 1
        trace = (
            f"补充检索工具：执行 {len(queries)} 条补充 query，"
            f"新增 {added} 条候选，合并后共 {len(merged)} 条。"
        )
        return merged[: max(top_k * 3, top_k)], trace

    def _accessible_scope(
        self,
        profile: UserProfile,
    ) -> tuple[list[KnowledgeDocument], list[KnowledgeChunk]]:
        accessible_documents = self.permission_service.filter_accessible(
            profile, self.knowledge_repository.list_documents()
        )
        accessible_chunks = self.knowledge_repository.list_chunks_for_documents(accessible_documents)
        return accessible_documents, accessible_chunks


class LatestVersionTool:
    """Checks whether retrieved evidence has newer accessible document versions."""

    def __init__(self, version_service: VersionService):
        self.version_service = version_service

    def run(
        self,
        profile: UserProfile,
        retrieved_chunks: list[RetrievedChunk],
    ) -> tuple[list[VersionCheckResult], str]:
        version_checks = self.version_service.check_versions(profile, retrieved_chunks)
        switched = sum(1 for item in version_checks if item.has_newer_version)
        aligned = sum(1 for item in version_checks if item.has_newer_version and item.latest_chunk_id)
        trace = (
            f"最新文件工具：完成 {len(version_checks)} 条版本检查，"
            f"发现 {switched} 条可升级命中，定位到 {aligned} 条新版对应 chunks。"
        )
        return version_checks, trace


class VersionDiffTool:
    """Compares aligned document chunks when the main agent requests version analysis."""

    def __init__(
        self,
        version_service: VersionService,
        version_diff_service: VersionDiffService,
    ):
        self.version_service = version_service
        self.version_diff_service = version_diff_service

    def run(
        self,
        question: str,
        profile: UserProfile,
        retrieved_chunks: list[RetrievedChunk],
        version_checks: list[VersionCheckResult],
    ) -> tuple[list[VersionDiffResult], str]:
        chunk_map = {item.chunk.chunk_id: item.chunk for item in retrieved_chunks}
        version_diffs: list[VersionDiffResult] = []
        for check in version_checks:
            if not check.source_chunk_id:
                continue
            source_chunk = chunk_map.get(check.source_chunk_id)
            if source_chunk is None:
                continue

            if check.has_newer_version:
                latest_chunk = self.version_service.latest_aligned_chunk_for_check(check)
                version_diffs.append(
                    self.version_diff_service.compare(
                        old_chunk=source_chunk,
                        new_chunk=latest_chunk,
                        question=question,
                    )
                )
                continue

            previous_document = self.version_service.previous_accessible_document(
                profile,
                self.version_service.document_for_chunk(source_chunk),
            )
            if previous_document is None:
                continue
            previous_chunk = self.version_service.aligned_chunk_to_document(
                source_chunk,
                previous_document,
            )
            if previous_chunk is None:
                continue
            version_diffs.append(
                self.version_diff_service.compare(
                    old_chunk=previous_chunk,
                    new_chunk=source_chunk,
                    question=question,
                )
            )
        trace = f"版本差异工具：完成 {len(version_diffs)} 条新旧内容差异分析。"
        return version_diffs, trace
