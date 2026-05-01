from __future__ import annotations

from app.schemas.answer_strategy import AnswerStrategyResult
from app.schemas.intent import IntentResult
from app.schemas.knowledge import RetrievedChunk
from app.schemas.version import VersionCheckResult, VersionDiffResult


class AnswerStrategyRouterService:
    CHANGE_TOKENS = ["差异", "区别", "改动", "变化", "更新", "新增", "删除", "修改", "调整"]
    OLD_VERSION_TOKENS = ["旧版", "历史", "之前", "原来", "v1", "v2", "老版本", "旧文档"]
    CURRENT_VERSION_TOKENS = ["最新", "当前", "现在", "现行", "生效", "最新版"]

    def route(
        self,
        *,
        question: str,
        intent: IntentResult,
        retrieved_chunks: list[RetrievedChunk],
        version_checks: list[VersionCheckResult],
        version_diffs: list[VersionDiffResult],
    ) -> AnswerStrategyResult:
        question_lower = question.lower()
        upgraded_checks = [item for item in version_checks if item.has_newer_version]
        asks_change = self._asks_change(question)
        asks_old = self._asks_old_version(question_lower)
        asks_current = self._asks_current_version(question)

        if asks_change and version_diffs:
            first_diff = version_diffs[0]
            return AnswerStrategyResult(
                mode="change_summary_mode",
                reason="用户明确关注版本变化，且当前已有新旧 chunk 差异结果。",
                preferred_doc_id=first_diff.latest_doc_id or first_diff.source_doc_id,
                compare_doc_id=first_diff.source_doc_id,
                use_latest_as_primary=True,
                include_version_notice=True,
                include_diff_summary=True,
                answer_old_first=False,
            )

        if asks_old and (upgraded_checks or version_diffs):
            if upgraded_checks:
                first_check = upgraded_checks[0]
                preferred_doc_id = first_check.source_doc_id
                compare_doc_id = first_check.latest_doc_id
            else:
                first_diff = version_diffs[0]
                preferred_doc_id = first_diff.source_doc_id
                compare_doc_id = first_diff.latest_doc_id
            return AnswerStrategyResult(
                mode="historical_lookup_mode",
                reason="用户明确询问旧版或历史口径，适合先回答旧版内容，再补充新版变化。",
                preferred_doc_id=preferred_doc_id,
                compare_doc_id=compare_doc_id,
                use_latest_as_primary=False,
                include_version_notice=True,
                include_diff_summary=True,
                answer_old_first=True,
            )

        if intent.name == "policy_lookup" and (asks_current or upgraded_checks or intent.version_sensitive):
            preferred_doc_id = None
            if upgraded_checks:
                preferred_doc_id = upgraded_checks[0].latest_doc_id
            else:
                latest_chunk = next((item.chunk for item in retrieved_chunks if item.chunk.is_latest), None)
                preferred_doc_id = latest_chunk.doc_id if latest_chunk else None
            return AnswerStrategyResult(
                mode="current_policy_mode",
                reason="制度类问题默认以当前生效版本为准；用户询问最新或当前口径时不额外展示版本提醒。",
                preferred_doc_id=preferred_doc_id,
                compare_doc_id=upgraded_checks[0].source_doc_id if upgraded_checks else None,
                use_latest_as_primary=True,
                include_version_notice=not asks_current and bool(upgraded_checks),
                include_diff_summary=not asks_current and bool(version_diffs),
                answer_old_first=False,
            )

        return AnswerStrategyResult(
            mode="general_answer_mode",
            reason="当前问题不属于版本变化优先场景，按常规检索结果组织回答。",
            preferred_doc_id=retrieved_chunks[0].chunk.doc_id if retrieved_chunks else None,
            compare_doc_id=None,
            use_latest_as_primary=False,
            include_version_notice=False,
            include_diff_summary=False,
            answer_old_first=False,
        )

    def should_run_version_diff(
        self,
        *,
        question: str,
        intent: IntentResult,
        version_checks: list[VersionCheckResult],
    ) -> bool:
        if not version_checks:
            return False

        question_lower = question.lower()
        if self._asks_change(question):
            return True
        if self._asks_old_version(question_lower):
            return True
        if intent.version_sensitive and intent.name in {"project_lookup", "policy_lookup"}:
            return any(token in question for token in ["对比", "相比", "注意什么", "有什么变化"])
        return False

    def _asks_change(self, question: str) -> bool:
        return any(token in question for token in self.CHANGE_TOKENS)

    def _asks_old_version(self, question_lower: str) -> bool:
        return any(token in question_lower for token in self.OLD_VERSION_TOKENS)

    def _asks_current_version(self, question: str) -> bool:
        return any(token in question for token in self.CURRENT_VERSION_TOKENS)
