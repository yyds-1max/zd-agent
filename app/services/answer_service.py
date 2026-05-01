from __future__ import annotations

import json

from app.schemas.answer_strategy import AnswerStrategyResult
from app.schemas.intent import IntentResult
from app.schemas.knowledge import Citation, KnowledgeChunk, KnowledgeDocument, RetrievedChunk
from app.schemas.llm import AgentAnswerLLMOutput
from app.schemas.query import QueryResponse
from app.schemas.user import UserProfile
from app.schemas.version import VersionCheckResult, VersionDiffResult
from app.services.main_agent_prompt import evidence_answer_block
from app.services.qwen_llm_service import QwenStructuredLLMService
from app.services.version_service import VersionService


class AnswerService:
    def __init__(
        self,
        version_service: VersionService,
        llm_service: QwenStructuredLLMService | None = None,
    ):
        self.version_service = version_service
        self.llm_service = llm_service

    def compose(
        self,
        question: str,
        profile: UserProfile,
        intent: IntentResult,
        retrieved_chunks: list[RetrievedChunk],
        version_checks: list[VersionCheckResult],
        version_diffs: list[VersionDiffResult],
        answer_strategy: AnswerStrategyResult,
        tool_trace: list[str],
    ) -> QueryResponse:
        if not retrieved_chunks:
            return QueryResponse(
                question=question,
                answer="我没有在你当前权限范围内找到可直接回答这个问题的知识片段。",
                user_profile=profile,
                intent=intent,
                citations=[],
                version_checks=[],
                version_diffs=[],
                answer_strategy=answer_strategy,
                version_notice=None,
                notes=["已执行权限过滤，但没有命中可访问 chunk。"],
                tool_trace=tool_trace,
            )

        strategy_chunks = self._select_strategy_chunks(
            retrieved_chunks=retrieved_chunks,
            answer_strategy=answer_strategy,
        )
        lead_chunk = self._select_lead_chunk(strategy_chunks)
        lead_document = self.version_service.document_for_chunk(lead_chunk)
        key_points = self._extract_key_points(
            question=question,
            intent=intent,
            retrieved_chunks=strategy_chunks,
            answer_strategy=answer_strategy,
            version_diffs=version_diffs,
        )
        version_notice = self._build_version_notice(lead_document, version_checks, answer_strategy)
        citations = self._build_citations(strategy_chunks)

        answer_text, llm_used = self._compose_answer_text(
            question=question,
            profile=profile,
            intent=intent,
            lead_chunk=lead_chunk,
            lead_document=lead_document,
            key_points=key_points,
            version_notice=version_notice,
            citations=citations,
            version_diffs=version_diffs,
            answer_strategy=answer_strategy,
        )

        notes = ["已先做权限过滤，再做 chunk 级混合检索和版本校验。"]
        final_tool_trace = list(tool_trace)
        if version_notice:
            notes.append("已按回答策略补充版本提醒。")
        if version_diffs:
            notes.append("已完成新旧版本 chunk 差异分析。")
        notes.append(f"回答策略：{answer_strategy.mode}")
        if llm_used:
            final_tool_trace.append("主Agent：已使用 qwen3-max 基于 chunk 检索证据生成最终回答。")
        else:
            final_tool_trace.append("主Agent：当前使用规则模板生成最终回答。")

        return QueryResponse(
            question=question,
            answer=answer_text,
            user_profile=profile,
            intent=intent,
            citations=citations,
            version_checks=version_checks,
            version_diffs=version_diffs,
            answer_strategy=answer_strategy,
            version_notice=version_notice,
            notes=notes,
            tool_trace=final_tool_trace,
        )

    def _compose_answer_text(
        self,
        *,
        question: str,
        profile: UserProfile,
        intent: IntentResult,
        lead_chunk: KnowledgeChunk,
        lead_document: KnowledgeDocument,
        key_points: list[str],
        version_notice: str | None,
        citations: list[Citation],
        version_diffs: list[VersionDiffResult],
        answer_strategy: AnswerStrategyResult,
    ) -> tuple[str, bool]:
        if self.llm_service is not None and self.llm_service.is_available():
            try:
                output = self._compose_with_llm(
                    question=question,
                    profile=profile,
                    intent=intent,
                    lead_chunk=lead_chunk,
                    lead_document=lead_document,
                    key_points=key_points,
                    version_notice=version_notice,
                    citations=citations,
                    version_diffs=version_diffs,
                    answer_strategy=answer_strategy,
                )
                return output.answer_markdown.strip(), True
            except Exception:
                pass
        return self._compose_with_template(
            lead_document=lead_document,
            key_points=key_points,
            version_notice=version_notice,
            version_diffs=version_diffs,
            answer_strategy=answer_strategy,
        ), False

    def _compose_with_llm(
        self,
        *,
        question: str,
        profile: UserProfile,
        intent: IntentResult,
        lead_chunk: KnowledgeChunk,
        lead_document: KnowledgeDocument,
        key_points: list[str],
        version_notice: str | None,
        citations: list[Citation],
        version_diffs: list[VersionDiffResult],
        answer_strategy: AnswerStrategyResult,
    ) -> AgentAnswerLLMOutput:
        system_prompt = (
            f"{evidence_answer_block()}"
            "输出必须是严格 JSON。"
            "不要在 answer_markdown 中输出参考文档列表，引用来源由系统另行追加。"
            "只有回答策略要求 include_version_notice 时，才简短说明版本变化。"
        )
        strategy_instruction = self._strategy_instruction(answer_strategy)
        citation_payload = [
            {
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "title": item.title,
                "doc_type": item.doc_type,
                "version": item.version,
                "is_latest": item.is_latest,
                "section_title": item.section_title,
                "subsection_title": item.subsection_title,
                "snippet": item.snippet,
                "chunk_text": item.chunk_text,
            }
            for item in citations[:4]
        ]
        diff_payload = [
            {
                "source_chunk_id": item.source_chunk_id,
                "source_doc_id": item.source_doc_id,
                "source_version": item.source_version,
                "latest_chunk_id": item.latest_chunk_id,
                "latest_doc_id": item.latest_doc_id,
                "latest_version": item.latest_version,
                "change_type": item.change_type,
                "summary": item.summary,
                "confidence": item.confidence,
                "key_changes": item.key_changes,
            }
            for item in version_diffs[:4]
        ]
        user_prompt = (
            f"用户问题:\n{question}\n\n"
            f"回答策略:\n{json.dumps(answer_strategy.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"模式说明:\n{strategy_instruction}\n\n"
            f"用户画像:\n{json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"意图识别结果:\n{json.dumps(intent.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"主参考文档:\n{json.dumps({'title': lead_document.title, 'doc_type': lead_document.doc_type, 'version': lead_document.version}, ensure_ascii=False, indent=2)}\n\n"
            f"主参考 chunk:\n{json.dumps({'chunk_id': lead_chunk.chunk_id, 'section_title': lead_chunk.section_title, 'subsection_title': lead_chunk.subsection_title, 'text': lead_chunk.text}, ensure_ascii=False, indent=2)}\n\n"
            f"规则整理出的关键点:\n{json.dumps(key_points, ensure_ascii=False, indent=2)}\n\n"
            f"版本提醒:\n{version_notice or '无'}\n\n"
            f"版本差异分析:\n{json.dumps(diff_payload, ensure_ascii=False, indent=2)}\n\n"
            f"可引用证据 chunks:\n{json.dumps(citation_payload, ensure_ascii=False, indent=2)}\n\n"
            "请输出 JSON，字段要求：\n"
            "- answer_markdown: 最终回答，中文，2-6 句，可包含短列表，必须只基于给定证据\n"
            "- cited_doc_ids: 实际参考到的 doc_id 列表\n"
            "- notes: 补充说明列表，若无可为空\n"
        )
        return self.llm_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AgentAnswerLLMOutput,
        )

    def _compose_with_template(
        self,
        *,
        lead_document: KnowledgeDocument,
        key_points: list[str],
        version_notice: str | None,
        version_diffs: list[VersionDiffResult],
        answer_strategy: AnswerStrategyResult,
    ) -> str:
        if answer_strategy.mode == "change_summary_mode":
            answer_lines = ["以下基于当前命中的新旧版本证据，直接总结关键变化："]
            for item in version_diffs[:3]:
                answer_lines.append(f"- {item.summary}")
            return "\n".join(answer_lines)

        if answer_strategy.mode == "historical_lookup_mode":
            answer_lines = [
                f"以下先按你询问的历史版本《{lead_document.title}》相关内容回答。",
            ]
        else:
            answer_lines = [
                f"按当前生效的《{lead_document.title}》：",
            ]

        if version_notice and answer_strategy.include_version_notice:
            answer_lines.append(version_notice)
        if version_diffs and answer_strategy.include_diff_summary:
            answer_lines.append("版本差异摘要：")
            for item in version_diffs[:2]:
                answer_lines.append(f"- {item.summary}")
        answer_lines.append("核心信息：")
        for point in key_points:
            answer_lines.append(f"- {point}")
        return "\n".join(answer_lines)

    def _select_lead_chunk(self, retrieved_chunks: list[RetrievedChunk]) -> KnowledgeChunk:
        return retrieved_chunks[0].chunk

    def _extract_key_points(
        self,
        question: str,
        intent: IntentResult,
        retrieved_chunks: list[RetrievedChunk],
        answer_strategy: AnswerStrategyResult,
        version_diffs: list[VersionDiffResult],
    ) -> list[str]:
        if answer_strategy.mode == "change_summary_mode" and version_diffs:
            key_points = []
            for item in version_diffs[:3]:
                key_points.append(item.summary)
                key_points.extend(item.key_changes[:2])
            return key_points[:6]

        candidate_lines: list[tuple[int, str]] = []
        for index, item in enumerate(retrieved_chunks[:4]):
            lines = [line.strip() for line in item.chunk.text.splitlines() if line.strip()]
            for line in lines:
                if self._looks_like_heading(line):
                    continue
                score = max(0, 8 - index)
                score += self._score_line(question, intent, line, item.chunk)
                if score > 0:
                    candidate_lines.append((score, line))

        if not candidate_lines:
            return [retrieved_chunks[0].chunk.text[:200]]

        candidate_lines.sort(key=lambda item: item[0], reverse=True)
        deduped: list[str] = []
        for _, line in candidate_lines:
            if line not in deduped:
                deduped.append(line)
        return deduped[:6]

    def _looks_like_heading(self, line: str) -> bool:
        text = line.strip()
        if not text:
            return True
        if len(text) <= 4 and not any(char.isdigit() for char in text):
            return True
        heading_prefixes = ("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、")
        if text.startswith(heading_prefixes) and len(text) <= 14:
            return True
        return False

    def _score_line(
        self,
        question: str,
        intent: IntentResult,
        line: str,
        chunk: KnowledgeChunk,
    ) -> int:
        score = 0
        section_text = " ".join(
            part for part in [chunk.section_title, chunk.subsection_title] if part
        )
        if intent.name == "policy_lookup":
            if "标准" in question and any(keyword in line for keyword in ["元", "标准", "费用", "住宿", "餐补", "高铁", "飞机"]):
                score += 5
            if any(word in question for word in ["报销", "流程"]) and "报销" in line:
                score += 3
            if any(word in question for word in ["最新", "版本"]) and any(marker in line for marker in ["生效", "废止", "版本"]):
                score += 2
            if "标准" in question and any(keyword in section_text for keyword in ["费用标准", "住宿费", "餐补", "交通费"]):
                score += 3
        elif intent.name == "project_lookup":
            if any(keyword in line for keyword in ["交付", "节点", "里程碑", "完成"]):
                score += 4
            if "2026-" in line:
                score += 2
        elif intent.name == "onboarding_lookup":
            if any(keyword in line for keyword in ["入职", "第一周", "订阅", "入口"]):
                score += 3

        score += sum(1 for keyword in intent.keywords if keyword and keyword in line.lower())
        return score

    def _build_version_notice(
        self,
        lead_document: KnowledgeDocument,
        version_checks: list[VersionCheckResult],
        answer_strategy: AnswerStrategyResult,
    ) -> str | None:
        if not answer_strategy.include_version_notice:
            return None
        for item in version_checks:
            if item.has_newer_version and item.source_doc_id == lead_document.doc_id:
                latest_hint = self._format_chunk_hint(
                    item.latest_section_title,
                    item.latest_subsection_title,
                )
                return (
                    f"版本提醒：当前命中的内容来自旧版文档，"
                    f"最新版为 {item.latest_version or '最新版'}（{item.latest_title}）"
                    f"{f'，对应内容位于 {latest_hint}' if latest_hint else ''}。"
                )
            if item.has_newer_version and item.latest_doc_id == lead_document.doc_id:
                latest_hint = self._format_chunk_hint(
                    item.latest_section_title,
                    item.latest_subsection_title,
                )
                return (
                    f"版本提醒：系统检测到命中的旧版内容，已优先定位到 "
                    f"{item.latest_version or '最新版'}（{item.latest_title}）"
                    f"{f' 的对应片段 {latest_hint}' if latest_hint else ''}。"
                )
        return None

    def _select_strategy_chunks(
        self,
        *,
        retrieved_chunks: list[RetrievedChunk],
        answer_strategy: AnswerStrategyResult,
    ) -> list[RetrievedChunk]:
        preferred_doc_id = answer_strategy.preferred_doc_id
        if answer_strategy.mode in {"current_policy_mode", "historical_lookup_mode"} and preferred_doc_id:
            filtered = [
                item
                for item in retrieved_chunks
                if item.chunk.doc_id == preferred_doc_id
            ]
            if filtered:
                return filtered
        if answer_strategy.mode == "current_policy_mode":
            latest_only = [item for item in retrieved_chunks if item.chunk.is_latest]
            if latest_only:
                return latest_only
        return retrieved_chunks

    def _strategy_instruction(self, answer_strategy: AnswerStrategyResult) -> str:
        if answer_strategy.mode == "current_policy_mode":
            return (
                "以当前生效版本为准回答，优先回答用户真正关注的现行规则。"
                "如果存在旧版命中，只做简明版本提醒，并在必要时补一句关键变化。"
            )
        if answer_strategy.mode == "historical_lookup_mode":
            return (
                "先回答旧版内容，再明确告诉用户现行版本及关键变化。"
                "必须清楚区分旧版内容和新版内容，不能混写。"
            )
        if answer_strategy.mode == "change_summary_mode":
            return (
                "不要展开普通问答，直接围绕新旧版本差异作答。"
                "优先使用 diff 结果中的 change_type、summary 和 key_changes。"
            )
        return "按常规问答方式回答，优先使用最高相关的检索证据。"

    def _format_chunk_hint(
        self,
        section_title: str | None,
        subsection_title: str | None,
    ) -> str | None:
        parts = [part for part in [section_title, subsection_title] if part]
        if not parts:
            return None
        return " / ".join(parts)

    def _build_citations(
        self, retrieved_chunks: list[RetrievedChunk]
    ) -> list[Citation]:
        citations: list[Citation] = []
        for item in retrieved_chunks:
            chunk = item.chunk
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    doc_id=chunk.doc_id,
                    title=chunk.doc_title,
                    doc_type=chunk.doc_type,
                    version=chunk.version,
                    permission_level=chunk.permission_level,
                    published_at=chunk.published_at,
                    is_latest=chunk.is_latest,
                    section_title=chunk.section_title,
                    subsection_title=chunk.subsection_title,
                    score=item.final_score,
                    snippet=item.snippet,
                    chunk_text=chunk.text,
                    source_path=chunk.source_path,
                )
            )
        return citations
