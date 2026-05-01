from __future__ import annotations

import json

from app.schemas.agent_loop import AgentActionDecision, AgentObservation
from app.schemas.intent import IntentResult
from app.schemas.knowledge import RetrievedChunk
from app.schemas.llm import AgentActionLLMOutput
from app.schemas.user import UserProfile
from app.services.qwen_llm_service import QwenStructuredLLMService


class AgentControllerService:
    ALLOWED_ACTIONS = {"retrieve", "finalize"}
    COMPLEX_TOKENS = [
        "对比",
        "比较",
        "差异",
        "区别",
        "变化",
        "更新",
        "旧版",
        "新版",
        "历史",
        "最近",
        "需要注意",
        "注意什么",
        "影响",
        "适用",
        "同时",
        "分别",
    ]

    def __init__(
        self,
        llm_service: QwenStructuredLLMService | None = None,
        max_retrieval_iterations: int = 2,
        min_confidence: float = 0.72,
    ):
        self.llm_service = llm_service
        self.max_retrieval_iterations = max_retrieval_iterations
        self.min_confidence = min_confidence

    def decide(
        self,
        *,
        question: str,
        profile: UserProfile,
        intent: IntentResult,
        retrieved_chunks: list[RetrievedChunk],
        observations: list[AgentObservation],
        retrieval_iterations: int,
    ) -> AgentActionDecision:
        if retrieval_iterations >= self.max_retrieval_iterations:
            return AgentActionDecision(
                action="finalize",
                evidence_sufficient=True,
                thought_summary="已达到补充检索轮数上限，进入回答生成。",
                reason=f"最多允许 {self.max_retrieval_iterations} 轮补充检索，当前已达到上限。",
                confidence=1.0,
                source="guardrail",
            )

        llm_decision = self._decide_with_llm(
            question=question,
            profile=profile,
            intent=intent,
            retrieved_chunks=retrieved_chunks,
            observations=observations,
            retrieval_iterations=retrieval_iterations,
        )
        if llm_decision is not None:
            return llm_decision

        return self._decide_with_rules(
            question=question,
            intent=intent,
            retrieved_chunks=retrieved_chunks,
            observations=observations,
            retrieval_iterations=retrieval_iterations,
        )

    def _decide_with_llm(
        self,
        *,
        question: str,
        profile: UserProfile,
        intent: IntentResult,
        retrieved_chunks: list[RetrievedChunk],
        observations: list[AgentObservation],
        retrieval_iterations: int,
    ) -> AgentActionDecision | None:
        if self.llm_service is None or not self.llm_service.is_available():
            return None

        evidence_payload = [
            {
                "title": item.chunk.doc_title,
                "doc_type": item.chunk.doc_type,
                "topic": item.chunk.topic,
                "version": item.chunk.version,
                "is_latest": item.chunk.is_latest,
                "section_title": item.chunk.section_title,
                "subsection_title": item.chunk.subsection_title,
                "snippet": item.snippet,
            }
            for item in retrieved_chunks[:8]
        ]
        try:
            output = self.llm_service.generate_structured(
                system_prompt=(
                    "你是企业知识问答 Agent 的控制器。"
                    "你要根据用户问题、已有证据和历史 observation，决定下一步 action。"
                    "允许的 action 只有 retrieve 和 finalize。"
                    "retrieve 表示还需要补充检索，必须给出 action_query。"
                    "finalize 表示证据已经足够或继续检索收益不高。"
                    "不要回答用户问题，不要编造企业事实。"
                    "只输出严格 JSON。"
                    "thought_summary 只写简短决策摘要，不要输出详细推理链。"
                ),
                user_prompt=(
                    f"最大补充检索轮数：{self.max_retrieval_iterations}\n"
                    f"已补充检索轮数：{retrieval_iterations}\n\n"
                    f"用户问题:\n{question}\n\n"
                    f"用户画像:\n{json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)}\n\n"
                    f"意图:\n{json.dumps(intent.model_dump(), ensure_ascii=False, indent=2)}\n\n"
                    f"已有证据摘要:\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}\n\n"
                    f"历史 observations:\n{json.dumps([item.model_dump() for item in observations], ensure_ascii=False, indent=2)}\n\n"
                    "请输出 JSON 字段：\n"
                    "- action: retrieve 或 finalize\n"
                    "- action_query: action=retrieve 时的中文检索 query，否则为 null\n"
                    "- evidence_sufficient: 当前证据是否足够进入回答\n"
                    "- thought_summary: 一句话决策摘要\n"
                    "- reason: 简短原因\n"
                    "- confidence: 0 到 1\n"
                ),
                response_model=AgentActionLLMOutput,
            )
        except Exception:
            return None

        if output.confidence < self.min_confidence:
            return None
        action = output.action.strip()
        if action not in self.ALLOWED_ACTIONS:
            return None
        if action == "retrieve" and not (output.action_query or "").strip():
            return None
        if action == "retrieve" and self._query_was_tried(output.action_query or "", observations):
            return AgentActionDecision(
                action="finalize",
                evidence_sufficient=True,
                thought_summary="模型建议的补充检索 query 已经尝试过，停止循环。",
                reason="避免重复检索同一 query。",
                confidence=0.9,
                source="guardrail",
            )
        return AgentActionDecision(
            action=action,
            action_query=(output.action_query or "").strip() or None,
            evidence_sufficient=output.evidence_sufficient,
            thought_summary=output.thought_summary,
            reason=output.reason,
            confidence=output.confidence,
            source="llm",
        )

    def _decide_with_rules(
        self,
        *,
        question: str,
        intent: IntentResult,
        retrieved_chunks: list[RetrievedChunk],
        observations: list[AgentObservation],
        retrieval_iterations: int,
    ) -> AgentActionDecision:
        if not retrieved_chunks:
            query = self._dedupe_query(question, observations)
            if query:
                return AgentActionDecision(
                    action="retrieve",
                    action_query=query,
                    evidence_sufficient=False,
                    thought_summary="首轮没有证据，尝试用原问题补充检索。",
                    reason="当前没有可用证据。",
                    confidence=0.82,
                    source="rule",
                )

        if not any(token in question for token in self.COMPLEX_TOKENS):
            return AgentActionDecision(
                action="finalize",
                evidence_sufficient=True,
                thought_summary="问题较直接，现有证据可以进入回答生成。",
                reason="未检测到多跳补查信号。",
                confidence=0.72,
                source="rule",
            )

        candidate_queries = self._candidate_queries(question, intent)
        for query in candidate_queries:
            if not self._query_was_tried(query, observations):
                return AgentActionDecision(
                    action="retrieve",
                    action_query=query,
                    evidence_sufficient=False,
                    thought_summary="复杂问题需要补齐一个证据面。",
                    reason="问题包含版本、变化、最近或注意事项等多跳信号。",
                    confidence=0.82,
                    source="rule",
                )

        return AgentActionDecision(
            action="finalize",
            evidence_sufficient=True,
            thought_summary="可生成的补充 query 均已尝试，进入回答生成。",
            reason="没有新的补充检索 query。",
            confidence=0.78,
            source="rule",
        )

    def _candidate_queries(self, question: str, intent: IntentResult) -> list[str]:
        subject = self._query_subject(question, intent)
        queries: list[str] = []
        if any(token in question for token in ["变化", "更新", "差异", "区别", "对比", "比较"]):
            queries.append(f"{subject} 最新版 变化 更新")
            queries.append(f"{subject} 旧版 历史版本 差异")
        if any(token in question for token in ["最近", "上周", "本周", "这个月"]):
            queries.append(f"{subject} 最近 更新 周报 结论")
        if any(token in question for token in ["注意", "影响", "适用"]):
            queries.append(f"{subject} 适用范围 注意事项 影响")
        return [query for query in queries if query.strip()]

    def _query_subject(self, question: str, intent: IntentResult) -> str:
        if intent.project_names:
            return " ".join(intent.project_names)
        if intent.keywords:
            return " ".join(intent.keywords[:4])
        return question

    def _dedupe_query(
        self,
        query: str,
        observations: list[AgentObservation],
    ) -> str | None:
        text = " ".join(query.split()).strip()
        if not text or self._query_was_tried(text, observations):
            return None
        return text

    def _query_was_tried(
        self,
        query: str,
        observations: list[AgentObservation],
    ) -> bool:
        normalized = " ".join(query.split()).strip()
        return any(item.query == normalized for item in observations if item.query)
