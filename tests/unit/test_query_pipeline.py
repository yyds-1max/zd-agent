from __future__ import annotations

from app.pipelines.query_pipeline import QueryPipeline
from app.pipelines.query_pipeline import build_query_pipeline
from app.schemas.llm import ConversationIntentLLMOutput, ConversationRewriteLLMOutput
from app.schemas.query import QueryRequest
from app.schemas.query import QueryResponse
from app.schemas.intent import IntentResult
from app.schemas.user import UserProfile
from app.repositories.conversation_repository import ConversationRepository
from app.services.conversation_intent_service import ConversationIntentService
from app.services.conversation_memory_service import ConversationMemoryService
from app.services.conversation_rewrite_service import ConversationRewriteService
from app.services.main_agent_service import KnowledgeDispatchMainAgent
from app.services.task_router_service import TaskRouterService


class _StaticUserProfileTool:
    def run(self, user_id, question, user_id_type="open_id"):
        return (
            UserProfile(
                user_id=user_id,
                user_id_type=user_id_type,
                name="测试用户",
                department="测试部门",
                title="测试岗位",
                level="P1",
                role="employee",
            ),
            IntentResult(
                name="general_knowledge",
                confidence=0.8,
                reasoning="测试画像工具。",
            ),
            "用户画像工具：测试画像。",
        )


class _UnexpectedTool:
    def run(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("RAG tools should not be called for direct conversation intent.")

    def run_supplemental(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Supplemental retrieval should not be called for direct conversation intent.")


class _UnexpectedAgentControllerService:
    def decide(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Agent controller should not be called for direct conversation intent.")


class _UnexpectedAnswerStrategyRouterService:
    def route(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Answer strategy router should not be called for direct conversation intent.")

    def should_run_version_diff(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Version diff routing should not be called for direct conversation intent.")


class _EchoMainAgent:
    def run(self, request):
        return QueryResponse(
            question=request.question,
            answer=f"answer for: {request.question}",
            conversation_id=request.conversation_id,
            user_profile=UserProfile(
                user_id=request.user_id,
                user_id_type=request.user_id_type,
                name="测试用户",
                department="测试部门",
                title="测试岗位",
                level="P1",
                role="employee",
            ),
            intent=IntentResult(
                name="general_knowledge",
                confidence=0.8,
                reasoning="测试主 Agent。",
            ),
            citations=[],
            version_checks=[],
            version_diffs=[],
            notes=[],
            tool_trace=[],
        )


def _build_direct_main_agent(
    conversation_intent_service: ConversationIntentService | None = None,
) -> KnowledgeDispatchMainAgent:
    return KnowledgeDispatchMainAgent(
        user_profile_tool=_StaticUserProfileTool(),
        task_router_service=TaskRouterService(
            conversation_intent_service=conversation_intent_service
            or ConversationIntentService(),
        ),
        retrieval_tool=_UnexpectedTool(),
        agent_controller_service=_UnexpectedAgentControllerService(),
        latest_version_tool=_UnexpectedTool(),
        version_diff_tool=_UnexpectedTool(),
        answer_strategy_router_service=_UnexpectedAnswerStrategyRouterService(),
        answer_service=_UnexpectedTool(),
    )


class _FakeConversationRouterLLM:
    def is_available(self) -> bool:
        return True

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        assert response_model is ConversationIntentLLMOutput
        return ConversationIntentLLMOutput(
            intent_name="capability",
            confidence=0.94,
            should_retrieve=False,
            direct_answer="你好，我可以帮你查询企业制度、项目资料、FAQ 和文档版本变化。",
            reasoning="用户询问助手功能。",
        )


class _FakeConversationRewriteLLM:
    def is_available(self) -> bool:
        return True

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        assert response_model is ConversationRewriteLLMOutput
        assert "那销售部门适用吗？" in user_prompt
        return ConversationRewriteLLMOutput(
            standalone_question="差旅报销制度 V2.0 是否适用于销售部门，销售部门是否有特殊规则？",
            needs_clarification=False,
            referenced_topic="差旅报销制度 V2.0",
            rewrite_reason="当前问题承接上一轮差旅报销制度主题，并追问销售部门适用范围。",
            confidence=0.91,
        )


def test_employee_gets_latest_travel_policy() -> None:
    pipeline = build_query_pipeline()

    response = pipeline.run(
        QueryRequest(user_id="u_employee_li", question="出差报销最新标准是什么？")
    )

    assert response.answer_strategy is not None
    assert response.answer_strategy.mode == "current_policy_mode"
    assert response.version_notice is None
    assert "V2.0" in response.answer
    assert "10" in response.answer
    assert any(citation.version == "V2.0" for citation in response.citations)
    assert all("财务" not in citation.permission_level for citation in response.citations)
    assert any("Agent控制器：action=finalize" in trace for trace in response.tool_trace)


def test_greeting_uses_direct_conversation_reply() -> None:
    pipeline = build_query_pipeline()

    response = pipeline.run(
        QueryRequest(user_id="u_employee_li", question="你好，你是谁？")
    )

    assert response.answer_strategy is not None
    assert response.answer_strategy.mode == "direct_conversation_mode"
    assert response.intent.name in {"greeting", "self_intro"}
    assert "知达Agent" in response.answer
    assert response.version_notice is None
    assert response.citations == []


def test_llm_conversation_router_handles_capability_question() -> None:
    pipeline = QueryPipeline(
        _build_direct_main_agent(
            ConversationIntentService(
                llm_service=_FakeConversationRouterLLM(),
            )
        ),
    )

    response = pipeline.run(
        QueryRequest(user_id="u_employee_li", question="你好，你有什么功能？")
    )

    assert response.answer_strategy is not None
    assert response.answer_strategy.mode == "direct_conversation_mode"
    assert response.intent.name == "capability"
    assert "企业制度" in response.answer
    assert response.citations == []
    assert any("source=llm" in trace for trace in response.tool_trace)


def test_business_action_request_does_not_enter_rag() -> None:
    pipeline = QueryPipeline(_build_direct_main_agent())

    response = pipeline.run(
        QueryRequest(user_id="u_employee_li", question="帮我订阅报销制度更新，有变化提醒我")
    )

    assert response.answer_strategy is not None
    assert response.answer_strategy.mode == "business_action_mode"
    assert response.intent.name == "business_action"
    assert "还没有接入真实操作 API" in response.answer
    assert response.citations == []
    assert any("route=business_action" in trace for trace in response.tool_trace)


def test_follow_up_question_uses_conversation_history(tmp_path) -> None:
    memory = ConversationMemoryService(
        ConversationRepository(tmp_path / "conversations.json"),
    )
    pipeline = QueryPipeline(
        _EchoMainAgent(),
        conversation_memory_service=memory,
    )

    first = pipeline.run(
        QueryRequest(
            user_id="u_employee_li",
            question="出差报销最新标准是什么？",
            conversation_id="conv-test",
        )
    )
    second = pipeline.run(
        QueryRequest(
            user_id="u_employee_li",
            question="那旧版呢？",
            conversation_id="conv-test",
        )
    )

    assert first.conversation_id == "conv-test"
    assert second.question == "那旧版呢？"
    assert second.contextual_question is not None
    assert "上一轮用户问题：出差报销最新标准是什么？" in second.contextual_question
    turns = memory.load_recent(conversation_id="conv-test", limit=10)
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]


def test_follow_up_question_can_use_llm_rewrite(tmp_path) -> None:
    memory = ConversationMemoryService(
        ConversationRepository(tmp_path / "conversations.json"),
        rewrite_service=ConversationRewriteService(
            llm_service=_FakeConversationRewriteLLM(),
        ),
    )
    pipeline = QueryPipeline(
        _EchoMainAgent(),
        conversation_memory_service=memory,
    )

    pipeline.run(
        QueryRequest(
            user_id="u_employee_li",
            question="出差报销最新标准是什么？",
            conversation_id="conv-llm-rewrite",
        )
    )
    response = pipeline.run(
        QueryRequest(
            user_id="u_employee_li",
            question="那销售部门适用吗？",
            conversation_id="conv-llm-rewrite",
        )
    )

    assert response.contextual_question == "差旅报销制度 V2.0 是否适用于销售部门，销售部门是否有特殊规则？"
    assert "已结合会话历史改写当前追问" in response.tool_trace[0]


def test_document_analysis_request_routes_to_agentic_rag() -> None:
    pipeline = build_query_pipeline()

    response = pipeline.run(
        QueryRequest(user_id="u_employee_li", question="旧版差旅报销制度和新版有什么区别？我需要注意什么？")
    )

    assert any("route=document_analysis" in trace for trace in response.tool_trace)
    assert response.answer_strategy is not None
    assert response.answer_strategy.mode in {"change_summary_mode", "historical_lookup_mode"}
    assert response.version_diffs
    assert any("Agent控制器：action=retrieve" in trace for trace in response.tool_trace)
    assert any("补充检索工具：执行" in trace for trace in response.tool_trace)


def test_finance_can_reach_finance_only_knowledge() -> None:
    pipeline = build_query_pipeline()

    response = pipeline.run(
        QueryRequest(user_id="u_finance_wang", question="报销超时提交如何处理？")
    )

    titles = [citation.title for citation in response.citations]
    assert any("财务报销 FAQ" in title for title in titles)


def test_pm_can_search_project_delivery_nodes() -> None:
    pipeline = build_query_pipeline()

    response = pipeline.run(
        QueryRequest(user_id="u_pm_zhou", question="项目北极星上周确认的交付节点是什么？")
    )

    assert any("交付节点说明" in citation.title for citation in response.citations)
    assert "2026-04-14" in response.answer or "2026-04-15" in response.answer


def test_old_version_query_triggers_version_notice() -> None:
    pipeline = build_query_pipeline()

    response = pipeline.run(
        QueryRequest(user_id="u_employee_li", question="旧版差旅报销制度V1现在还适用吗？")
    )

    assert response.answer_strategy is not None
    assert response.answer_strategy.mode == "historical_lookup_mode"
    assert response.version_notice is not None
    assert "V2.0" in response.version_notice
    assert any(check.latest_chunk_id is not None for check in response.version_checks if check.has_newer_version)
    assert response.version_diffs
