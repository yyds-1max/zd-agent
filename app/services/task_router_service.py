from __future__ import annotations

import re

from app.schemas.llm import TaskRouterLLMOutput
from app.schemas.task_route import TaskRouteResult
from app.services.conversation_intent_service import ConversationIntentService
from app.services.main_agent_prompt import direct_reply_block
from app.services.qwen_llm_service import QwenStructuredLLMService


class TaskRouterService:
    RETRIEVAL_ROUTES = {"permission_rag", "document_analysis"}
    DIRECT_ROUTES = {"direct_conversation", "business_action", "clarify"}

    BUSINESS_ACTION_TOKENS = [
        "订阅",
        "取消订阅",
        "关注",
        "取消关注",
        "推送",
        "提醒",
        "创建提醒",
        "设置提醒",
        "通知我",
        "有更新告诉我",
    ]
    DOCUMENT_ANALYSIS_TOKENS = [
        "总结",
        "概括",
        "对比",
        "比较",
        "差异",
        "区别",
        "变化",
        "更新",
        "新版",
        "旧版",
        "历史版本",
        "版本差异",
        "有什么不同",
        "注意什么",
    ]
    KNOWLEDGE_TOKENS = [
        "制度",
        "政策",
        "流程",
        "报销",
        "差旅",
        "年假",
        "项目",
        "文档",
        "FAQ",
        "常见问题",
        "入职",
        "办公",
        "权限",
        "标准",
        "规则",
    ]

    def __init__(
        self,
        llm_service: QwenStructuredLLMService | None = None,
        conversation_intent_service: ConversationIntentService | None = None,
        min_confidence: float = 0.72,
    ):
        self.llm_service = llm_service
        self.conversation_intent_service = (
            conversation_intent_service or ConversationIntentService()
        )
        self.min_confidence = min_confidence

    def route(self, question: str) -> TaskRouteResult:
        normalized = self._normalize(question)
        if not normalized:
            return self._direct_route(
                route_name="clarify",
                intent_name="empty_message",
                answer="我在，你可以直接告诉我想查询的制度、项目资料或办公问题。",
                reasoning="用户输入为空，需要引导补充。",
            )

        llm_route = self._route_with_llm(question)
        if llm_route is not None:
            return llm_route

        direct_reply = self.conversation_intent_service.direct_reply(question)
        if direct_reply is not None:
            return self._direct_route(
                route_name="direct_conversation",
                intent_name=direct_reply.intent_name,
                answer=direct_reply.answer,
                confidence=direct_reply.confidence,
                source=direct_reply.source,
                reasoning="命中轻量对话意图，未进入知识库检索。",
            )

        if self._is_business_action(question):
            return self._direct_route(
                route_name="business_action",
                intent_name="business_action",
                answer=(
                    "这个请求需要调用订阅、推送或提醒类业务接口。"
                    "当前版本还没有接入真实操作 API，我先不替你确认已完成，避免造成误解。"
                ),
                reasoning="用户请求包含订阅、推送或提醒等操作意图。",
            )

        if self._is_document_analysis(question):
            return TaskRouteResult(
                route_name="document_analysis",
                intent_name="document_analysis",
                confidence=0.82,
                should_retrieve=True,
                source="rule",
                reasoning="用户需要文档总结、对比或版本差异分析，进入工具增强 RAG。",
            )

        if self._is_knowledge_question(question):
            return TaskRouteResult(
                route_name="permission_rag",
                intent_name="business_question",
                confidence=0.78,
                should_retrieve=True,
                source="rule",
                reasoning="用户询问企业制度、项目、FAQ 或文档内容，进入权限过滤 RAG。",
            )

        if len(normalized) <= 4:
            return self._direct_route(
                route_name="clarify",
                intent_name="unclear",
                answer="你可以再补充一下想查的制度、项目或文档范围，我再帮你找。",
                reasoning="用户输入较短且未命中企业知识关键词。",
            )

        return TaskRouteResult(
            route_name="permission_rag",
            intent_name="business_question",
            confidence=0.55,
            should_retrieve=True,
            source="rule",
            reasoning="未能高置信分类，降级进入权限过滤 RAG，避免漏答企业知识问题。",
        )

    def _route_with_llm(self, question: str) -> TaskRouteResult | None:
        if self.llm_service is None or not self.llm_service.is_available():
            return None

        try:
            output = self.llm_service.generate_structured(
                system_prompt=(
                    f"{direct_reply_block()}"
                    "你当前承担的是主Agent的任务路由职责。"
                    "输出必须是严格 JSON。"
                    "route_name 只能是 direct_conversation、permission_rag、"
                    "document_analysis、business_action、clarify。"
                    "闲聊、自我介绍、功能说明、感谢走 direct_conversation。"
                    "企业制度、项目、FAQ、普通文档问答走 permission_rag。"
                    "文档总结、文档对比、版本差异、新旧版变化走 document_analysis。"
                    "订阅、推送、创建提醒、取消订阅等操作请求走 business_action。"
                    "内容不清或缺少必要范围走 clarify。"
                    "如果需要知识库或工具，should_retrieve 必须为 true，direct_answer 必须为空。"
                    "如果不需要检索，direct_answer 应由主Agent直接自然回复，而不是固定模板。"
                ),
                user_prompt=(
                    "返回 JSON 字段：\n"
                    "- route_name: direct_conversation / permission_rag / document_analysis / business_action / clarify\n"
                    "- intent_name: 更细意图名，例如 greeting、capability、policy_lookup、project_lookup、version_compare、subscribe_request、unclear\n"
                    "- confidence: 0 到 1\n"
                    "- should_retrieve: 是否进入知识库或工具流程\n"
                    "- direct_answer: route_name 不需要检索时给用户的中文短回复；否则为 null\n"
                    "- reasoning: 简短分类依据\n\n"
                    "direct_answer 的要求：\n"
                    "- 直接回应用户这一轮意图\n"
                    "- 语气自然、简洁、像真实助手\n"
                    "- 可以简短说明能力边界，但不要空泛复读\n"
                    "- 不要编造已完成的订阅、提醒、推送等操作结果\n\n"
                    f"用户输入：{question}"
                ),
                response_model=TaskRouterLLMOutput,
            )
        except Exception:
            return None

        route_name = output.route_name.strip()
        if output.confidence < self.min_confidence:
            return None
        if route_name not in self.RETRIEVAL_ROUTES | self.DIRECT_ROUTES:
            return None
        if route_name in self.RETRIEVAL_ROUTES and not output.should_retrieve:
            return None

        direct_answer = output.direct_answer
        if route_name == "business_action":
            direct_answer = direct_answer or (
                "这个请求需要调用订阅、推送或提醒类业务接口。"
                "当前版本还没有接入真实操作 API，我先不替你确认已完成。"
            )
        if route_name == "clarify":
            direct_answer = direct_answer or "你可以再补充一下想查的制度、项目或文档范围。"
        if route_name == "direct_conversation":
            direct_answer = direct_answer or self._fallback_direct_answer(output.intent_name)

        should_retrieve = route_name in self.RETRIEVAL_ROUTES
        return TaskRouteResult(
            route_name=route_name,
            intent_name=output.intent_name,
            confidence=output.confidence,
            should_retrieve=should_retrieve,
            direct_answer=direct_answer,
            source="llm",
            reasoning=output.reasoning,
        )

    def _direct_route(
        self,
        *,
        route_name: str,
        intent_name: str,
        answer: str,
        confidence: float = 1.0,
        source: str = "rule",
        reasoning: str,
    ) -> TaskRouteResult:
        return TaskRouteResult(
            route_name=route_name,
            intent_name=intent_name,
            confidence=confidence,
            should_retrieve=False,
            direct_answer=answer,
            source=source,
            reasoning=reasoning,
        )

    def _normalize(self, question: str) -> str:
        text = re.sub(r"\s+", "", question.strip().lower())
        return text.strip("，。！？!?.,;；")

    def _is_business_action(self, question: str) -> bool:
        return any(token in question for token in self.BUSINESS_ACTION_TOKENS)

    def _is_document_analysis(self, question: str) -> bool:
        return any(token in question for token in self.DOCUMENT_ANALYSIS_TOKENS)

    def _is_knowledge_question(self, question: str) -> bool:
        return any(token.lower() in question.lower() for token in self.KNOWLEDGE_TOKENS)

    def _fallback_direct_answer(self, intent_name: str) -> str:
        if intent_name == "capability":
            return (
                "我可以帮你查询企业制度、项目资料、办公 FAQ 和文档版本变化。"
                "你可以直接问我具体问题，例如“出差报销最新标准是什么？”"
            )
        if intent_name == "self_intro":
            return "我是知达Agent，面向企业内部使用的知识问答助手。"
        if intent_name == "thanks":
            return "不客气，有需要可以继续问我。"
        return "我在，你可以直接告诉我想查询的制度、项目资料或办公问题。"
