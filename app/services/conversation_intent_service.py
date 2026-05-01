from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.llm import ConversationIntentLLMOutput
from app.services.main_agent_prompt import direct_reply_block
from app.services.qwen_llm_service import QwenStructuredLLMService


@dataclass(frozen=True)
class ConversationReply:
    intent_name: str
    answer: str
    confidence: float = 1.0
    source: str = "rule"


class ConversationIntentService:
    DIRECT_INTENTS = {
        "greeting",
        "self_intro",
        "capability",
        "smalltalk",
        "thanks",
        "unclear",
    }

    def __init__(
        self,
        llm_service: QwenStructuredLLMService | None = None,
        min_confidence: float = 0.72,
    ):
        self.llm_service = llm_service
        self.min_confidence = min_confidence

    def direct_reply(self, question: str) -> ConversationReply | None:
        normalized = self._normalize(question)
        if not normalized:
            return ConversationReply(
                intent_name="empty_message",
                answer="我在，你可以直接告诉我想查询的制度、项目资料或办公问题。",
            )

        llm_reply = self._direct_reply_with_llm(question)
        if llm_reply is not None:
            return llm_reply

        if self._is_greeting(normalized):
            return ConversationReply(
                intent_name="greeting",
                answer=(
                    "你好，我是知达Agent，企业知识问答助手。\n\n"
                    "我可以根据你的部门、岗位和权限范围，帮你查询制度、项目资料、FAQ "
                    "和版本更新内容。你可以直接问我，例如“出差报销最新标准是什么？”"
                ),
            )

        if self._is_identity_question(normalized):
            return ConversationReply(
                intent_name="self_intro",
                answer=(
                    "我是知达Agent，面向企业内部使用的知识问答助手。\n\n"
                    "我会优先基于你有权限查看的文档回答问题，并在你询问历史版本或版本变化时，"
                    "帮你区分新版和旧版内容。"
                ),
            )

        if self._is_capability_question(normalized):
            return ConversationReply(
                intent_name="capability",
                answer=(
                    "我可以帮你查询企业制度、项目资料、办公 FAQ 和文档版本变化。\n\n"
                    "比较适合这样问：\n"
                    "- 出差报销最新标准是什么？\n"
                    "- 项目北极星上周确认的交付节点是什么？\n"
                    "- 旧版报销制度和新版有什么变化？"
                ),
            )

        return None

    def _normalize(self, question: str) -> str:
        text = re.sub(r"\s+", "", question.strip().lower())
        return text.strip("，。！？!?.,;；")

    def _direct_reply_with_llm(self, question: str) -> ConversationReply | None:
        if self.llm_service is None or not self.llm_service.is_available():
            return None

        try:
            output = self.llm_service.generate_structured(
                system_prompt=(
                    f"{direct_reply_block()}"
                    "你当前承担的是主Agent的轻量对话决策职责。"
                    "你的任务是判断用户输入是否需要进入企业知识库检索。"
                    "只输出严格 JSON。"
                    "如果用户是在寒暄、询问你是谁、询问功能、感谢、闲聊或表达不清，"
                    "should_retrieve 必须为 false，并给出适合飞书 IM 的自然直接回复。"
                    "如果用户在询问具体制度、流程、项目资料、FAQ、版本内容或需要企业文档依据，"
                    "should_retrieve 必须为 true，direct_answer 置空。"
                    "direct_answer 不要写成模板话术，要像主Agent在当前上下文中的自然回复。"
                ),
                user_prompt=(
                    "请将用户输入分类到以下 intent_name 之一：\n"
                    "- greeting: 打招呼\n"
                    "- self_intro: 询问助手身份\n"
                    "- capability: 询问助手能力、功能、使用方式、能帮什么\n"
                    "- thanks: 感谢或结束语\n"
                    "- smalltalk: 与企业知识无关的闲聊\n"
                    "- business_question: 需要企业知识库回答的问题\n"
                    "- version_question: 询问历史版本、最新版、版本差异的问题\n"
                    "- unclear: 内容太短或不清楚，需要引导用户补充\n\n"
                    "返回 JSON 字段：\n"
                    "- intent_name: 上述类别之一\n"
                    "- confidence: 0 到 1\n"
                    "- should_retrieve: 是否需要进入知识库检索\n"
                    "- direct_answer: should_retrieve=false 时给用户的中文短回复；否则为 null\n"
                    "- reasoning: 简短分类依据\n\n"
                    "直接回复时请遵守：\n"
                    "- 优先正面回应用户当前问题，不要先讲一大段固定介绍\n"
                    "- 可以顺手补一句你能帮什么，但不要喧宾夺主\n"
                    "- 对感谢、寒暄、功能咨询分别给出贴合场景的自然答复\n\n"
                    f"用户输入：{question}"
                ),
                response_model=ConversationIntentLLMOutput,
            )
        except Exception:
            return None

        if output.confidence < self.min_confidence:
            return None
        if output.should_retrieve or output.intent_name not in self.DIRECT_INTENTS:
            return None

        answer = (output.direct_answer or "").strip()
        if not answer:
            answer = self._fallback_direct_answer(output.intent_name)
        return ConversationReply(
            intent_name=output.intent_name,
            answer=answer,
            confidence=output.confidence,
            source="llm",
        )

    def _fallback_direct_answer(self, intent_name: str) -> str:
        if intent_name == "capability":
            return (
                "我可以帮你查询企业制度、项目资料、办公 FAQ 和文档版本变化。\n\n"
                "你可以直接问我具体问题，例如“出差报销最新标准是什么？”"
            )
        if intent_name == "self_intro":
            return "我是知达Agent，面向企业内部使用的知识问答助手。"
        if intent_name == "thanks":
            return "不客气，有需要可以继续问我。"
        return "我在，你可以直接告诉我想查询的制度、项目资料或办公问题。"

    def _is_greeting(self, text: str) -> bool:
        return text in {
            "你好",
            "您好",
            "hi",
            "hello",
            "哈喽",
            "在吗",
            "在不在",
            "嗨",
        }

    def _is_identity_question(self, text: str) -> bool:
        identity_patterns = [
            "你是谁",
            "你是什么",
            "介绍一下你自己",
            "自我介绍",
            "你叫什么",
            "你是干什么的",
        ]
        return any(pattern in text for pattern in identity_patterns)

    def _is_capability_question(self, text: str) -> bool:
        capability_patterns = [
            "你能做什么",
            "你可以做什么",
            "你会什么",
            "怎么使用你",
            "怎么用你",
            "你能帮我什么",
            "你支持什么",
            "你有什么功能",
            "有什么功能",
            "你的功能",
        ]
        return any(pattern in text for pattern in capability_patterns)
