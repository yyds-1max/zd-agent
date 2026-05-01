from __future__ import annotations

import json

from app.schemas.intent import IntentResult
from app.schemas.llm import UserProfileLLMOutput
from app.schemas.user import DirectoryUser
from app.services.qwen_llm_service import QwenStructuredLLMService

ALLOWED_INTENTS = [
    "general_knowledge",
    "policy_lookup",
    "project_lookup",
    "faq_lookup",
    "onboarding_lookup",
]


class UserProfileLLMService:
    def __init__(self, llm_service: QwenStructuredLLMService, known_projects: list[str]):
        self.llm_service = llm_service
        self.known_projects = known_projects

    def is_available(self) -> bool:
        return self.llm_service.is_available()

    def understand(self, question: str, directory_user: DirectoryUser) -> IntentResult:
        system_prompt = (
            "你是企业知识助手中的用户画像理解模块。"
            "你的任务是结合用户的企业身份信息和提问内容，输出严格 JSON。"
            "不要输出任何 JSON 之外的内容，不要编造不存在的项目名。"
            f"合法 intent_name 仅可为: {', '.join(ALLOWED_INTENTS)}。"
        )
        user_prompt = (
            "请根据以下信息完成用户意图识别和结构化抽取。\n"
            f"用户目录信息:\n{json.dumps(directory_user.model_dump(), ensure_ascii=False, indent=2)}\n"
            f"已知项目名:\n{json.dumps(self.known_projects, ensure_ascii=False)}\n"
            f"用户问题:\n{question}\n\n"
            "输出字段说明:\n"
            "- intent_name: 问题意图\n"
            "- confidence: 0~1\n"
            "- keywords: 检索关键词，最多 8 个\n"
            "- project_names: 提问中涉及的项目名，必须来自已知项目名或明确提到的项目\n"
            "- version_sensitive: 是否涉及最新/旧版/当前生效等版本敏感问题\n"
            "- reasoning: 简短解释\n"
            "- ambiguity_note: 如果存在歧义，可给一句短提示，否则为 null\n"
        )
        output = self.llm_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=UserProfileLLMOutput,
        )
        intent_name = output.intent_name if output.intent_name in ALLOWED_INTENTS else "general_knowledge"
        return IntentResult(
            name=intent_name,
            confidence=output.confidence,
            keywords=output.keywords[:8],
            project_names=output.project_names,
            version_sensitive=output.version_sensitive,
            reasoning=output.reasoning if not output.ambiguity_note else f"{output.reasoning}；歧义: {output.ambiguity_note}",
        )
