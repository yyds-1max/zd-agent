from __future__ import annotations

import re

from app.schemas.intent import IntentResult

STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "一下",
    "现在",
    "是否",
    "一个",
    "我们",
    "可以",
    "需要",
}

DOMAIN_KEYWORDS = [
    "报销",
    "出差",
    "标准",
    "制度",
    "规则",
    "流程",
    "费用",
    "住宿",
    "餐补",
    "交通",
    "审核",
    "超时",
    "交付",
    "节点",
    "里程碑",
    "最新",
    "版本",
    "项目",
    "周报",
    "结论",
    "入职",
    "办公",
    "推荐",
    "FAQ",
]


class IntentParserService:
    def __init__(self, known_projects: list[str]):
        self.known_projects = known_projects

    def parse(self, question: str) -> IntentResult:
        project_names = self._extract_projects(question)
        keywords = self._extract_keywords(question)

        intent_name = "general_knowledge"
        if project_names or "项目" in question:
            intent_name = "project_lookup"
        elif any(word in question for word in ["FAQ", "常见问题"]):
            intent_name = "faq_lookup"
        elif any(word in question for word in ["入职", "新员工"]):
            intent_name = "onboarding_lookup"
        elif any(word in question for word in ["制度", "报销", "标准", "规则", "流程"]):
            intent_name = "policy_lookup"

        version_sensitive = any(
            word in question for word in ["最新", "最新版", "当前", "生效", "版本", "旧版", "过期"]
        ) or intent_name in {"policy_lookup", "project_lookup"}

        confidence = 0.65
        if project_names:
            confidence += 0.15
        if version_sensitive:
            confidence += 0.1

        reasoning_parts = [
            f"识别到意图 `{intent_name}`",
            f"项目线索: {project_names or ['无']}",
            f"版本敏感: {'是' if version_sensitive else '否'}",
        ]

        return IntentResult(
            name=intent_name,
            confidence=min(confidence, 0.95),
            keywords=keywords,
            project_names=project_names,
            version_sensitive=version_sensitive,
            reasoning="；".join(reasoning_parts),
        )

    def _extract_projects(self, question: str) -> list[str]:
        detected: list[str] = []
        for project in self.known_projects:
            if project in question and project not in detected:
                detected.append(project)
        regex_match = re.findall(r"项目[“\"]?([^”\"，。！？、 ]{1,8})", question)
        for match in regex_match:
            normalized = match
            for suffix in ["上周", "这周", "本周", "最新", "现在", "是什么", "情况"]:
                normalized = normalized.split(suffix, 1)[0]
            normalized = normalized.strip()
            if not normalized or normalized in detected:
                continue
            if any(project in normalized for project in self.known_projects):
                continue
            detected.append(normalized)
        return detected

    def _extract_keywords(self, question: str) -> list[str]:
        tokens: list[str] = []
        for keyword in DOMAIN_KEYWORDS:
            if keyword.lower() in question.lower() and keyword.lower() not in tokens:
                tokens.append(keyword.lower())
        for token in re.findall(r"[A-Za-z0-9]{2,}", question):
            normalized = token.strip().lower()
            if normalized not in tokens:
                tokens.append(normalized)
        if tokens:
            return tokens[:8]
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", question):
            for index in range(len(chunk) - 1):
                normalized = chunk[index : index + 2].lower()
                if normalized in STOPWORDS:
                    continue
                if normalized not in tokens:
                    tokens.append(normalized)
        return tokens[:8]
