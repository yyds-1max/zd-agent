import os
import re

from app.core.config import settings
from app.schemas.citation import Citation

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 依赖缺失时走规则兜底
    OpenAI = None  # type: ignore[assignment]


class AnswerService:
    POLICY_KEYWORDS = (
        "制度",
        "规范",
        "规则",
        "报销",
        "审批",
        "流程",
        "标准",
        "适用范围",
        "生效",
        "合规",
    )

    def __init__(self, enable_llm_generation: bool | None = None) -> None:
        if enable_llm_generation is None:
            self._enable_llm_generation = settings.enable_llm_answer_generation
        else:
            self._enable_llm_generation = enable_llm_generation
        self._model_name = settings.answer_model.strip() or settings.chat_model.strip() or "qwen3-max"
        self._client = None

    def compose(
        self,
        question: str,
        citations: list[Citation],
        intent_type: str | None = None,
        version_hint: str | None = None,
    ) -> str:
        if not citations:
            return self._build_empty_answer(question)

        llm_answer = self._compose_with_llm(question, citations, intent_type, version_hint)
        if llm_answer:
            return self._append_version_notice(llm_answer, version_hint)

        version_focus = self._build_version_focus(question, citations)
        top = (
            version_focus.get("latest_item")  # type: ignore[assignment]
            if version_focus and version_focus.get("latest_item") is not None
            else citations[0]
        )
        summary = self._build_summary(question, top, version_focus=version_focus)
        version = self._build_version(top, version_focus=version_focus)
        effective_time = self._build_effective_time(top, version_focus=version_focus)
        scope = self._build_scope(top)
        sources = self._build_sources(citations)
        version_extra = self._build_version_extra_blocks(version_focus)

        if self._is_policy_question(question, citations, intent_type):
            answer = (
                "【制度类答复】\n"
                "本回答按企业制度模板生成，优先使用当前用户可见的制度依据。\n\n"
                f"【答案摘要】\n{summary}\n\n"
                f"【制度口径】\n"
                f"1. 请按引用中的当前制度版本执行。\n"
                f"2. 如业务场景有例外，需按制度要求补充审批。\n\n"
                f"【引用来源】\n{sources}\n\n"
                f"【版本】\n{version}\n\n"
                f"【生效时间】\n{effective_time}\n\n"
                f"【适用范围】\n{scope}"
            )
            if version_extra:
                answer = f"{answer}\n\n{version_extra}"
            return self._append_version_notice(answer, version_hint)

        answer = (
            f"【答案摘要】\n{summary}\n\n"
            f"【引用来源】\n{sources}\n\n"
            f"【版本】\n{version}\n\n"
            f"【生效时间】\n{effective_time}\n\n"
            f"【适用范围】\n{scope}"
        )
        if version_extra:
            answer = f"{answer}\n\n{version_extra}"
        return self._append_version_notice(answer, version_hint)

    def _compose_with_llm(
        self,
        question: str,
        citations: list[Citation],
        intent_type: str | None,
        version_hint: str | None,
    ) -> str | None:
        if not self._enable_llm_generation:
            return None
        client = self._get_client()
        if client is None:
            return None

        version_focus = self._build_version_focus(question, citations)
        refs = self._build_reference_context(citations)
        required_sections = ["【答案摘要】", "【引用来源】", "【版本】", "【生效时间】", "【适用范围】"]
        extra_policy_instruction = ""
        if self._is_policy_question(question, citations, intent_type):
            extra_policy_instruction = "如果是制度类问题，必须额外输出【制度口径】小节（两条以内）。"
        extra_version_instruction = ""
        version_focus_context = "无"
        if version_focus and version_focus.get("latest_item") is not None:
            extra_version_instruction = (
                "若用户明确提问的是指定旧版本（例如 V1.0），你必须先回答该指定版本口径，"
                "再说明当前有更新版本，并输出【新旧差异】小节（1-3条）。"
                "此场景下，请额外输出【版本更新提醒】与【新旧差异】两个小节。"
            )
            version_focus_context = self._format_version_focus_context(version_focus)

        prompt = (
            "你是企业知识助手。你只能依据提供的引用片段回答，不能补充未给出的事实。"
            "输出必须是中文，并严格包含这些小节标题："
            + "、".join(required_sections)
            + "。"
            "在【引用来源】中必须逐条写出 doc_id、标题、版本、更新时间。"
            "在【版本】中明确当前使用的版本信息。"
            + extra_policy_instruction
            + extra_version_instruction
        )
        user_content = (
            f"用户问题：{question}\n"
            f"意图类型：{intent_type or 'unknown'}\n"
            f"版本提示：{version_hint or '无'}\n"
            f"版本专项上下文：{version_focus_context}\n"
            f"可用引用片段（rerank结果，共{len(citations)}条）：\n{refs}\n"
            "请基于以上引用生成最终答案。"
        )

        try:
            response = client.chat.completions.create(
                model=self._model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception:
            return None

        if not self._is_valid_llm_answer(content, citations):
            return None
        return content

    def _get_client(self):
        if OpenAI is None:
            return None
        if self._client is not None:
            return self._client

        base_url = os.getenv("OPENAI_BASE_URL")
        api_key: str | None = None
        if base_url:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        elif os.getenv("DASHSCOPE_API_KEY"):
            api_key = os.getenv("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=20.0)
        else:
            self._client = OpenAI(api_key=api_key, max_retries=0, timeout=20.0)
        return self._client

    def _build_reference_context(self, citations: list[Citation]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(citations, start=1):
            updated = item.updated_at.isoformat() if item.updated_at else "未知"
            effective = item.effective_date.isoformat() if item.effective_date else "未知"
            text = re.sub(r"\s+", " ", item.content_chunk).strip()
            if len(text) > 360:
                text = f"{text[:360]}..."
            lines.append(
                f"[{idx}] doc_id={item.doc_id} | title={item.title} | source={item.source_type} | "
                f"version={item.version} | effective_date={effective} | updated_at={updated}\n"
                f"片段：{text}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _is_valid_llm_answer(answer: str, citations: list[Citation]) -> bool:
        required_headers = ("【答案摘要】", "【引用来源】", "【版本】", "【生效时间】", "【适用范围】")
        if not answer:
            return False
        if not all(header in answer for header in required_headers):
            return False
        if not citations:
            return True
        # 至少命中一个引用 doc_id，避免“有引用标题但无溯源”。
        return any(item.doc_id and item.doc_id in answer for item in citations)

    def _build_summary(
        self,
        question: str,
        top: Citation,
        version_focus: dict[str, Citation | str] | None = None,
    ) -> str:
        if version_focus and version_focus.get("latest_item") is not None:
            requested_item = version_focus["requested_item"]
            latest_item = version_focus["latest_item"]
            requested_fact = self._extract_key_fact(str(requested_item.content_chunk))
            latest_fact = self._extract_key_fact(str(latest_item.content_chunk))
            return (
                f"针对“{question}”，你提到的 {requested_item.version} 口径为：{requested_fact}；"
                f"当前最新版本 {latest_item.version} 口径为：{latest_fact}。"
            )
        snippet = top.content_chunk.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = f"{snippet[:220]}..."
        return f"针对“{question}”，当前可用知识摘要如下：{snippet}"

    @staticmethod
    def _build_sources(citations: list[Citation]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(citations[:3], start=1):
            updated_at = item.updated_at.isoformat() if item.updated_at else "未知"
            lines.append(f"{idx}. {item.title}（doc_id={item.doc_id}，更新时间={updated_at}）")
        return "\n".join(lines)

    @staticmethod
    def _build_version(
        top: Citation,
        version_focus: dict[str, Citation | str] | None = None,
    ) -> str:
        if version_focus and version_focus.get("latest_item") is not None:
            requested_item = version_focus["requested_item"]
            latest_item = version_focus["latest_item"]
            return f"指定查询版本：{requested_item.version}；当前生效版本：{latest_item.version}"
        latest_text = "当前生效版本" if top.is_latest else "非最新版本（请留意版本更新）"
        return f"{top.version}，{latest_text}"

    @staticmethod
    def _build_scope(top: Citation) -> str:
        role = "、".join(top.role_scope) if top.role_scope else "未标注"
        dept = "、".join(top.department_scope) if top.department_scope else "未标注"
        project = "、".join(top.project_scope) if top.project_scope else "未标注"
        return f"角色：{role}；部门：{dept}；项目：{project}"

    def _is_policy_question(
        self, question: str, citations: list[Citation], intent_type: str | None = None
    ) -> bool:
        if intent_type == "policy":
            return True
        if intent_type in {"faq", "project", "chat_summary", "recommendation"}:
            return False
        if citations and citations[0].source_type == "policy":
            return True
        normalized = question.strip().lower()
        return any(keyword in normalized for keyword in self.POLICY_KEYWORDS)

    def _build_empty_answer(self, question: str) -> str:
        return (
            f"【答案摘要】\n未找到与“{question}”相关且当前用户可访问的知识。\n\n"
            f"【引用来源】\n无\n\n"
            f"【版本】\n无可用版本\n\n"
            f"【生效时间】\n未知\n\n"
            f"【适用范围】\n当前问题未命中可用知识"
        )

    @staticmethod
    def _append_version_notice(answer: str, version_hint: str | None) -> str:
        if "【版本更新提醒】" in answer:
            return answer
        if not version_hint or "当前有更新版本" not in version_hint:
            return answer
        return f"{answer}\n\n【版本更新提醒】\n{version_hint}"

    def _build_effective_time(
        self,
        top: Citation,
        version_focus: dict[str, Citation | str] | None = None,
    ) -> str:
        if version_focus and version_focus.get("latest_item") is not None:
            requested_item = version_focus["requested_item"]
            latest_item = version_focus["latest_item"]
            requested_effective = (
                requested_item.effective_date.isoformat() if requested_item.effective_date else "未知"
            )
            latest_effective = latest_item.effective_date.isoformat() if latest_item.effective_date else "未知"
            return f"指定版本生效时间：{requested_effective}；当前生效版本时间：{latest_effective}"
        return top.effective_date.isoformat() if top.effective_date else "未知"

    def _build_version_extra_blocks(self, version_focus: dict[str, Citation | str] | None) -> str:
        if not version_focus or version_focus.get("latest_item") is None:
            return ""
        requested_item = version_focus["requested_item"]
        latest_item = version_focus["latest_item"]
        latest_updated = latest_item.updated_at.isoformat() if latest_item.updated_at else "未知"
        diff = str(version_focus.get("diff") or "").strip()
        reminder = (
            f"你询问的是旧版本 {requested_item.version}。当前有更新版本："
            f"{latest_item.title}（{latest_item.version}，更新时间={latest_updated}）。"
        )
        if not diff:
            return f"【版本更新提醒】\n{reminder}"
        return f"【版本更新提醒】\n{reminder}\n\n【新旧差异】\n{diff}"

    def _build_version_focus(
        self, question: str, citations: list[Citation]
    ) -> dict[str, Citation | str] | None:
        requested_version = self._extract_requested_version(question)
        if not requested_version:
            return None
        normalized = self._normalize_version_token(requested_version)
        if not normalized:
            return None

        matched = [
            item
            for item in citations
            if self._normalize_version_token(item.version) == normalized
        ]
        if not matched:
            return None
        requested_item = matched[0]
        latest_item = self._find_latest_related_item(requested_item, citations)
        if latest_item is None:
            return None
        if self._normalize_version_token(latest_item.version) == self._normalize_version_token(requested_item.version):
            return None

        diff = self._build_diff_with_rules(requested_item.content_chunk, latest_item.content_chunk)
        return {
            "requested_version": requested_version,
            "requested_item": requested_item,
            "latest_item": latest_item,
            "diff": diff,
        }

    @staticmethod
    def _extract_requested_version(question: str) -> str | None:
        match = re.search(r"\bV?\d+(?:\.\d+)?\b", question, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(0)

    @staticmethod
    def _normalize_version_token(version: object) -> str:
        text = str(version or "").strip().lower()
        match = re.search(r"v?(\d+(?:\.\d+)?)", text)
        if not match:
            return ""
        return f"v{match.group(1)}"

    @staticmethod
    def _normalize_title(title: str) -> str:
        text = title.strip().lower()
        text = re.sub(r"[（(]\s*v?\d+(?:\.\d+)?\s*[）)]", "", text)
        text = re.sub(r"\bv?\d+(?:\.\d+)?\b", "", text)
        return re.sub(r"\s+", "", text)

    def _find_latest_related_item(self, requested_item: Citation, citations: list[Citation]) -> Citation | None:
        topic_key = self._normalize_title(requested_item.title)
        candidates = [
            item
            for item in citations
            if self._normalize_title(item.title) == topic_key
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                int(bool(item.is_latest)),
                item.effective_date.isoformat() if item.effective_date else "",
                item.updated_at.isoformat() if item.updated_at else "",
                self._version_sort_value(item.version),
            ),
            reverse=True,
        )
        return candidates[0]

    @staticmethod
    def _version_sort_value(version: object) -> float:
        text = str(version or "").lower()
        match = re.search(r"v?(\d+(?:\.\d+)?)", text)
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    @staticmethod
    def _build_diff_with_rules(old_text: str, new_text: str) -> str:
        old_segments = AnswerService._segments(old_text)
        new_segments = AnswerService._segments(new_text)
        added = [s for s in new_segments if s not in old_segments][:2]
        removed = [s for s in old_segments if s not in new_segments][:2]
        parts: list[str] = []
        if added:
            parts.append(f"新版新增/强调：{'；'.join(added)}")
        if removed:
            parts.append(f"旧版提及但新版未出现：{'；'.join(removed)}")
        if not parts:
            return "核心规则整体一致，建议按最新版本执行。"
        return "；".join(parts)

    @staticmethod
    def _segments(text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return []
        parts = [p.strip() for p in re.split(r"[。；;!！?？\n]", compact) if p.strip()]
        return [p[:36] for p in parts]

    @staticmethod
    def _extract_key_fact(text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return "未提取到明确口径。"
        sentences = [s.strip() for s in re.split(r"[。；;!！?？\n]", compact) if s.strip()]
        for sentence in sentences:
            if any(token in sentence for token in ("报销", "时限", "天", "日", "标准", "提交")):
                return sentence[:80]
        return sentences[0][:80]

    @staticmethod
    def _format_version_focus_context(version_focus: dict[str, Citation | str]) -> str:
        requested_item = version_focus["requested_item"]
        latest_item = version_focus["latest_item"]
        diff = str(version_focus.get("diff") or "").strip() or "无"
        return (
            f"用户指定版本：{requested_item.version}；"
            f"指定版本标题：{requested_item.title}；"
            f"当前最新版本：{latest_item.version}；"
            f"最新版本标题：{latest_item.title}；"
            f"差异线索：{diff}"
        )
