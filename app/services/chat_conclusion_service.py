import os
import re

from app.core.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 依赖缺失时走规则兜底
    OpenAI = None  # type: ignore[assignment]


class ChatConclusionService:
    """从原始聊天记录中自动提炼可入库的结构化结论。"""

    _DECISION_KEYWORDS = (
        "确认",
        "决定",
        "结论",
        "统一",
        "通过",
        "确定",
        "采用",
        "调整为",
        "改为",
        "优先",
        "必须",
        "需",
        "本期",
    )
    _FOLLOW_UP_KEYWORDS = ("待确认", "后续", "跟进", "补充", "todo", "待办", "安排")

    def __init__(self, enable_llm: bool | None = None) -> None:
        if enable_llm is None:
            self._enable_llm = settings.enable_llm_chat_conclusion
        else:
            self._enable_llm = enable_llm
        self._client = None

    def refine(self, body: str, title: str = "") -> str:
        text = self._normalize_text(body)
        if not text:
            return body
        if self._looks_like_structured_summary(text):
            return text

        messages = self._extract_messages(text)
        if not messages:
            return self._fallback_summary(text, title)

        rule_summary = self._build_rule_summary(messages, title)
        llm_summary = self._build_with_llm(text, title, rule_summary)
        return llm_summary or rule_summary

    @staticmethod
    def _normalize_text(text: str) -> str:
        compact = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        return compact.strip()

    @staticmethod
    def _looks_like_structured_summary(text: str) -> bool:
        if "讨论确认" in text and "结论" in text:
            return True
        if re.search(r"^[一二三四五六七八九十]+、", text, flags=re.MULTILINE):
            return True
        if re.search(r"^\d+\.\s+", text, flags=re.MULTILINE):
            return True
        return False

    def _extract_messages(self, text: str) -> list[str]:
        rows: list[str] = []
        for line in text.split("\n"):
            normalized = line.strip()
            if not normalized:
                continue
            content = self._strip_message_prefix(normalized)
            content = content.strip(" -\t")
            if len(content) < 4:
                continue
            rows.append(content)
        return rows

    @staticmethod
    def _strip_message_prefix(line: str) -> str:
        # 例：2026-04-02 10:30 张三：文本 / [10:30] 李四: 文本 / 张三：文本
        patterns = (
            r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?\s+[^\s:：]{1,20}\s*[:：]\s*(.+)$",
            r"^\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s+[^\s:：]{1,20}\s*[:：]\s*(.+)$",
            r"^[^\s:：]{1,20}\s*[:：]\s*(.+)$",
        )
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()
        return line

    def _build_rule_summary(self, messages: list[str], title: str) -> str:
        conclusions = self._pick_conclusions(messages)
        follow_ups = self._pick_follow_ups(messages)

        lines: list[str] = []
        header = f"{title}（自动提炼）" if title else "聊天结论提炼（自动提炼）"
        lines.append(header)
        lines.append("")
        lines.append("一、关键结论")
        if conclusions:
            for idx, item in enumerate(conclusions, start=1):
                lines.append(f"{idx}. {item}")
        else:
            lines.append("1. 本次聊天未识别到明确决策，请结合原文人工复核。")

        if follow_ups:
            lines.append("")
            lines.append("二、待跟进事项")
            for idx, item in enumerate(follow_ups, start=1):
                lines.append(f"{idx}. {item}")

        lines.append("")
        lines.append(f"（原始聊天共 {len(messages)} 条消息，以上为系统自动提炼）")
        return "\n".join(lines).strip()

    def _pick_conclusions(self, messages: list[str]) -> list[str]:
        scored: list[tuple[float, str]] = []
        for idx, message in enumerate(messages):
            for sentence in self._split_sentences(message):
                clean = self._clean_sentence(sentence)
                if len(clean) < 8:
                    continue
                score = self._conclusion_score(clean, idx, len(messages))
                scored.append((score, clean))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for _, sentence in scored:
            key = re.sub(r"\s+", "", sentence)
            if key in seen:
                continue
            seen.add(key)
            selected.append(sentence)
            if len(selected) >= 5:
                break
        return selected

    def _pick_follow_ups(self, messages: list[str]) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()
        for message in messages:
            for sentence in self._split_sentences(message):
                clean = self._clean_sentence(sentence)
                if len(clean) < 8:
                    continue
                lower = clean.lower()
                if not any(token in lower for token in self._FOLLOW_UP_KEYWORDS):
                    continue
                key = re.sub(r"\s+", "", clean)
                if key in seen:
                    continue
                seen.add(key)
                items.append(clean)
                if len(items) >= 3:
                    return items
        return items

    def _conclusion_score(self, sentence: str, index: int, total: int) -> float:
        lower = sentence.lower()
        score = 0.0
        if any(keyword in lower for keyword in self._DECISION_KEYWORDS):
            score += 3.0
        if re.search(r"\d", sentence):
            score += 1.2
        if any(token in sentence for token in ("版本", "权限", "接入", "上线", "范围", "规则", "项目")):
            score += 1.0
        # 越靠后通常越接近最终结论
        score += (index + 1) / max(total, 1)
        score += min(len(sentence), 120) / 200.0
        return score

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = [part.strip() for part in re.split(r"[。；;!?！？\n]", text) if part.strip()]
        return parts

    @staticmethod
    def _clean_sentence(text: str) -> str:
        sentence = re.sub(r"\s+", " ", text).strip()
        sentence = re.sub(r"^(收到|好的|ok|嗯|明白了)[,，。]*", "", sentence, flags=re.IGNORECASE)
        sentence = sentence.strip(" -\t")
        return sentence

    def _build_with_llm(self, raw_text: str, title: str, rule_summary: str) -> str | None:
        if not self._enable_llm:
            return None
        client = self._get_client()
        if client is None:
            return None

        prompt = (
            "你是企业聊天纪要助手。请将原始聊天提炼为结构化结论，必须忠于原文，不得捏造。"
            "输出格式：标题、'一、关键结论'（3-5条）、可选'二、待跟进事项'（0-3条）。"
            "每条不超过60字，语气客观。"
        )
        raw_excerpt = raw_text[:2600]
        fallback_excerpt = rule_summary[:1600]
        user_content = (
            f"文档标题：{title or '未命名聊天'}\n"
            f"原始聊天：\n{raw_excerpt}\n\n"
            f"规则提炼草稿：\n{fallback_excerpt}\n"
            "请输出最终提炼结果。"
        )
        try:
            model_name = settings.chat_conclusion_model.strip() or settings.chat_model.strip() or "qwen3-max"
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception:
            return None

        if not content or "关键结论" not in content:
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
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)
        return self._client

    @staticmethod
    def _fallback_summary(text: str, title: str) -> str:
        one_line = re.sub(r"\s+", " ", text).strip()
        snippet = one_line[:220]
        header = f"{title}（自动提炼）" if title else "聊天结论提炼（自动提炼）"
        return (
            f"{header}\n\n"
            "一、关键结论\n"
            f"1. {snippet}\n\n"
            "（未识别到标准聊天格式，已使用全文摘要兜底）"
        )
