import os
import re
from datetime import date, datetime

from app.core.config import settings
from app.repositories.metadata_repository import MetadataRepository
from app.schemas.citation import Citation

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 依赖缺失时走规则兜底
    OpenAI = None  # type: ignore[assignment]


class VersionService:
    def __init__(
        self, metadata_repo: MetadataRepository | None = None, enable_llm_diff: bool | None = None
    ) -> None:
        self._metadata_repo = metadata_repo or MetadataRepository()
        if enable_llm_diff is None:
            self._enable_llm_diff = settings.enable_llm_version_diff
        else:
            self._enable_llm_diff = enable_llm_diff
        self._client = None

    def build_version_hint(self, items: list[Citation], question: str | None = None) -> str | None:
        if not items:
            return None
        requested = self._match_requested_version(items, question)
        if requested is not None:
            if not requested.is_latest:
                return self._build_obsolete_hint(requested)
            return f"你指定的版本为 {requested.version}，该版本为当前生效版本。"

        top = items[0]
        if not top.is_latest:
            return self._build_obsolete_hint(top)

        versions = {item.version for item in items if item.version}
        if len(versions) > 1:
            return f"检测到多个版本，已优先返回最新版本。当前版本：{top.version}"
        return f"当前版本：{next(iter(versions))}" if versions else None

    def _match_requested_version(self, items: list[Citation], question: str | None) -> Citation | None:
        version_token = self._extract_requested_version(question)
        if not version_token:
            return None
        normalized = self._normalize_version_token(version_token)
        if not normalized:
            return None

        candidates = [
            item
            for item in items
            if self._normalize_version_token(item.version) == normalized
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                int(bool(item.is_latest)),
                self._date_value(item.effective_date),
                self._datetime_value(item.updated_at),
                self._version_value(item.version),
            ),
            reverse=True,
        )
        return candidates[0]

    def _build_obsolete_hint(self, old_item: Citation) -> str:
        latest = self._find_latest_version(old_item)
        if latest is None:
            return f"当前有更新版本。你当前命中的是旧版本：{old_item.version}，请核对最新制度。"

        latest_updated = latest.get("updated_at") or "未知"
        latest_effective = latest.get("effective_date") or "未知"
        latest_version = latest.get("version", "未知")
        latest_summary = str(latest.get("summary") or "").strip()
        latest_text = str(latest.get("content_chunk") or "")
        if not latest_summary:
            latest_summary = self._truncate(latest_text)

        diff = self._build_diff(old_item.content_chunk, latest_text or latest_summary)
        return (
            f"当前有更新版本：{latest_version}。"
            f"你当前命中的是旧版本：{old_item.version}。"
            f"新版本更新时间：{latest_updated}，生效时间：{latest_effective}。"
            f"新版本要点：{self._truncate(latest_summary)}。"
            f"主要差异：{diff}"
        )

    def _find_latest_version(self, item: Citation) -> dict | None:
        try:
            docs = self._metadata_repo.list_document_versions(source_type=item.source_type)
        except Exception:
            return None
        if not docs:
            return None

        topic_key = self._normalize_title(item.title)
        candidates = [doc for doc in docs if self._normalize_title(str(doc.get("title", ""))) == topic_key]
        if not candidates:
            return None

        candidates.sort(
            key=lambda doc: (
                int(bool(doc.get("is_latest"))),
                self._date_value(doc.get("effective_date")),
                self._datetime_value(doc.get("updated_at")),
                self._version_value(doc.get("version")),
            ),
            reverse=True,
        )
        latest = candidates[0]
        if str(latest.get("version")) == item.version and bool(latest.get("is_latest")) == item.is_latest:
            return None
        return latest

    @staticmethod
    def _normalize_title(title: str) -> str:
        text = title.strip().lower()
        text = re.sub(r"[（(]\s*v?\d+(?:\.\d+)?\s*[）)]", "", text)
        text = re.sub(r"\bv?\d+(?:\.\d+)?\b", "", text)
        return re.sub(r"\s+", "", text)

    @staticmethod
    def _version_value(version: object) -> float:
        text = str(version or "").lower()
        match = re.search(r"v?(\d+(?:\.\d+)?)", text)
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    @staticmethod
    def _normalize_version_token(version: object) -> str:
        text = str(version or "").strip().lower()
        match = re.search(r"v?(\d+(?:\.\d+)?)", text)
        if not match:
            return ""
        return f"v{match.group(1)}"

    @staticmethod
    def _extract_requested_version(question: str | None) -> str | None:
        if not question:
            return None
        match = re.search(r"\bV?\d+(?:\.\d+)?\b", question, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(0)

    @staticmethod
    def _datetime_value(raw: object) -> float:
        if isinstance(raw, datetime):
            return raw.timestamp()
        text = str(raw or "").strip()
        if not text:
            return 0.0
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return 0.0

    @staticmethod
    def _date_value(raw: object) -> float:
        if isinstance(raw, date):
            return raw.toordinal()
        text = str(raw or "").strip()
        if not text:
            return 0.0
        try:
            return date.fromisoformat(text).toordinal()
        except ValueError:
            return 0.0

    @staticmethod
    def _truncate(text: str, limit: int = 120) -> str:
        content = re.sub(r"\s+", " ", text).strip()
        if len(content) <= limit:
            return content or "无"
        return f"{content[:limit]}..."

    def _build_diff(self, old_text: str, new_text: str) -> str:
        llm_diff = self._build_diff_with_llm(old_text, new_text)
        if llm_diff:
            return llm_diff
        return self._build_diff_with_rules(old_text, new_text)

    def _build_diff_with_rules(self, old_text: str, new_text: str) -> str:
        old_segments = self._segments(old_text)
        new_segments = self._segments(new_text)
        added = [s for s in new_segments if s not in old_segments][:2]
        removed = [s for s in old_segments if s not in new_segments][:2]

        parts: list[str] = []
        if added:
            parts.append(f"新增/强调：{'；'.join(added)}")
        if removed:
            parts.append(f"旧版提及但新版未出现：{'；'.join(removed)}")
        if not parts:
            return "核心规则整体一致，建议按最新版本执行细则。"
        return "；".join(parts)

    def _build_diff_with_llm(self, old_text: str, new_text: str) -> str | None:
        if not self._enable_llm_diff:
            return None
        client = self._get_client()
        if client is None:
            return None

        old_excerpt = self._truncate(old_text, limit=600)
        new_excerpt = self._truncate(new_text, limit=600)
        prompt = (
            "你是企业制度版本对比助手。"
            "请基于旧版与新版原文，总结“新增/删除/变更”差异。"
            "要求：1) 只使用给定文本；2) 用中文，1-3句；3) 不要输出无关解释。"
        )
        user_content = (
            f"旧版片段：{old_excerpt}\n"
            f"新版片段：{new_excerpt}\n"
            "请直接输出差异总结。"
        )
        try:
            model_name = settings.version_diff_model.strip() or settings.chat_model.strip() or "qwen3-max"
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                return None
            return self._truncate(content, limit=180)
        except Exception:
            return None

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
    def _segments(text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return []
        parts = [p.strip() for p in re.split(r"[。；;!！?？\n]", compact) if p.strip()]
        return [p[:32] for p in parts]
