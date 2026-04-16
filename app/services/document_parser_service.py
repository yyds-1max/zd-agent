import re
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.services.chat_conclusion_service import ChatConclusionService

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - 依赖未安装时的兜底
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


class DocumentParserService:
    def __init__(self) -> None:
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("未安装 langchain-text-splitters，请执行 `pip install -r requirements.txt`。")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.text_chunk_size,
            chunk_overlap=settings.text_chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
            keep_separator=False,
        )
        self._chat_conclusion = ChatConclusionService()

    def parse_directory(self, source_dir: str) -> list[dict]:
        base = Path(source_dir)
        files = sorted(base.rglob("*.txt"))
        chunks: list[dict] = []
        for file_path in files:
            chunks.extend(self._parse_file(file_path))
        return chunks

    def _parse_file(self, file_path: Path) -> list[dict]:
        raw_text = self._read_text(file_path)
        text = self._clean_text(raw_text)
        meta, body = self._extract_meta_and_body(text)
        if not body.strip():
            return []

        doc_id = self._build_doc_id(file_path)
        title = self._extract_title(file_path, body)
        source_type = self._infer_source_type(file_path, meta)
        version = self._extract_version(file_path, meta)
        status = (meta.get("状态") or "").strip()
        is_latest = "过期" not in status and "废弃" not in status
        publish_date = self._extract_date(meta.get("发布日期", ""))
        update_date = self._extract_date(meta.get("更新时间", "")) or publish_date

        role_scope, department_scope, project_scope = self._build_scopes(file_path, meta)
        tags = self._build_tags(source_type, title, file_path)
        refined_body = self._refine_body(source_type, body, title)
        summary = self._summarize(refined_body)
        doc_chunks = self._splitter.split_text(refined_body)

        records: list[dict] = []
        for part in doc_chunks:
            records.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source_type": source_type,
                    "content_chunk": part,
                    "summary": summary,
                    "department_scope": department_scope,
                    "role_scope": role_scope,
                    "project_scope": project_scope,
                    "version": version,
                    "is_latest": is_latest,
                    "effective_date": publish_date,
                    "updated_at": self._as_datetime(update_date),
                    "tags": tags,
                }
            )
        return records

    def _refine_body(self, source_type: str, body: str, title: str) -> str:
        if source_type != "chat_summary":
            return body
        return self._chat_conclusion.refine(body, title=title)

    @staticmethod
    def _read_text(file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="gb18030")

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in text.split("\n")]
        compact = "\n".join(lines)
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        return compact.strip()

    @staticmethod
    def _extract_meta_and_body(text: str) -> tuple[dict[str, str], str]:
        lines = text.split("\n")
        meta: dict[str, str] = {}
        body_start = 0
        for idx, line in enumerate(lines):
            normalized = line.strip()
            if normalized == "正文":
                body_start = idx + 1
                break
            if "：" in normalized:
                key, value = normalized.split("：", 1)
                meta[key.strip()] = value.strip()
            elif ":" in normalized:
                key, value = normalized.split(":", 1)
                meta[key.strip()] = value.strip()
        body = "\n".join(lines[body_start:]).strip() if body_start > 0 else text
        return meta, body

    @staticmethod
    def _build_doc_id(file_path: Path) -> str:
        match = re.match(r"^(\d+)_", file_path.stem)
        if match:
            return f"fixture-{match.group(1)}"
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", file_path.stem)
        return f"fixture-{safe}".strip("-")

    @staticmethod
    def _extract_title(file_path: Path, body: str) -> str:
        for line in body.split("\n"):
            text = line.strip()
            if not text:
                continue
            if len(text) <= 60 and "、" not in text and not re.match(r"^[一二三四五六七八九十]+[、.]", text):
                return text
            break
        name = re.sub(r"^\d+_", "", file_path.stem)
        name = re.sub(r"_V\d+(\.\d+)?$", "", name, flags=re.IGNORECASE)
        return name

    @staticmethod
    def _infer_source_type(file_path: Path, meta: dict[str, str]) -> str:
        doc_type = (meta.get("文档类型") or "").lower()
        name = file_path.stem.lower()
        if "聊天" in doc_type or "结论" in doc_type or "群结论" in name:
            return "chat_summary"
        if "faq" in doc_type or "faq" in name:
            return "faq"
        if "项目" in doc_type or "项目" in name:
            return "project"
        if "制度" in doc_type or "细则" in name or "制度" in name:
            return "policy"
        return "policy"

    @staticmethod
    def _extract_version(file_path: Path, meta: dict[str, str]) -> str:
        version = (meta.get("版本号") or "").strip()
        if version:
            return version
        match = re.search(r"_V(\d+(?:\.\d+)?)", file_path.stem, flags=re.IGNORECASE)
        if match:
            return f"V{match.group(1)}"
        return "unknown"

    @staticmethod
    def _extract_date(raw: str) -> str | None:
        text = raw.strip()
        if not text:
            return None
        text = text.replace("年", "-").replace("月", "-").replace("日", "")
        text = text.replace("/", "-").replace(".", "-")
        text = re.sub(r"\s+", "", text)
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m%d"):
            try:
                dt = datetime.strptime(text, fmt)
                if fmt == "%Y-%m":
                    return dt.strftime("%Y-%m-01")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def _as_datetime(raw_date: str | None) -> str | None:
        if not raw_date:
            return None
        if "T" in raw_date:
            return raw_date
        return f"{raw_date}T00:00:00"

    @staticmethod
    def _build_scopes(file_path: Path, meta: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
        permission = (meta.get("权限级别") or "").strip()

        if not permission or "全员可见" in permission or "公开" in permission:
            role_scope = ["*"]
            department_scope = ["*"]
        else:
            roles: list[str] = []
            depts: list[str] = []
            if "普通员工" in permission or "项目组成员" in permission or "项目组可见" in permission:
                roles.append("employee")
            if "财务" in permission:
                roles.append("finance")
                depts.append("finance")
            if "项目经理" in permission or "PMO" in permission or "项目组" in permission:
                roles.append("pm")
                depts.extend(["operations", "delivery"])
            if "管理员" in permission or "管理者" in permission or "部门负责人" in permission:
                roles.append("pm")
            role_scope = sorted(set(roles)) if roles else ["*"]
            department_scope = sorted(set(depts)) if depts else ["*"]

        name = file_path.stem
        if "北极星" in name:
            project_scope = ["A"]
        elif "项目" in name:
            project_scope = ["A"]
        else:
            project_scope = ["*"]

        return role_scope, department_scope, project_scope

    @staticmethod
    def _build_tags(source_type: str, title: str, file_path: Path) -> list[str]:
        tags = [source_type, "fixtures"]
        name = file_path.stem
        if "差旅" in name or "报销" in name:
            tags.append("报销")
        if "北极星" in name:
            tags.append("项目北极星")
        if "FAQ" in name.upper():
            tags.append("FAQ")
        if "结论" in name:
            tags.append("结论")
        tags.append(title[:16])
        return sorted(set(tags))

    @staticmethod
    def _summarize(body: str) -> str:
        one_line = re.sub(r"\s+", " ", body).strip()
        return one_line[:120]
