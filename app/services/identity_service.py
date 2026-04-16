import json
import os
import re

from app.core.config import settings
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserProfile
from app.services.feishu_contact_service import FeishuContactService

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 依赖缺失时走规则兜底
    OpenAI = None  # type: ignore[assignment]


class IdentityService:
    """提问时实时识别用户身份：角色/部门来自飞书，项目来自问题文本。"""

    _DEPT_ALIASES = (
        ("finance", ("财务", "finance")),
        ("hr", ("人力", "hr")),
        ("operations", ("运营", "operations")),
        ("delivery", ("交付", "delivery")),
        ("engineering", ("研发", "工程", "开发", "engineering")),
    )

    def __init__(
        self, user_repo: UserRepository | None = None, feishu_contact: FeishuContactService | None = None
    ) -> None:
        self._feishu = feishu_contact or FeishuContactService()
        self._user_repo = user_repo or UserRepository()
        self._client = None

    def resolve(
        self,
        *,
        question: str,
        user_id: str | None = None,
        user_profile: UserProfile | None = None,
    ) -> UserProfile:
        if user_profile is not None:
            return user_profile

        base = self._load_base_profile(user_id)
        projects = self._extract_projects_with_llm(question)
        if not projects:
            projects = self._extract_projects_with_rules(question)
        if not projects:
            projects = ["unknown"]

        return UserProfile(
            user_id=base.user_id,
            role=base.role,
            department=base.department,
            projects=projects,
        )

    def _load_base_profile(self, user_id: str | None) -> UserProfile:
        if user_id:
            from_feishu = self._load_from_feishu(user_id)
            if from_feishu is not None:
                return from_feishu

        if user_id:
            try:
                profile = UserProfile.model_validate(self._user_repo.get_profile(user_id))
                return UserProfile(
                    user_id=profile.user_id,
                    role=profile.role or "employee",
                    department=profile.department or "unknown",
                    projects=[],
                )
            except Exception:
                pass
        return UserProfile(user_id=user_id or "anonymous", role="employee", department="unknown", projects=[])

    def _load_from_feishu(self, user_id: str) -> UserProfile | None:
        profile = self._feishu.get_user_profile(user_id)
        if profile is None:
            return None

        department = self._normalize_department(profile.get("department")) or "unknown"
        role = self._infer_role_from_org(
            department=department,
            job_title=str(profile.get("job_title") or ""),
            user_name=str(profile.get("name") or ""),
        )
        normalized_user_id = str(profile.get("user_id") or user_id).strip() or user_id
        return UserProfile(
            user_id=normalized_user_id,
            role=role,
            department=department,
            projects=[],
        )

    def _extract_projects_with_llm(self, question: str) -> list[str]:
        client = self._get_client()
        if client is None:
            return []

        prompt = (
            "你是企业项目识别器。请只从用户问题中提取明确提到的项目。"
            "只输出 JSON，字段为 projects，类型必须为数组。"
            "如果问题未明确提及任何项目，返回空数组。"
        )
        try:
            model_name = settings.chat_model.strip() or "qwen3-max"
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
        except Exception:
            return []
        return self._normalize_projects(payload.get("projects"))

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
    def _infer_role_from_org(*, department: str, job_title: str, user_name: str) -> str:
        dept_text = department.lower()
        title_text = job_title.lower()
        name_text = user_name.lower()
        combined = f"{dept_text} {title_text} {name_text}"

        if any(token in combined for token in ("财务", "finance")):
            return "finance"
        if any(
            token in combined
            for token in ("项目经理", "项目负责人", "pmo", "pm", "管理", "manager", "delivery", "交付")
        ):
            return "pm"
        return "employee"

    def _extract_projects_with_rules(self, question: str) -> list[str]:
        projects: set[str] = set()
        if "北极星" in question:
            projects.add("A")

        for match in re.findall(r"(?:项目|负责)\s*([A-Za-z]\w*)", question):
            projects.add(match.upper())
        for match in re.findall(r"\b([A-Za-z])项目\b", question):
            projects.add(match.upper())
        return sorted(projects)

    def _normalize_department(self, raw: object) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip().lower()
        if not text:
            return None
        for canonical, aliases in self._DEPT_ALIASES:
            if text == canonical or text in aliases:
                return canonical
            if any(alias in text for alias in aliases):
                return canonical
        return text or "unknown"

    def _normalize_projects(self, raw: object) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [part.strip() for part in re.split(r"[，,、\s]+", raw) if part.strip()]
        if not isinstance(raw, list):
            return []
        items: list[str] = []
        for item in raw:
            project = str(item).strip()
            if not project:
                continue
            if project == "北极星":
                items.append("A")
            else:
                items.append(project.upper())
        return sorted(set(items))
