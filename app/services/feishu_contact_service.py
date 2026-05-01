from __future__ import annotations

from typing import Any, Protocol

from app.repositories.user_repository import UserRepository
from app.schemas.user import DirectoryUser


class ContactService(Protocol):
    def get_user_profile(self, user_id: str, user_id_type: str = "open_id") -> DirectoryUser:
        ...


class FeishuContactServiceError(RuntimeError):
    pass


class MockFeishuContactService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def get_user_profile(self, user_id: str, user_id_type: str = "open_id") -> DirectoryUser:
        user = self.user_repository.get_by_user_id(user_id)
        return user.model_copy(
            update={
                "user_id_type": user_id_type,
                "source": "mock_feishu_directory",
            }
        )


class FeishuContactService:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        log_level: str = "INFO",
        default_user_id_type: str = "open_id",
        user_repository: UserRepository | None = None,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.log_level = log_level.upper()
        self.default_user_id_type = default_user_id_type
        self.user_repository = user_repository
        self._client: Any | None = None

    def get_user_profile(self, user_id: str, user_id_type: str = "open_id") -> DirectoryUser:
        effective_user_id_type = user_id_type or self.default_user_id_type
        try:
            return self._fetch_user_profile(user_id, effective_user_id_type)
        except Exception as exc:
            fallback_user = self._fallback_user(user_id, effective_user_id_type)
            if fallback_user is not None:
                return fallback_user
            raise FeishuContactServiceError(
                f"Failed to load user profile from Feishu for `{user_id}` ({effective_user_id_type})."
            ) from exc

    def _fetch_user_profile(self, user_id: str, user_id_type: str) -> DirectoryUser:
        lark, contact_api = self._import_lark_modules()
        client = self._get_client(lark)
        GetDepartmentRequest = contact_api.GetDepartmentRequest
        GetJobLevelRequest = contact_api.GetJobLevelRequest
        GetUserRequest = contact_api.GetUserRequest

        request = (
            GetUserRequest.builder()
            .user_id(user_id)
            .user_id_type(user_id_type)
            .build()
        )
        response = client.contact.v3.user.get(request)
        if not response.success():
            raise FeishuContactServiceError(
                self._format_error(
                    "GetUser",
                    response.code,
                    response.msg,
                    response.get_log_id(),
                )
            )

        user = response.data.user
        department_ids = list(getattr(user, "department_ids", []) or [])
        department_id = department_ids[0] if department_ids else None
        job_level_id = getattr(user, "job_level_id", None)

        department_name = None
        if department_id:
            department_request = GetDepartmentRequest.builder().department_id(department_id).build()
            department_response = client.contact.v3.department.get(department_request)
            if department_response.success() and department_response.data.department is not None:
                department_name = getattr(department_response.data.department, "name", None)

        job_level_name = None
        if job_level_id:
            job_level_request = GetJobLevelRequest.builder().job_level_id(job_level_id).build()
            job_level_response = client.contact.v3.job_level.get(job_level_request)
            if job_level_response.success() and job_level_response.data.job_level is not None:
                job_level_name = getattr(job_level_response.data.job_level, "name", None)

        fallback_user = self.user_repository.find_by_user_id(user_id) if self.user_repository else None
        return self._build_directory_user(
            requested_user_id=user_id,
            user_id_type=user_id_type,
            feishu_user=user,
            department_id=department_id,
            department_name=department_name,
            job_level_id=job_level_id,
            job_level_name=job_level_name,
            fallback_user=fallback_user,
        )

    def _fallback_user(self, user_id: str, user_id_type: str) -> DirectoryUser | None:
        if self.user_repository is None:
            return None
        user = self.user_repository.find_by_user_id(user_id)
        if user is None:
            return None
        return user.model_copy(
            update={
                "user_id_type": user_id_type,
                "source": "mock_feishu_directory",
            }
        )

    def _build_directory_user(
        self,
        *,
        requested_user_id: str,
        user_id_type: str,
        feishu_user: Any,
        department_id: str | None,
        department_name: str | None,
        job_level_id: str | None,
        job_level_name: str | None,
        fallback_user: DirectoryUser | None,
    ) -> DirectoryUser:
        resolved_user_id = self._first_non_empty(
            self._user_id_from_payload(feishu_user, user_id_type),
            requested_user_id,
        )
        name = self._first_non_empty(
            getattr(feishu_user, "name", None),
            fallback_user.name if fallback_user else None,
            requested_user_id,
        )
        title = self._first_non_empty(
            getattr(feishu_user, "job_title", None),
            getattr(feishu_user, "title", None),
            fallback_user.title if fallback_user else None,
            "未知岗位",
        )
        level = self._first_non_empty(
            job_level_name,
            getattr(feishu_user, "job_level", None),
            job_level_id,
            fallback_user.level if fallback_user else None,
            "未知职级",
        )
        department = self._first_non_empty(
            department_name,
            fallback_user.department if fallback_user else None,
            "未知部门",
        )
        role = self._infer_role(
            department=department,
            title=title,
            level=level,
            fallback_role=fallback_user.role if fallback_user else None,
        )

        return DirectoryUser(
            user_id=resolved_user_id,
            user_id_type=user_id_type,
            name=name,
            department=department,
            department_id=department_id,
            title=title,
            level=level,
            job_level_id=job_level_id,
            role=role,
            projects=list(fallback_user.projects) if fallback_user else [],
            managed_projects=list(fallback_user.managed_projects) if fallback_user else [],
            is_new_hire=fallback_user.is_new_hire if fallback_user else False,
            source="feishu_contact_api",
        )

    def _get_client(self, lark: Any) -> Any:
        if self._client is None:
            self._client = (
                lark.Client.builder()
                .app_id(self.app_id)
                .app_secret(self.app_secret)
                .log_level(getattr(lark.LogLevel, self.log_level, lark.LogLevel.INFO))
                .build()
            )
        return self._client

    def _import_lark_modules(self) -> tuple[Any, Any]:
        try:
            import lark_oapi as lark
            from lark_oapi.api.contact import v3 as contact_v3
        except ModuleNotFoundError as exc:
            raise FeishuContactServiceError(
                "Missing dependency `lark-oapi`. Install it before enabling Feishu contact API."
            ) from exc
        return lark, contact_v3

    def _user_id_from_payload(self, feishu_user: Any, user_id_type: str) -> str | None:
        if user_id_type == "open_id":
            return getattr(feishu_user, "open_id", None)
        if user_id_type == "union_id":
            return getattr(feishu_user, "union_id", None)
        if user_id_type == "user_id":
            return getattr(feishu_user, "user_id", None)
        return getattr(feishu_user, user_id_type, None)

    def _infer_role(
        self,
        *,
        department: str,
        title: str,
        level: str,
        fallback_role: str | None,
    ) -> str:
        department_text = department.lower()
        title_text = title.lower()
        level_text = level.upper()

        if "admin" in title_text or "管理员" in title:
            inferred_role = "admin"
        elif "财务" in department or "finance" in department_text or "财务" in title:
            inferred_role = "finance"
        elif "项目经理" in title or title_text == "pm" or "project manager" in title_text:
            inferred_role = "pm"
        elif any(keyword in title for keyword in ["负责人", "总监", "主管", "leader", "Leader"]):
            inferred_role = "department_head"
        elif level_text.startswith("M"):
            inferred_role = "department_head"
        else:
            inferred_role = "employee"

        if inferred_role == "employee" and fallback_role:
            return fallback_role
        return inferred_role

    def _first_non_empty(self, *values: str | None) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _format_error(self, action: str, code: Any, message: Any, log_id: Any) -> str:
        return f"{action} failed, code={code}, msg={message}, log_id={log_id}"
