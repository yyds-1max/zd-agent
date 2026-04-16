import json
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from app.core.config import settings


class FeishuContactService:
    """封装飞书通讯录 API：获取用户信息与所属部门。"""

    def __init__(self) -> None:
        self._cached_tenant_token: str | None = None
        self._token_expire_at: float = 0.0

    def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        token = self._get_tenant_access_token()
        if not token:
            return None

        user = self._get_user(user_id=user_id, token=token)
        if not user:
            return None

        department_name = self._resolve_department_name(user.get("department_ids"), token)
        return {
            "user_id": str(user.get("user_id") or user_id),
            "name": str(user.get("name") or ""),
            "job_title": str(user.get("job_title") or ""),
            "department": department_name,
            "department_ids": user.get("department_ids") or [],
        }

    def _get_tenant_access_token(self) -> str | None:
        direct_token = settings.feishu_tenant_access_token.strip()
        if direct_token:
            return direct_token

        now = time.time()
        if self._cached_tenant_token and now < self._token_expire_at:
            return self._cached_tenant_token

        app_id = settings.feishu_app_id.strip()
        app_secret = settings.feishu_app_secret.strip()
        if not app_id or not app_secret:
            return None

        url = f"{settings.feishu_base_url.rstrip('/')}/open-apis/auth/v3/tenant_access_token/internal"
        payload = {"app_id": app_id, "app_secret": app_secret}
        response = self._request_json("POST", url, payload=payload)
        if not response:
            return None
        if int(response.get("code", -1)) != 0:
            return None

        token = str(response.get("tenant_access_token") or "").strip()
        expire = int(response.get("expire", 0) or 0)
        if not token:
            return None

        # 提前 2 分钟刷新，避免边界失效。
        self._cached_tenant_token = token
        self._token_expire_at = now + max(expire - 120, 60)
        return token

    def _get_user(self, *, user_id: str, token: str) -> dict[str, Any] | None:
        query = urlencode(
            {
                "user_id_type": settings.feishu_user_id_type,
                "department_id_type": settings.feishu_department_id_type,
            }
        )
        url = (
            f"{settings.feishu_base_url.rstrip('/')}/open-apis/contact/v3/users/{quote(user_id, safe='')}"
            f"?{query}"
        )
        response = self._request_json("GET", url, token=token)
        if not response:
            return None
        if int(response.get("code", -1)) != 0:
            return None
        data = response.get("data") or {}
        user = data.get("user") or {}
        if not isinstance(user, dict):
            return None
        return user

    def _resolve_department_name(self, department_ids: Any, token: str) -> str:
        if not isinstance(department_ids, list) or not department_ids:
            return "unknown"

        department_id = str(department_ids[0]).strip()
        if not department_id:
            return "unknown"

        query = urlencode(
            {
                "department_id_type": settings.feishu_department_id_type,
                "user_id_type": settings.feishu_user_id_type,
            }
        )
        url = (
            f"{settings.feishu_base_url.rstrip('/')}/open-apis/contact/v3/departments/"
            f"{quote(department_id, safe='')}?{query}"
        )
        response = self._request_json("GET", url, token=token)
        if not response:
            return "unknown"
        if int(response.get("code", -1)) != 0:
            return "unknown"

        data = response.get("data") or {}
        department = data.get("department") or {}
        name = str(department.get("name") or "").strip()
        return name or "unknown"

    def _request_json(
        self, method: str, url: str, token: str | None = None, payload: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        body: bytes | None = None
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        request = Request(url=url, method=method.upper(), headers=headers, data=body)
        try:
            with urlopen(request, timeout=settings.feishu_http_timeout) as response:
                raw = response.read().decode("utf-8")
        except Exception:
            return None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed
