from app.core.config import settings
from app.services.feishu_contact_service import FeishuContactService


def test_get_user_profile_should_call_user_and_department_api() -> None:
    service = FeishuContactService()
    old_token = settings.feishu_tenant_access_token
    settings.feishu_tenant_access_token = "t-test"
    called_urls: list[str] = []

    def fake_request_json(method: str, url: str, token=None, payload=None):  # type: ignore[no-untyped-def]
        called_urls.append(url)
        if "/contact/v3/users/" in url:
            return {
                "code": 0,
                "data": {
                    "user": {
                        "user_id": "u1",
                        "name": "张三",
                        "job_title": "项目经理",
                        "department_ids": ["od-100"],
                    }
                },
            }
        if "/contact/v3/departments/" in url:
            return {"code": 0, "data": {"department": {"name": "交付部"}}}
        return None

    try:
        service._request_json = fake_request_json  # type: ignore[method-assign]
        profile = service.get_user_profile("u1")
    finally:
        settings.feishu_tenant_access_token = old_token

    assert profile is not None
    assert profile["user_id"] == "u1"
    assert profile["department"] == "交付部"
    assert any("/contact/v3/users/" in url for url in called_urls)
    assert any("/contact/v3/departments/" in url for url in called_urls)


def test_get_user_profile_should_return_none_when_user_api_failed() -> None:
    service = FeishuContactService()
    old_token = settings.feishu_tenant_access_token
    settings.feishu_tenant_access_token = "t-test"

    try:
        service._request_json = lambda method, url, token=None, payload=None: {"code": 999, "msg": "failed"}  # type: ignore[method-assign]
        profile = service.get_user_profile("u1")
    finally:
        settings.feishu_tenant_access_token = old_token

    assert profile is None
