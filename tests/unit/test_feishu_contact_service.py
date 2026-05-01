from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.repositories.user_repository import UserRepository
from app.services.feishu_contact_service import FeishuContactService, MockFeishuContactService


def test_mock_feishu_contact_service_preserves_requested_user_id_type() -> None:
    repository = UserRepository(Path("data/feishu_users.json"))
    service = MockFeishuContactService(repository)

    user = service.get_user_profile("u_employee_li", "open_id")

    assert user.user_id == "u_employee_li"
    assert user.user_id_type == "open_id"
    assert user.source == "mock_feishu_directory"


def test_feishu_contact_service_builds_directory_user_from_feishu_payload() -> None:
    service = FeishuContactService(
        app_id="cli_test",
        app_secret="secret",
    )
    feishu_user = SimpleNamespace(
        open_id="ou_test_user",
        name="周航",
        job_title="项目经理",
    )

    profile = service._build_directory_user(
        requested_user_id="ou_test_user",
        user_id_type="open_id",
        feishu_user=feishu_user,
        department_id="od_xxx",
        department_name="产品部",
        job_level_id="jl_xxx",
        job_level_name="M1",
        fallback_user=None,
    )

    assert profile.user_id == "ou_test_user"
    assert profile.user_id_type == "open_id"
    assert profile.department == "产品部"
    assert profile.level == "M1"
    assert profile.role == "pm"
    assert profile.source == "feishu_contact_api"
