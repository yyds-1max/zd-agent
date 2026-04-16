from app.schemas.user import UserProfile
from app.services.identity_service import IdentityService


class _FakeUserRepo:
    def __init__(self, profile: dict) -> None:
        self._profile = profile
        self.called_user_id: str | None = None

    def get_profile(self, user_id: str) -> dict:
        self.called_user_id = user_id
        return {"user_id": user_id, **self._profile}


class _FakeFeishu:
    def __init__(self, profile: dict | None) -> None:
        self.profile = profile
        self.called_user_id: str | None = None

    def get_user_profile(self, user_id: str) -> dict | None:
        self.called_user_id = user_id
        return self.profile


def test_resolve_should_return_given_user_profile_first() -> None:
    service = IdentityService(
        user_repo=_FakeUserRepo({"role": "finance", "department": "finance", "projects": []}),  # type: ignore[arg-type]
        feishu_contact=_FakeFeishu(None),  # type: ignore[arg-type]
    )
    profile = UserProfile(user_id="u1", role="pm", department="delivery", projects=["A"])

    resolved = service.resolve(question="我是财务", user_profile=profile)

    assert resolved == profile


def test_resolve_should_use_feishu_profile_and_default_unknown_project() -> None:
    service = IdentityService(
        user_repo=_FakeUserRepo({"role": "employee", "department": "general", "projects": []}),  # type: ignore[arg-type]
        feishu_contact=_FakeFeishu(
            {
                "user_id": "u2",
                "name": "张三",
                "job_title": "财务专员",
                "department": "财务部",
                "department_ids": ["od-1"],
            }
        ),  # type: ignore[arg-type]
    )
    service._extract_projects_with_llm = lambda question: []  # type: ignore[method-assign]

    resolved = service.resolve(question="本周报销制度有更新吗？", user_id="u2")

    assert resolved.user_id == "u2"
    assert resolved.role == "finance"
    assert resolved.department == "finance"
    assert resolved.projects == ["unknown"]


def test_resolve_should_fallback_to_repo_when_feishu_unavailable() -> None:
    repo = _FakeUserRepo({"role": "employee", "department": "hr", "projects": []})
    service = IdentityService(user_repo=repo, feishu_contact=_FakeFeishu(None))  # type: ignore[arg-type]
    service._extract_projects_with_llm = lambda question: ["A"]  # type: ignore[method-assign]

    resolved = service.resolve(question="我负责 A 项目，帮我总结本周进展。", user_id="u-employee")

    assert repo.called_user_id == "u-employee"
    assert resolved.role == "employee"
    assert resolved.department == "hr"
    assert resolved.projects == ["A"]
