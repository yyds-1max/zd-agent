from __future__ import annotations

from app.schemas.knowledge import KnowledgeDocument
from app.schemas.user import UserProfile
from app.services.permission_service import PermissionService


def _build_document(permission_level: str, project_name: str | None = None) -> KnowledgeDocument:
    return KnowledgeDocument(
        doc_id=f"doc-{permission_level}",
        title="测试文档",
        doc_type="policy",
        topic="测试主题",
        permission_level=permission_level,
        project_name=project_name,
        source_path="tests/fixture.txt",
        body="测试内容",
    )


def _build_profile(
    *,
    department: str,
    title: str,
    level: str,
    role: str = "employee",
    projects: list[str] | None = None,
    managed_projects: list[str] | None = None,
) -> UserProfile:
    return UserProfile(
        user_id="ou_test",
        user_id_type="open_id",
        name="测试用户",
        department=department,
        title=title,
        level=level,
        role=role,
        projects=projects or [],
        managed_projects=managed_projects or [],
    )


def test_finance_department_can_access_finance_documents_without_explicit_role() -> None:
    profile = _build_profile(department="财务部", title="财务BP", level="P3")
    document = _build_document("财务可见")

    assert PermissionService().can_access(profile, document) is True


def test_manager_level_can_access_management_documents_without_explicit_role() -> None:
    profile = _build_profile(department="市场部", title="市场经理", level="M1")
    document = _build_document("管理者可见")

    assert PermissionService().can_access(profile, document) is True


def test_pm_title_can_access_project_manager_documents_when_project_matches() -> None:
    profile = _build_profile(
        department="产品部",
        title="项目经理",
        level="P5",
        projects=["北极星"],
    )
    document = _build_document("项目经理可见", project_name="北极星")

    assert PermissionService().can_access(profile, document) is True
