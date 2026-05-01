from __future__ import annotations

from app.schemas.knowledge import KnowledgeDocument
from app.schemas.user import UserProfile


class PermissionService:
    def can_access(self, profile: UserProfile, document: KnowledgeDocument) -> bool:
        permission = document.permission_level
        department = profile.department
        project_scope = profile.accessible_projects
        project_related = not document.project_name or document.project_name in project_scope

        if "全员可见" in permission:
            return True
        if "管理员" in permission and self._is_admin(profile):
            return True
        if "财务" in permission and self._is_finance(profile):
            return True
        if "管理者" in permission and self._is_manager(profile):
            return True
        if "部门负责人" in permission and self._is_manager(profile):
            return True
        if "项目组成员" in permission and project_related:
            return True
        if "项目经理" in permission and self._is_pm(profile) and project_related:
            return True
        if "PMO" in permission and department == "PMO" and project_related:
            return True
        if (
            "研发负责人" in permission
            and self._is_manager(profile)
            and department.startswith("研发")
            and project_related
        ):
            return True

        return False

    def filter_accessible(
        self, profile: UserProfile, documents: list[KnowledgeDocument]
    ) -> list[KnowledgeDocument]:
        return [document for document in documents if self.can_access(profile, document)]

    def _is_admin(self, profile: UserProfile) -> bool:
        title = profile.title.lower()
        return profile.role == "admin" or "admin" in title or "管理员" in profile.title

    def _is_finance(self, profile: UserProfile) -> bool:
        department = profile.department.lower()
        title = profile.title.lower()
        return (
            profile.role == "finance"
            or "财务" in profile.department
            or "finance" in department
            or "财务" in profile.title
            or "finance" in title
        )

    def _is_manager(self, profile: UserProfile) -> bool:
        title = profile.title
        lowered_title = title.lower()
        manager_keywords = ["负责人", "总监", "主管", "leader"]
        return (
            self._is_admin(profile)
            or profile.role == "department_head"
            or profile.level.upper().startswith("M")
            or any(keyword in title or keyword in lowered_title for keyword in manager_keywords)
        )

    def _is_pm(self, profile: UserProfile) -> bool:
        title = profile.title
        lowered_title = title.lower()
        return (
            self._is_admin(profile)
            or profile.role == "pm"
            or "项目经理" in title
            or lowered_title == "pm"
            or "project manager" in lowered_title
        )
