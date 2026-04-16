from app.schemas.citation import Citation
from app.schemas.user import UserProfile


class PermissionService:
    def build_vector_filter(self, user: UserProfile) -> dict:
        clauses = [
            self._scope_clause("role_scope", [user.role]),
            self._scope_clause("department_scope", [user.department]),
            self._scope_clause("project_scope", user.projects),
        ]
        return {"$and": clauses}

    def filter_citations(self, items: list[Citation], user: UserProfile) -> list[Citation]:
        return [item for item in items if self.is_citation_allowed(item, user)]

    def is_citation_allowed(self, item: Citation, user: UserProfile) -> bool:
        role_ok = self._scope_match(item.role_scope, [user.role])
        department_ok = self._scope_match(item.department_scope, [user.department])
        project_ok = self._scope_match(item.project_scope, user.projects)
        return role_ok and department_ok and project_ok

    @staticmethod
    def _scope_clause(field: str, values: list[str]) -> dict:
        checks: list[dict] = [{field: {"$contains": "*"}}]
        for value in values:
            checks.append({field: {"$contains": value}})
        return {"$or": checks}

    @staticmethod
    def _scope_match(scope: list[str], user_values: list[str]) -> bool:
        if not scope:
            return False
        if "*" in scope:
            return True
        return any(value in scope for value in user_values)
