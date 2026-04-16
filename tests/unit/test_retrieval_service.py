from typing import Any

from app.schemas.user import UserProfile
from app.services.retrieval_service import RetrievalService


class _FakeRepo:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.captured_where: dict[str, Any] | None = None

    def query_chunks(
        self, question: str, where: dict[str, Any] | None = None, n_results: int = 8
    ) -> list[dict[str, Any]]:
        self.captured_where = where
        return self.rows


def test_retrieve_should_pass_permission_filter_to_vector_query() -> None:
    user = UserProfile(user_id="u1", role="employee", department="hr", projects=["A"])
    row = {
        "document": "content",
        "metadata": {
            "doc_id": "d1",
            "title": "doc",
            "source_type": "policy",
            "version": "V1",
            "is_latest": True,
            "role_scope": ["*"],
            "department_scope": ["*"],
            "project_scope": ["*"],
        },
    }
    service = RetrievalService()
    fake_repo = _FakeRepo([row])
    service._repo = fake_repo  # type: ignore[assignment]

    citations = service.retrieve("test", user)

    expected_where = service._permission.build_vector_filter(user)  # type: ignore[attr-defined]
    assert fake_repo.captured_where == expected_where
    assert len(citations) == 1
    assert citations[0].doc_id == "d1"


def test_retrieve_should_defensively_filter_unauthorized_rows() -> None:
    user = UserProfile(user_id="u1", role="employee", department="hr", projects=["A"])
    unauthorized_row = {
        "document": "secret",
        "metadata": {
            "doc_id": "d-secret",
            "title": "secret-doc",
            "source_type": "policy",
            "version": "V1",
            "is_latest": True,
            "role_scope": ["finance"],
            "department_scope": ["finance"],
            "project_scope": ["B"],
        },
    }
    service = RetrievalService()
    service._repo = _FakeRepo([unauthorized_row])  # type: ignore[assignment]

    citations = service.retrieve("test", user)

    assert citations == []
