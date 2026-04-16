from app.schemas.citation import Citation
from app.schemas.user import UserProfile
from app.services.permission_service import PermissionService


def test_build_vector_filter_should_contain_three_scope_clauses() -> None:
    service = PermissionService()
    user = UserProfile(user_id="u1", role="employee", department="hr", projects=["A"])

    where = service.build_vector_filter(user)

    assert "$and" in where
    assert len(where["$and"]) == 3
    assert where["$and"][0]["$or"][1] == {"role_scope": {"$contains": "employee"}}
    assert where["$and"][1]["$or"][1] == {"department_scope": {"$contains": "hr"}}
    assert where["$and"][2]["$or"][1] == {"project_scope": {"$contains": "A"}}


def test_filter_citations_should_block_unauthorized_items() -> None:
    service = PermissionService()
    user = UserProfile(user_id="u1", role="employee", department="hr", projects=["A"])
    allowed = Citation(
        doc_id="d1",
        title="allow",
        source_type="policy",
        chunk_text="ok",
        version="V1",
        role_scope=["*"],
        department_scope=["*"],
        project_scope=["*"],
    )
    blocked = Citation(
        doc_id="d2",
        title="block",
        source_type="policy",
        chunk_text="no",
        version="V1",
        role_scope=["finance"],
        department_scope=["finance"],
        project_scope=["B"],
    )

    visible = service.filter_citations([allowed, blocked], user)

    assert [item.doc_id for item in visible] == ["d1"]
