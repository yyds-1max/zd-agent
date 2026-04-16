from datetime import date

from app.schemas.citation import Citation
from app.schemas.intent import QueryIntent
from app.schemas.query import QueryRequest
from app.schemas.user import UserProfile
from app.services.query_graph_service import QueryGraphService


class _FakeIntentParser:
    def __init__(self) -> None:
        self.called_question: str | None = None
        self.called_user_role: str | None = None

    def parse(self, question: str, user: UserProfile) -> QueryIntent:
        self.called_question = question
        self.called_user_role = user.role
        return QueryIntent(
            original_question=question,
            retrieval_query="改写后的检索query",
            intent_type="policy",
            source="llm",
        )


class _FakeRetrieval:
    def __init__(self) -> None:
        self.called_query: str | None = None
        self.called_user_role: str | None = None

    def retrieve(self, question: str, user: UserProfile) -> list[Citation]:
        self.called_query = question
        self.called_user_role = user.role
        return [
            Citation(
                doc_id="d1",
                title="制度文档",
                source_type="policy",
                chunk_text="报销需在 10 天内提交。",
                version="V2.0",
                is_latest=True,
                effective_date=date(2026, 3, 15),
                role_scope=["*"],
                department_scope=["*"],
                project_scope=["*"],
            )
        ]


class _FakePermission:
    def filter_citations(self, items: list[Citation], user: UserProfile) -> list[Citation]:
        return items


class _FakeIdentity:
    def __init__(self) -> None:
        self.called_question: str | None = None

    def resolve(
        self,
        *,
        question: str,
        user_id: str | None = None,
        user_profile: UserProfile | None = None,
    ) -> UserProfile:
        self.called_question = question
        if user_profile:
            return user_profile
        return UserProfile(user_id=user_id or "anonymous", role="employee", department="hr", projects=["A"])


class _FakeAnswer:
    def __init__(self) -> None:
        self.called_intent_type: str | None = None
        self.called_version_hint: str | None = None

    def compose(
        self,
        question: str,
        citations: list[Citation],
        intent_type: str | None = None,
        version_hint: str | None = None,
    ) -> str:
        self.called_intent_type = intent_type
        self.called_version_hint = version_hint
        return "structured answer"


class _FakeVersion:
    def build_version_hint(self, items: list[Citation], question: str | None = None) -> str:
        return "当前版本：V2.0"


class _FakeQueryLogRepo:
    def __init__(self) -> None:
        self.last_row = None

    def save_query_log(self, row):
        self.last_row = row


def test_query_graph_should_connect_intent_retrieve_answer_version() -> None:
    fake_identity = _FakeIdentity()
    fake_intent = _FakeIntentParser()
    fake_retrieval = _FakeRetrieval()
    fake_answer = _FakeAnswer()
    fake_query_log_repo = _FakeQueryLogRepo()
    service = QueryGraphService(
        identity=fake_identity,  # type: ignore[arg-type]
        intent=fake_intent,  # type: ignore[arg-type]
        retrieval=fake_retrieval,  # type: ignore[arg-type]
        permission=_FakePermission(),  # type: ignore[arg-type]
        answer=fake_answer,  # type: ignore[arg-type]
        version=_FakeVersion(),  # type: ignore[arg-type]
        query_log_repo=fake_query_log_repo,  # type: ignore[arg-type]
    )
    payload = QueryRequest(
        question="出差报销最新制度是什么？",
        user_profile=UserProfile(user_id="u1", role="employee", department="hr", projects=["A"]),
    )

    resp = service.run(payload)

    assert fake_identity.called_question == "出差报销最新制度是什么？"
    assert fake_intent.called_question == "出差报销最新制度是什么？"
    assert fake_intent.called_user_role == "employee"
    assert fake_retrieval.called_query == "改写后的检索query"
    assert fake_retrieval.called_user_role == "employee"
    assert fake_answer.called_intent_type == "policy"
    assert fake_answer.called_version_hint == "当前版本：V2.0"
    assert resp.answer == "structured answer"
    assert resp.version_hint == "当前版本：V2.0"
    assert len(resp.citations) == 1
    assert resp.query_id
    assert fake_query_log_repo.last_row is not None
    assert fake_query_log_repo.last_row["query_id"] == resp.query_id
    assert fake_query_log_repo.last_row["citation_doc_ids"] == ["d1"]
