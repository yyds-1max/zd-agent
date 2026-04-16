from app.pipelines.query_pipeline import QueryPipeline
from app.schemas.query import QueryRequest
from app.schemas.query import QueryResponse
from app.schemas.user import UserProfile


class _FakeGraph:
    def __init__(self) -> None:
        self.called_question: str | None = None
    
    def run(self, payload: QueryRequest):
        self.called_question = payload.question
        return QueryResponse(query_id="q-test", answer="ok", version_hint=None, citations=[])


def test_query_pipeline_should_delegate_to_graph() -> None:
    pipeline = QueryPipeline()
    fake_graph = _FakeGraph()
    pipeline.graph = fake_graph  # type: ignore[assignment]

    payload = QueryRequest(
        question="出差报销怎么走？",
        user_profile=UserProfile(user_id="u1", role="employee", department="hr", projects=["A"]),
    )
    resp = pipeline.run(payload)

    assert resp.answer == "ok"
    assert fake_graph.called_question == "出差报销怎么走？"
