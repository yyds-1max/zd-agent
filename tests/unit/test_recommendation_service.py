from app.schemas.user import UserProfile
from app.services.recommendation_service import RecommendationService


class _FakeRecommendationRepo:
    def __init__(self) -> None:
        self.push_logs = []

    def list_latest_knowledge(self, limit: int = 300):
        return [
            {
                "doc_id": "d-policy-1",
                "title": "差旅报销最新标准说明",
                "source_type": "policy",
                "version": "V2.0",
                "is_latest": 1,
                "effective_date": "2026-03-15",
                "updated_at": "2026-04-10T10:00:00",
                "summary": "报销时效由 5 个工作日调整为 10 个自然日。",
                "tags": '["报销","差旅","policy"]',
                "role_scope": '["*"]',
                "department_scope": '["*"]',
                "project_scope": '["*"]',
            },
            {
                "doc_id": "d-policy-finance",
                "title": "财务专用报销审批细则",
                "source_type": "policy",
                "version": "V1.0",
                "is_latest": 1,
                "effective_date": "2026-04-01",
                "updated_at": "2026-04-11T10:00:00",
                "summary": "仅财务角色可见。",
                "tags": '["财务","报销"]',
                "role_scope": '["finance"]',
                "department_scope": '["finance"]',
                "project_scope": '["*"]',
            },
            {
                "doc_id": "d-project-a",
                "title": "项目A交付节点说明",
                "source_type": "project",
                "version": "V1.2",
                "is_latest": 1,
                "effective_date": "2026-01-10",
                "updated_at": "2026-02-01T09:00:00",
                "summary": "项目A里程碑规划。",
                "tags": '["项目A","交付"]',
                "role_scope": '["employee","pm"]',
                "department_scope": '["*"]',
                "project_scope": '["A"]',
            },
        ]

    def list_recent_queries(self, user_id: str, limit: int = 30):
        return [
            {
                "query_id": "q1",
                "question": "出差报销标准是什么",
                "retrieval_query": "报销 标准 最新",
                "intent_type": "policy",
                "citation_doc_ids": '["d-policy-1"]',
                "created_at": "2026-04-15T10:00:00",
            }
        ]

    def list_recent_clicks(self, user_id: str, limit: int = 30):
        return [
            {
                "id": 1,
                "query_id": "q1",
                "doc_id": "d-policy-1",
                "title": "差旅报销最新标准说明",
                "action": "open_citation",
                "position": 1,
                "created_at": "2026-04-15T10:01:00",
            }
        ]

    def save_push_log(self, item):
        self.push_logs.append(item)
        return 42


def test_recommend_should_filter_by_permission_and_rank_by_behavior() -> None:
    repo = _FakeRecommendationRepo()
    service = RecommendationService(repo=repo)  # type: ignore[arg-type]
    user = UserProfile(user_id="u-employee", role="employee", department="hr", projects=["A"])

    items = service.recommend(user, top_k=5)
    ids = [item.doc_id for item in items]

    assert "d-policy-finance" not in ids
    assert ids[0] == "d-policy-1"
    assert any("匹配最近搜索主题" in reason for reason in items[0].reasons)


def test_trigger_push_should_store_push_log_and_return_push_id() -> None:
    repo = _FakeRecommendationRepo()
    service = RecommendationService(repo=repo)  # type: ignore[arg-type]
    user = UserProfile(user_id="u-employee", role="employee", department="hr", projects=["A"])

    resp = service.trigger_push(user=user, top_k=3, channel="manual")

    assert resp.status == "triggered"
    assert resp.push_id == "42"
    assert resp.item_count >= 1
    assert len(repo.push_logs) == 1
    assert repo.push_logs[0]["user_id"] == "u-employee"
