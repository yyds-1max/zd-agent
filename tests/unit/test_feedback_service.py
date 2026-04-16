from app.schemas.feedback import FeedbackRequest
from app.services.feedback_service import FeedbackService


class _FakeFeedbackRepo:
    def __init__(self) -> None:
        self.saved_feedback = []
        self.saved_issues = []
        self._feedback_seq = 0
        self._issue_seq = 0

    def save_feedback(self, item):
        self._feedback_seq += 1
        self.saved_feedback.append(item)
        return self._feedback_seq

    def upsert_governance_issue(self, item):
        self._issue_seq += 1
        self.saved_issues.append(item)
        return self._issue_seq


def test_save_should_only_persist_feedback_when_helpful_and_not_obsolete() -> None:
    repo = _FakeFeedbackRepo()
    service = FeedbackService(repo=repo)  # type: ignore[arg-type]

    resp = service.save(
        FeedbackRequest(
            query_id="q-1",
            helpful=True,
            is_obsolete=False,
            note="回答有帮助",
        )
    )

    assert resp["status"] == "已保存"
    assert resp["query_id"] == "q-1"
    assert resp["governance_issue_count"] == "0"
    assert len(repo.saved_feedback) == 1
    assert repo.saved_issues == []


def test_save_should_create_low_quality_issue_when_not_helpful() -> None:
    repo = _FakeFeedbackRepo()
    service = FeedbackService(repo=repo)  # type: ignore[arg-type]

    resp = service.save(
        FeedbackRequest(
            query_id="q-2",
            helpful=False,
            is_obsolete=False,
            note="答案太泛",
        )
    )

    assert resp["governance_issue_count"] == "1"
    assert len(repo.saved_issues) == 1
    assert repo.saved_issues[0]["issue_type"] == "low_quality_answer"
    assert repo.saved_issues[0]["issue_key"] == "q-2:low_quality_answer"


def test_save_should_create_two_issues_when_not_helpful_and_obsolete() -> None:
    repo = _FakeFeedbackRepo()
    service = FeedbackService(repo=repo)  # type: ignore[arg-type]

    resp = service.save(
        FeedbackRequest(
            query_id="q-3",
            helpful=False,
            is_obsolete=True,
            note="内容不准且像旧版",
        )
    )

    assert resp["governance_issue_count"] == "2"
    assert len(repo.saved_issues) == 2
    issue_types = {item["issue_type"] for item in repo.saved_issues}
    assert issue_types == {"stale_doc", "low_quality_answer"}
