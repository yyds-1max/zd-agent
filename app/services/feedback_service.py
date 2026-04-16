from app.repositories.feedback_repository import FeedbackRepository
from app.schemas.feedback import FeedbackRequest


class FeedbackService:
    def __init__(self, repo: FeedbackRepository | None = None) -> None:
        self._repo = repo or FeedbackRepository()

    def save(self, feedback: FeedbackRequest) -> dict[str, str]:
        feedback_id = self._repo.save_feedback(feedback.model_dump())
        issue_ids: list[int] = []

        if feedback.is_obsolete:
            issue_ids.append(
                self._repo.upsert_governance_issue(
                    {
                        "issue_key": f"{feedback.query_id}:stale_doc",
                        "query_id": feedback.query_id,
                        "feedback_id": feedback_id,
                        "issue_type": "stale_doc",
                        "severity": "high",
                        "title": "文档疑似过期或版本问题",
                        "detail": "用户反馈命中的知识可能已过期，需要核验版本与生效状态。",
                        "note": feedback.note,
                    }
                )
            )
        if not feedback.helpful:
            issue_ids.append(
                self._repo.upsert_governance_issue(
                    {
                        "issue_key": f"{feedback.query_id}:low_quality_answer",
                        "query_id": feedback.query_id,
                        "feedback_id": feedback_id,
                        "issue_type": "low_quality_answer",
                        "severity": "medium",
                        "title": "答案质量待提升",
                        "detail": "用户反馈答案帮助度不足，需要复核召回结果与答案生成质量。",
                        "note": feedback.note,
                    }
                )
            )

        return {
            "status": "已保存",
            "query_id": feedback.query_id,
            "feedback_id": str(feedback_id),
            "governance_issue_count": str(len(issue_ids)),
        }
