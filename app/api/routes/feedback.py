from fastapi import APIRouter

from app.schemas.feedback import FeedbackRequest
from app.services.feedback_service import FeedbackService

router = APIRouter()
service = FeedbackService()


@router.post("")
def save_feedback(payload: FeedbackRequest) -> dict[str, str]:
    return service.save(payload)
