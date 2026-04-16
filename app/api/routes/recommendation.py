from fastapi import APIRouter

from app.schemas.recommendation import (
    PushTriggerRequest,
    PushTriggerResponse,
    RecommendationResponse,
)
from app.schemas.user import UserProfile
from app.services.recommendation_service import RecommendationService

router = APIRouter()
service = RecommendationService()


@router.post("", response_model=RecommendationResponse)
def get_recommendation(user: UserProfile) -> RecommendationResponse:
    return RecommendationResponse(items=service.recommend(user))


@router.post("/push/trigger", response_model=PushTriggerResponse)
def trigger_push(payload: PushTriggerRequest) -> PushTriggerResponse:
    return service.trigger_push(
        user=payload.user_profile,
        top_k=payload.top_k,
        channel=payload.channel,
    )
