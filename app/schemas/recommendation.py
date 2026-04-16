from pydantic import BaseModel, Field

from app.schemas.user import UserProfile


class RecommendationItem(BaseModel):
    doc_id: str
    title: str
    source_type: str
    version: str
    updated_at: str | None = None
    summary: str | None = None
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    items: list[RecommendationItem] = Field(default_factory=list)


class PushTriggerRequest(BaseModel):
    user_profile: UserProfile
    top_k: int = Field(default=5, ge=1, le=20)
    channel: str = Field(default="manual")


class PushTriggerResponse(BaseModel):
    status: str
    push_id: str | None = None
    user_id: str
    item_count: int = 0
    items: list[RecommendationItem] = Field(default_factory=list)
    message: str | None = None
