from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.answer_strategy import AnswerStrategyResult
from app.schemas.intent import IntentResult
from app.schemas.knowledge import Citation
from app.schemas.user import UserProfile
from app.schemas.version import VersionCheckResult, VersionDiffResult


class QueryRequest(BaseModel):
    user_id: str
    user_id_type: str = "open_id"
    question: str
    routing_question: str | None = None
    top_k: int = 4
    conversation_id: str | None = None
    use_history: bool = True
    history_limit: int = 6


class QueryResponse(BaseModel):
    question: str
    answer: str
    conversation_id: str | None = None
    contextual_question: str | None = None
    user_profile: UserProfile
    intent: IntentResult
    citations: list[Citation] = Field(default_factory=list)
    version_checks: list[VersionCheckResult] = Field(default_factory=list)
    version_diffs: list[VersionDiffResult] = Field(default_factory=list)
    answer_strategy: AnswerStrategyResult | None = None
    version_notice: str | None = None
    notes: list[str] = Field(default_factory=list)
    tool_trace: list[str] = Field(default_factory=list)
