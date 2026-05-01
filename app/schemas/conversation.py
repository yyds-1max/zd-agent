from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)


class ConversationRewriteResult(BaseModel):
    standalone_question: str
    needs_clarification: bool = False
    referenced_topic: str | None = None
    rewrite_reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "rule"
