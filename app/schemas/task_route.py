from __future__ import annotations

from pydantic import BaseModel, Field


class TaskRouteResult(BaseModel):
    route_name: str
    intent_name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    should_retrieve: bool = True
    direct_answer: str | None = None
    source: str = "rule"
    reasoning: str = ""
