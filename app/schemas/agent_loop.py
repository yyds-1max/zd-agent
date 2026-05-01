from __future__ import annotations

from pydantic import BaseModel, Field


class AgentActionDecision(BaseModel):
    action: str = "finalize"
    action_query: str | None = None
    evidence_sufficient: bool = True
    thought_summary: str = ""
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "rule"


class AgentObservation(BaseModel):
    iteration: int
    action: str
    query: str | None = None
    added_chunks: int = 0
    total_chunks: int = 0
    summary: str = ""
