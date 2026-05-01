from __future__ import annotations

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    name: str
    confidence: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    project_names: list[str] = Field(default_factory=list)
    version_sensitive: bool = False
    reasoning: str = ""
