from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfileLLMOutput(BaseModel):
    intent_name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    project_names: list[str] = Field(default_factory=list)
    version_sensitive: bool = False
    reasoning: str = ""
    ambiguity_note: str | None = None


class AgentAnswerLLMOutput(BaseModel):
    answer_markdown: str
    cited_doc_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ConversationIntentLLMOutput(BaseModel):
    intent_name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    should_retrieve: bool = True
    direct_answer: str | None = None
    reasoning: str = ""


class TaskRouterLLMOutput(BaseModel):
    route_name: str
    intent_name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    should_retrieve: bool = True
    direct_answer: str | None = None
    reasoning: str = ""


class ConversationRewriteLLMOutput(BaseModel):
    standalone_question: str
    needs_clarification: bool = False
    referenced_topic: str | None = None
    rewrite_reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AgentActionLLMOutput(BaseModel):
    action: str = "finalize"
    action_query: str | None = None
    evidence_sufficient: bool = True
    thought_summary: str = ""
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VersionDiffLLMOutput(BaseModel):
    change_type: str
    summary: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    key_changes: list[str] = Field(default_factory=list)
