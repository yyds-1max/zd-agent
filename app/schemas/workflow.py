from __future__ import annotations

from typing import TypedDict

from app.schemas.agent_loop import AgentObservation
from app.schemas.answer_strategy import AnswerStrategyResult
from app.schemas.intent import IntentResult
from app.schemas.knowledge import RetrievedChunk
from app.schemas.query import QueryResponse
from app.schemas.task_route import TaskRouteResult
from app.schemas.user import UserProfile
from app.schemas.version import VersionCheckResult, VersionDiffResult


class QueryWorkflowState(TypedDict, total=False):
    user_id: str
    user_id_type: str
    question: str
    routing_question: str | None
    conversation_id: str | None
    top_k: int
    user_profile: UserProfile
    intent: IntentResult
    task_route: TaskRouteResult
    retrieved_chunks: list[RetrievedChunk]
    observations: list[AgentObservation]
    retrieval_iterations: int
    version_checks: list[VersionCheckResult]
    version_diffs: list[VersionDiffResult]
    answer_strategy: AnswerStrategyResult
    tool_trace: list[str]
    response: QueryResponse
