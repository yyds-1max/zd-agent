from __future__ import annotations

from uuid import uuid4
from typing import Any, TypedDict

from app.repositories.query_log_repository import QueryLogRepository
from app.schemas.citation import Citation
from app.schemas.intent import QueryIntent
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.user import UserProfile
from app.services.answer_service import AnswerService
from app.services.identity_service import IdentityService
from app.services.intent_parser_service import IntentParserService
from app.services.permission_service import PermissionService
from app.services.retrieval_service import RetrievalService
from app.services.version_service import VersionService

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - 依赖缺失时使用顺序执行兜底
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]


class QueryGraphState(TypedDict, total=False):
    """LangGraph 共享状态：各节点通过读写这些字段进行数据传递。"""

    request: QueryRequest  # 初始输入：包含 question 和 user_profile。
    resolved_user: UserProfile  # identity 节点输出：用于后续链路的最终用户画像。
    intent: QueryIntent  # intent 节点输出：意图分类、改写检索词、关键词等。
    retrieval_query: str  # intent 节点输出：用于召回的最终检索 query。
    citations: list[Citation]  # retrieve 节点输出：最终可见引用（供后续节点使用）。
    version_hint: str | None  # version 节点输出：版本提示文案（如“当前版本：V2.0”）。
    answer: str  # answer 节点输出：最终结构化回答文本。


class QueryGraphService:
    def __init__(
        self,
        *,
        identity: IdentityService | None = None,
        intent: IntentParserService | None = None,
        retrieval: RetrievalService | None = None,
        permission: PermissionService | None = None,
        answer: AnswerService | None = None,
        version: VersionService | None = None,
        query_log_repo: QueryLogRepository | None = None,
    ) -> None:
        self.identity = identity or IdentityService()
        self.intent = intent or IntentParserService()
        self.retrieval = retrieval or RetrievalService()
        self.permission = permission or PermissionService()
        self.answer = answer or AnswerService()
        self.version = version or VersionService()
        self.query_log_repo = query_log_repo or QueryLogRepository()
        self._compiled_graph = self._build_graph()

    def run(self, payload: QueryRequest) -> QueryResponse:
        initial: QueryGraphState = {"request": payload}
        if self._compiled_graph is not None:
            result = self._compiled_graph.invoke(initial)
        else:
            result = self._run_fallback(initial)
        query_id = uuid4().hex
        self._log_query(query_id=query_id, state=result)
        return QueryResponse(
            query_id=query_id,
            answer=result.get("answer", ""),
            version_hint=result.get("version_hint"),
            citations=result.get("citations", []),
        )

    def _build_graph(self):
        if StateGraph is None:
            return None
        graph = StateGraph(QueryGraphState)
        graph.add_node("identity", self._identity_node)
        graph.add_node("intent", self._intent_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("version", self._version_node)
        graph.add_node("answer", self._answer_node)

        graph.add_edge(START, "identity")
        graph.add_edge("identity", "intent")
        graph.add_edge("intent", "retrieve")
        graph.add_edge("retrieve", "version")
        graph.add_edge("version", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _run_fallback(self, state: QueryGraphState) -> QueryGraphState:
        state.update(self._identity_node(state))
        state.update(self._intent_node(state))
        state.update(self._retrieve_node(state))
        state.update(self._version_node(state))
        state.update(self._answer_node(state))
        return state

    def _identity_node(self, state: QueryGraphState) -> dict[str, Any]:
        request = state["request"]
        resolved_user = self.identity.resolve(
            question=request.question,
            user_id=request.user_id,
            user_profile=request.user_profile,
        )
        return {"resolved_user": resolved_user}

    def _intent_node(self, state: QueryGraphState) -> dict[str, Any]:
        request = state["request"]
        user = state["resolved_user"]
        intent = self.intent.parse(request.question, user)
        retrieval_query = intent.retrieval_query or request.question
        return {"intent": intent, "retrieval_query": retrieval_query}

    def _retrieve_node(self, state: QueryGraphState) -> dict[str, Any]:
        request = state["request"]
        user = state["resolved_user"]
        query = state.get("retrieval_query", request.question)
        retrieved = self.retrieval.retrieve(query, user)
        citations = self.permission.filter_citations(retrieved, user)
        return {"citations": citations}

    def _version_node(self, state: QueryGraphState) -> dict[str, Any]:
        citations = state.get("citations", [])
        request = state["request"]
        version_hint = self.version.build_version_hint(citations, question=request.question)
        return {"version_hint": version_hint}

    def _answer_node(self, state: QueryGraphState) -> dict[str, Any]:
        request = state["request"]
        citations = state.get("citations", [])
        intent = state.get("intent")
        intent_type = intent.intent_type if intent else None
        version_hint = state.get("version_hint")
        answer = self.answer.compose(
            request.question, citations, intent_type=intent_type, version_hint=version_hint
        )
        return {"answer": answer}

    def _log_query(self, *, query_id: str, state: QueryGraphState) -> None:
        request = state.get("request")
        if request is None:
            return

        user = state.get("resolved_user")
        if user is None:
            user = request.user_profile or UserProfile(
                user_id=request.user_id or "anonymous",
                role="employee",
                department="unknown",
                projects=[],
            )
        intent = state.get("intent")
        citations = state.get("citations", [])

        row = {
            "query_id": query_id,
            "question": request.question,
            "retrieval_query": state.get("retrieval_query", request.question),
            "intent_type": intent.intent_type if intent else "general",
            "user_id": user.user_id,
            "role": user.role,
            "department": user.department,
            "projects": user.projects,
            "citation_doc_ids": [item.doc_id for item in citations if item.doc_id],
            "citation_count": len(citations),
            "version_hint": state.get("version_hint"),
        }
        try:
            self.query_log_repo.save_query_log(row)
        except Exception:
            # 日志失败不应影响主问答流程。
            return
