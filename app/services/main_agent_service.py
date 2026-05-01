from __future__ import annotations

from app.schemas.answer_strategy import AnswerStrategyResult
from app.schemas.intent import IntentResult
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.task_route import TaskRouteResult
from app.schemas.workflow import QueryWorkflowState
from app.services.answer_service import AnswerService
from app.services.langgraph_support import (
    END,
    START,
    LANGGRAPH_AVAILABLE,
    StateGraph,
)
from app.services.task_router_service import TaskRouterService
from app.schemas.agent_loop import AgentObservation
from app.services.agent_controller_service import AgentControllerService
from app.services.answer_strategy_router_service import AnswerStrategyRouterService
from app.services.knowledge_tools import (
    KnowledgeRetrievalTool,
    LatestVersionTool,
    UserProfileTool,
    VersionDiffTool,
)


class KnowledgeDispatchMainAgent:
    def __init__(
        self,
        user_profile_tool: UserProfileTool,
        task_router_service: TaskRouterService,
        retrieval_tool: KnowledgeRetrievalTool,
        agent_controller_service: AgentControllerService,
        latest_version_tool: LatestVersionTool,
        version_diff_tool: VersionDiffTool,
        answer_strategy_router_service: AnswerStrategyRouterService,
        answer_service: AnswerService,
    ):
        self.user_profile_tool = user_profile_tool
        self.task_router_service = task_router_service
        self.retrieval_tool = retrieval_tool
        self.agent_controller_service = agent_controller_service
        self.latest_version_tool = latest_version_tool
        self.version_diff_tool = version_diff_tool
        self.answer_strategy_router_service = answer_strategy_router_service
        self.answer_service = answer_service
        self._workflow = self._build_workflow() if LANGGRAPH_AVAILABLE else None

    def run(self, request: QueryRequest) -> QueryResponse:
        if self._workflow is not None:
            result = self._workflow.invoke(
                {
                    "user_id": request.user_id,
                    "user_id_type": request.user_id_type,
                    "question": request.question,
                    "routing_question": request.routing_question,
                    "conversation_id": request.conversation_id,
                    "top_k": request.top_k,
                    "observations": [],
                    "retrieval_iterations": 0,
                    "tool_trace": [],
                }
            )
            return result["response"]
        return self._run_sequential(request)

    def _build_workflow(self):
        builder = StateGraph(QueryWorkflowState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("answer", self._answer_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "tools")
        builder.add_edge("tools", "answer")
        builder.add_edge("answer", END)
        return builder.compile()

    def _run_sequential(self, request: QueryRequest) -> QueryResponse:
        state: QueryWorkflowState = {
            "user_id": request.user_id,
            "user_id_type": request.user_id_type,
            "question": request.question,
            "routing_question": request.routing_question,
            "conversation_id": request.conversation_id,
            "top_k": request.top_k,
            "observations": [],
            "retrieval_iterations": 0,
            "tool_trace": [],
        }
        state.update(self._plan_node(state))
        state.update(self._tools_node(state))
        state.update(self._answer_node(state))
        return state["response"]

    def _plan_node(self, state: QueryWorkflowState) -> QueryWorkflowState:
        profile, intent, profile_trace = self.user_profile_tool.run(
            state["user_id"],
            state["question"],
            state.get("user_id_type", "open_id"),
        )
        task_route, route_trace = self._route_task(
            state.get("routing_question") or state["question"]
        )
        if not task_route.should_retrieve:
            intent = IntentResult(
                name=task_route.intent_name,
                confidence=task_route.confidence,
                reasoning=task_route.reasoning,
            )
        return {
            "user_profile": profile,
            "intent": intent,
            "task_route": task_route,
            "tool_trace": [*state.get("tool_trace", []), profile_trace, route_trace],
        }

    def _tools_node(self, state: QueryWorkflowState) -> QueryWorkflowState:
        task_route = state["task_route"]
        if not task_route.should_retrieve:
            return {
                "retrieved_chunks": [],
                "version_checks": [],
                "version_diffs": [],
            }

        profile = state["user_profile"]
        intent = state["intent"]
        question = state["question"]
        top_k = state["top_k"]
        retrieved_chunks, retrieval_trace = self.retrieval_tool.run(
            question=question,
            profile=profile,
            intent=intent,
            top_k=top_k,
        )
        observations: list[AgentObservation] = []
        retrieval_iterations = 0
        loop_traces: list[str] = []
        while True:
            decision = self.agent_controller_service.decide(
                question=question,
                profile=profile,
                intent=intent,
                retrieved_chunks=retrieved_chunks,
                observations=observations,
                retrieval_iterations=retrieval_iterations,
            )
            controller_trace = self._controller_trace(decision)
            loop_traces.append(controller_trace)
            if decision.action != "retrieve" or not decision.action_query:
                break

            before_count = len(retrieved_chunks)
            retrieved_chunks, supplemental_trace = self.retrieval_tool.run_supplemental(
                queries=[decision.action_query],
                existing_chunks=retrieved_chunks,
                profile=profile,
                intent=intent,
                top_k=top_k,
            )
            retrieval_iterations += 1
            added_chunks = max(0, len(retrieved_chunks) - before_count)
            observations.append(
                AgentObservation(
                    iteration=retrieval_iterations,
                    action="retrieve",
                    query=decision.action_query,
                    added_chunks=added_chunks,
                    total_chunks=len(retrieved_chunks),
                    summary=f"补充检索新增 {added_chunks} 条候选。",
                )
            )
            loop_traces.append(supplemental_trace)

        version_checks, version_trace = self.latest_version_tool.run(
            profile=profile,
            retrieved_chunks=retrieved_chunks,
        )
        if self._should_run_version_diff(
            question=question,
            intent=intent,
            version_checks=version_checks,
        ):
            version_diffs, version_diff_trace = self.version_diff_tool.run(
                question=question,
                profile=profile,
                retrieved_chunks=retrieved_chunks,
                version_checks=version_checks,
            )
        else:
            version_diffs = []
            version_diff_trace = "版本差异工具：普通问答不需要新旧版本对比，已跳过。"
        answer_strategy = self.answer_strategy_router_service.route(
            question=question,
            intent=intent,
            retrieved_chunks=retrieved_chunks,
            version_checks=version_checks,
            version_diffs=version_diffs,
        )
        strategy_trace = self._strategy_trace(answer_strategy)
        return {
            "retrieved_chunks": retrieved_chunks,
            "observations": observations,
            "retrieval_iterations": retrieval_iterations,
            "version_checks": version_checks,
            "version_diffs": version_diffs,
            "answer_strategy": answer_strategy,
            "tool_trace": [
                *state.get("tool_trace", []),
                retrieval_trace,
                *loop_traces,
                version_trace,
                version_diff_trace,
                strategy_trace,
            ],
        }

    def _answer_node(self, state: QueryWorkflowState) -> QueryWorkflowState:
        task_route = state["task_route"]
        if not task_route.should_retrieve:
            response = self._compose_direct_response(
                request=self._request_from_state(state),
                profile=state["user_profile"],
                task_route=task_route,
                tool_trace=state.get("tool_trace", []),
            )
            return {"response": response}

        response = self.answer_service.compose(
            question=state["question"],
            profile=state["user_profile"],
            intent=state["intent"],
            retrieved_chunks=state.get("retrieved_chunks", []),
            version_checks=state.get("version_checks", []),
            version_diffs=state.get("version_diffs", []),
            answer_strategy=state["answer_strategy"],
            tool_trace=state.get("tool_trace", []),
        )
        return {"response": response}

    def _should_run_version_diff(
        self,
        *,
        question: str,
        intent,
        version_checks,
    ) -> bool:
        return self.answer_strategy_router_service.should_run_version_diff(
            question=question,
            intent=intent,
            version_checks=version_checks,
        )

    def _request_from_state(self, state: QueryWorkflowState) -> QueryRequest:
        return QueryRequest(
            user_id=state["user_id"],
            user_id_type=state.get("user_id_type", "open_id"),
            question=state["question"],
            routing_question=state.get("routing_question"),
            conversation_id=state.get("conversation_id"),
            top_k=state["top_k"],
        )

    def _route_task(self, question: str) -> tuple[TaskRouteResult, str]:
        task_route = self.task_router_service.route(question)
        trace = (
            f"主Agent决策中心：route={task_route.route_name}，"
            f"intent={task_route.intent_name}，source={task_route.source}，"
            f"confidence={task_route.confidence:.2f}。"
        )
        return task_route, trace

    def _compose_direct_response(
        self,
        *,
        request: QueryRequest,
        profile,
        task_route: TaskRouteResult,
        tool_trace: list[str],
    ) -> QueryResponse:
        answer = task_route.direct_answer or "你可以再补充一下想查的制度、项目或文档范围。"
        mode = f"{task_route.route_name}_mode"
        intent = IntentResult(
            name=task_route.intent_name,
            confidence=task_route.confidence,
            reasoning=task_route.reasoning,
        )
        return QueryResponse(
            question=request.question,
            answer=answer,
            conversation_id=request.conversation_id,
            user_profile=profile,
            intent=intent,
            citations=[],
            version_checks=[],
            version_diffs=[],
            answer_strategy=AnswerStrategyResult(
                mode=mode,
                reason=task_route.reasoning or "该任务不需要进入知识库检索。",
                include_version_notice=False,
            ),
            version_notice=None,
            notes=["主Agent判断本轮不需要进入知识库检索。"],
            tool_trace=tool_trace,
        )

    def _controller_trace(self, decision) -> str:
        trace = (
            f"Agent控制器：action={decision.action}，"
            f"source={decision.source}，confidence={decision.confidence:.2f}，"
            f"summary={decision.thought_summary or decision.reason}"
        )
        if decision.action_query:
            trace += f"，query={decision.action_query}"
        return trace + "。"

    def _strategy_trace(self, strategy: AnswerStrategyResult) -> str:
        return f"回答路由工具：已选择 `{strategy.mode}`，原因={strategy.reason}"
