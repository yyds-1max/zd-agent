from __future__ import annotations

from app.core.config import get_settings
from app.core.vector_store import ChromaVectorStore
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.repositories.user_repository import UserRepository
from app.schemas.query import QueryRequest, QueryResponse
from app.services.answer_service import AnswerService
from app.services.answer_strategy_router_service import AnswerStrategyRouterService
from app.services.agent_controller_service import AgentControllerService
from app.services.conversation_intent_service import ConversationIntentService
from app.services.conversation_memory_service import ConversationMemoryService
from app.services.conversation_rewrite_service import ConversationRewriteService
from app.services.embedding_service import DashScopeEmbeddingService
from app.services.feishu_contact_service import FeishuContactService, MockFeishuContactService
from app.services.intent_parser_service import IntentParserService
from app.services.knowledge_tools import (
    KnowledgeRetrievalTool,
    LatestVersionTool,
    UserProfileTool,
    VersionDiffTool,
)
from app.services.main_agent_service import KnowledgeDispatchMainAgent
from app.services.permission_service import PermissionService
from app.services.qwen_llm_service import QwenStructuredLLMService
from app.services.retrieval_service import RetrievalService
from app.services.rerank_service import DashScopeRerankService
from app.services.task_router_service import TaskRouterService
from app.services.user_profile_llm_service import UserProfileLLMService
from app.services.version_diff_service import VersionDiffService
from app.services.version_service import VersionService


class QueryPipeline:
    def __init__(
        self,
        main_agent: KnowledgeDispatchMainAgent,
        conversation_memory_service: ConversationMemoryService | None = None,
    ):
        self.main_agent = main_agent
        self.conversation_memory_service = conversation_memory_service

    def run(self, request: QueryRequest) -> QueryResponse:
        original_question = request.question
        conversation_id = self._conversation_id(request)
        history = self._load_history(request, conversation_id)
        contextual_question = self._contextual_question(
            question=original_question,
            history=history,
            use_history=request.use_history,
        )
        effective_request = request.model_copy(
            update={
                "question": contextual_question,
                "routing_question": original_question,
                "conversation_id": conversation_id,
            }
        )

        response = self.main_agent.run(effective_request)
        response.question = original_question
        response.conversation_id = conversation_id
        response.contextual_question = (
            contextual_question if contextual_question != original_question else None
        )
        if contextual_question != original_question:
            response.tool_trace.insert(0, "QueryPipeline：已结合会话历史改写当前追问。")
            response.notes.append("已结合会话历史理解本轮追问。")
        self._record_history(conversation_id, original_question, response)
        return response

    def _conversation_id(self, request: QueryRequest) -> str | None:
        if self.conversation_memory_service is None:
            return request.conversation_id
        return self.conversation_memory_service.conversation_id_for(
            user_id=request.user_id,
            requested_conversation_id=request.conversation_id,
        )

    def _load_history(
        self,
        request: QueryRequest,
        conversation_id: str | None,
    ):
        if (
            self.conversation_memory_service is None
            or not request.use_history
            or not conversation_id
        ):
            return []
        return self.conversation_memory_service.load_recent(
            conversation_id=conversation_id,
            limit=request.history_limit,
        )

    def _contextual_question(self, *, question: str, history, use_history: bool) -> str:
        if self.conversation_memory_service is None or not use_history:
            return question
        return self.conversation_memory_service.contextualize_question(
            question=question,
            history=history,
        )

    def _record_history(
        self,
        conversation_id: str | None,
        original_question: str,
        response: QueryResponse,
    ) -> None:
        if self.conversation_memory_service is None or not conversation_id:
            return
        self.conversation_memory_service.record_exchange(
            conversation_id=conversation_id,
            user_question=original_question,
            response=response,
        )


def build_query_pipeline() -> QueryPipeline:
    settings = get_settings()
    conversation_repository = ConversationRepository(settings.conversation_store_path)
    knowledge_repository = KnowledgeRepository(settings.fixtures_dir)
    user_repository = UserRepository(settings.feishu_directory_path)
    embedding_service = DashScopeEmbeddingService(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_embedding_base_url,
        model=settings.dashscope_embedding_model,
        dimensions=settings.dashscope_embedding_dimensions,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_dir,
        collection_name=settings.chroma_collection_name,
        embedding_service=embedding_service,
    )
    vector_store.sync_chunks(knowledge_repository.list_chunks())
    rerank_service = DashScopeRerankService(
        api_key=settings.dashscope_api_key,
        rerank_url=settings.dashscope_rerank_url,
        model=settings.dashscope_rerank_model,
    )
    user_profile_llm = QwenStructuredLLMService(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_llm_base_url,
        model=settings.dashscope_llm_model,
        temperature=settings.dashscope_llm_temperature,
        timeout_seconds=settings.dashscope_llm_timeout_seconds,
        enabled=settings.enable_user_profile_llm,
    )
    conversation_router_llm = QwenStructuredLLMService(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_llm_base_url,
        model=settings.dashscope_llm_model,
        temperature=0.0,
        timeout_seconds=settings.dashscope_llm_timeout_seconds,
        enabled=settings.enable_conversation_router_llm,
    )
    answer_llm = QwenStructuredLLMService(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_llm_base_url,
        model=settings.dashscope_llm_model,
        temperature=settings.dashscope_llm_temperature,
        timeout_seconds=settings.dashscope_llm_timeout_seconds,
        enabled=settings.enable_answer_llm,
    )
    version_diff_llm = QwenStructuredLLMService(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_llm_base_url,
        model=settings.dashscope_llm_model,
        temperature=settings.dashscope_llm_temperature,
        timeout_seconds=settings.dashscope_llm_timeout_seconds,
        enabled=settings.enable_version_diff_llm,
    )
    conversation_memory_service = ConversationMemoryService(
        conversation_repository,
        default_history_limit=settings.conversation_history_limit,
        rewrite_service=ConversationRewriteService(
            llm_service=conversation_router_llm,
        ),
    )

    if (
        settings.feishu_contact_api_enabled
        and settings.feishu_app_id
        and settings.feishu_app_secret
    ):
        feishu_contact_service = FeishuContactService(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            log_level=settings.feishu_log_level,
            default_user_id_type=settings.feishu_default_user_id_type,
            user_repository=user_repository if settings.feishu_use_directory_fallback else None,
        )
    else:
        feishu_contact_service = MockFeishuContactService(user_repository)
    intent_parser_service = IntentParserService(knowledge_repository.list_known_projects())
    user_profile_llm_service = UserProfileLLMService(
        llm_service=user_profile_llm,
        known_projects=knowledge_repository.list_known_projects(),
    )
    permission_service = PermissionService()
    retrieval_service = RetrievalService(
        vector_store=vector_store,
        rerank_service=rerank_service,
        candidate_limit=settings.retrieval_candidate_limit,
    )
    version_service = VersionService(knowledge_repository, permission_service)
    version_diff_service = VersionDiffService(llm_service=version_diff_llm)
    agent_controller_service = AgentControllerService(
        llm_service=answer_llm,
        max_retrieval_iterations=2,
    )
    answer_strategy_router_service = AnswerStrategyRouterService()
    answer_service = AnswerService(version_service, llm_service=answer_llm)

    user_profile_tool = UserProfileTool(
        feishu_contact_service=feishu_contact_service,
        intent_parser_service=intent_parser_service,
        user_profile_llm_service=user_profile_llm_service,
    )
    retrieval_tool = KnowledgeRetrievalTool(
        knowledge_repository=knowledge_repository,
        permission_service=permission_service,
        retrieval_service=retrieval_service,
    )
    latest_version_tool = LatestVersionTool(version_service=version_service)
    version_diff_tool = VersionDiffTool(
        version_service=version_service,
        version_diff_service=version_diff_service,
    )

    main_agent = KnowledgeDispatchMainAgent(
        user_profile_tool=user_profile_tool,
        task_router_service=TaskRouterService(
            llm_service=conversation_router_llm,
            conversation_intent_service=ConversationIntentService(
                llm_service=conversation_router_llm,
            ),
        ),
        retrieval_tool=retrieval_tool,
        agent_controller_service=agent_controller_service,
        latest_version_tool=latest_version_tool,
        version_diff_tool=version_diff_tool,
        answer_strategy_router_service=answer_strategy_router_service,
        answer_service=answer_service,
    )
    return QueryPipeline(
        main_agent,
        conversation_memory_service=conversation_memory_service,
    )
