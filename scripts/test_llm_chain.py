from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.vector_store import ChromaVectorStore
from app.repositories.knowledge_repository import KnowledgeRepository
from app.repositories.user_repository import UserRepository
from app.schemas.answer_strategy import AnswerStrategyResult
from app.schemas.intent import IntentResult
from app.schemas.knowledge import Citation, KnowledgeChunk, KnowledgeDocument, RetrievedChunk
from app.schemas.user import DirectoryUser, UserProfile
from app.services.answer_service import AnswerService
from app.services.answer_strategy_router_service import AnswerStrategyRouterService
from app.services.embedding_service import DashScopeEmbeddingService
from app.services.intent_parser_service import IntentParserService
from app.services.permission_service import PermissionService
from app.services.qwen_llm_service import QwenStructuredLLMService
from app.services.rerank_service import DashScopeRerankService
from app.services.retrieval_service import RetrievalService
from app.services.user_profile_llm_service import UserProfileLLMService
from app.services.version_diff_service import VersionDiffService
from app.services.version_service import VersionService
from app.schemas.version import VersionCheckResult, VersionDiffResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test qwen3-max LLM chain components.")
    parser.add_argument(
        "target",
        choices=["profile", "answer", "both"],
        default="both",
        nargs="?",
        help="Which LLM stage to test.",
    )
    parser.add_argument(
        "--user-id",
        default="u_employee_li",
        help="Demo user id from data/feishu_users.json",
    )
    parser.add_argument(
        "--question",
        default="出差报销最新标准是什么？",
        help="Question to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="How many retrieval candidates to prepare for answer testing.",
    )
    parser.add_argument(
        "--print-context",
        action="store_true",
        help="Print retrieved docs, citations, and backend summary before LLM output.",
    )
    return parser


def build_llm_service(*, enable: bool) -> QwenStructuredLLMService:
    settings = get_settings()
    return QwenStructuredLLMService(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_llm_base_url,
        model=settings.dashscope_llm_model,
        temperature=settings.dashscope_llm_temperature,
        timeout_seconds=settings.dashscope_llm_timeout_seconds,
        enabled=enable,
    )


def build_runtime() -> dict[str, object]:
    settings = get_settings()
    knowledge_repository = KnowledgeRepository(settings.fixtures_dir)
    user_repository = UserRepository(settings.feishu_directory_path)
    permission_service = PermissionService()
    version_service = VersionService(knowledge_repository, permission_service)
    version_diff_service = VersionDiffService(llm_service=build_llm_service(enable=True))
    answer_strategy_router_service = AnswerStrategyRouterService()
    intent_parser_service = IntentParserService(knowledge_repository.list_known_projects())

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
    if vector_store.available:
        vector_store.sync_chunks(knowledge_repository.list_chunks())

    rerank_service = DashScopeRerankService(
        api_key=settings.dashscope_api_key,
        rerank_url=settings.dashscope_rerank_url,
        model=settings.dashscope_rerank_model,
    )
    retrieval_service = RetrievalService(
        vector_store=vector_store,
        rerank_service=rerank_service,
        candidate_limit=settings.retrieval_candidate_limit,
    )
    return {
        "settings": settings,
        "knowledge_repository": knowledge_repository,
        "user_repository": user_repository,
        "permission_service": permission_service,
        "version_service": version_service,
        "version_diff_service": version_diff_service,
        "answer_strategy_router_service": answer_strategy_router_service,
        "intent_parser_service": intent_parser_service,
        "retrieval_service": retrieval_service,
        "vector_store": vector_store,
    }


def ensure_llm_ready(llm_service: QwenStructuredLLMService) -> None:
    if llm_service.is_available():
        return
    raise RuntimeError(
        "qwen3-max LLM 未启用。请确认 .env 中已设置 DASHSCOPE_API_KEY，"
        "并且当前脚本可以访问 DASHSCOPE_LLM_BASE_URL。"
    )


def build_profile_and_intent(
    *,
    user_repository: UserRepository,
    intent_parser_service: IntentParserService,
    user_id: str,
    question: str,
) -> tuple[DirectoryUser, UserProfile, IntentResult]:
    base_user = user_repository.get_by_user_id(user_id)
    intent = intent_parser_service.parse(question)
    profile = UserProfile(
        **base_user.model_dump(),
        project_mentions=intent.project_names,
        active_projects=sorted(
            set(base_user.projects) | set(base_user.managed_projects) | set(intent.project_names)
        ),
        intent_hint=intent.name,
    )
    return base_user, profile, intent


def print_runtime_summary(runtime: dict[str, object]) -> None:
    settings = runtime["settings"]
    vector_store = runtime["vector_store"]
    print("=== RUNTIME SUMMARY ===")
    print(
        json.dumps(
            {
                "llm_model": settings.dashscope_llm_model,
                "embedding_model": settings.dashscope_embedding_model,
                "rerank_model": settings.dashscope_rerank_model,
                "vector_backend": vector_store.describe(),
                "vector_backend_status": vector_store.status_reason,
                "vector_backend_error": vector_store.last_error_detail or None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def print_answer_context(
    *,
    question: str,
    profile: UserProfile,
    intent: IntentResult,
    retrieval_service: RetrievalService,
    retrieved_chunks: list[RetrievedChunk],
    version_checks: list[VersionCheckResult],
    version_diffs: list[VersionDiffResult],
    answer_strategy: AnswerStrategyResult,
    citations: list[Citation],
    lead_chunk: KnowledgeChunk,
    lead_document: KnowledgeDocument,
    version_notice: str | None,
    key_points: list[str],
) -> None:
    citation_payload = [
        {
            "doc_id": item.doc_id,
            "title": item.title,
            "doc_type": item.doc_type,
            "version": item.version,
            "is_latest": item.is_latest,
            "snippet": item.snippet,
        }
        for item in citations[:4]
    ]
    print("=== ANSWER TEST CONTEXT ===")
    print(
        json.dumps(
            {
                "profile": profile.model_dump(),
                "intent": intent.model_dump(),
                "retrieval_backend": retrieval_service.describe_backend(),
                "lead_document": {
                    "doc_id": lead_document.doc_id,
                    "title": lead_document.title,
                    "version": lead_document.version,
                    "is_latest": lead_document.is_latest,
                },
                "lead_chunk": {
                    "chunk_id": lead_chunk.chunk_id,
                    "section_title": lead_chunk.section_title,
                    "subsection_title": lead_chunk.subsection_title,
                    "text": lead_chunk.text,
                },
                "version_notice": version_notice,
                "version_checks": [item.model_dump(mode="json") for item in version_checks],
                "version_diffs": [item.model_dump(mode="json") for item in version_diffs],
                "answer_strategy": answer_strategy.model_dump(mode="json"),
                "key_points": key_points,
                "retrieved_chunks": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "doc_id": item.chunk.doc_id,
                        "title": item.chunk.doc_title,
                        "section_title": item.chunk.section_title,
                        "subsection_title": item.chunk.subsection_title,
                        "bm25_score": round(item.bm25_score, 4),
                        "vector_score": round(item.vector_score, 4),
                        "rrf_score": round(item.rrf_score, 4),
                        "rerank_score": round(item.rerank_score, 4),
                        "final_score": round(item.final_score, 4),
                        "snippet": item.snippet,
                    }
                    for item in retrieved_chunks
                ],
                "citations": [item.model_dump(mode="json") for item in citations],
                "answer_llm_context_summary": {
                    "question": question,
                    "profile": profile.model_dump(),
                    "intent": intent.model_dump(),
                    "lead_document": {
                        "title": lead_document.title,
                        "doc_type": lead_document.doc_type,
                        "version": lead_document.version,
                    },
                    "lead_chunk": {
                        "chunk_id": lead_chunk.chunk_id,
                        "section_title": lead_chunk.section_title,
                        "subsection_title": lead_chunk.subsection_title,
                        "text": lead_chunk.text,
                    },
                    "key_points": key_points,
                    "version_notice": version_notice or "无",
                    "version_checks": [item.model_dump(mode="json") for item in version_checks],
                    "version_diffs": [item.model_dump(mode="json") for item in version_diffs],
                    "answer_strategy": answer_strategy.model_dump(mode="json"),
                    "citation_payload": citation_payload,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def print_retrieval_summary(
    *,
    retrieval_service: RetrievalService,
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    rerank_hits = [
        {
            "chunk_id": item.chunk.chunk_id,
            "doc_id": item.chunk.doc_id,
            "title": item.chunk.doc_title,
            "section_title": item.chunk.section_title,
            "subsection_title": item.chunk.subsection_title,
            "rerank_score": round(item.rerank_score, 4),
            "final_score": round(item.final_score, 4),
        }
        for item in retrieved_chunks
        if item.rerank_score > 0
    ]
    print("=== RETRIEVAL SUMMARY ===")
    print(
        json.dumps(
            {
                "retrieval_backend": retrieval_service.describe_backend(),
                "retrieved_count": len(retrieved_chunks),
                "rerank_hit_count": len(rerank_hits),
                "rerank_hits": rerank_hits,
                "ranked_candidates": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "doc_id": item.chunk.doc_id,
                        "title": item.chunk.doc_title,
                        "section_title": item.chunk.section_title,
                        "subsection_title": item.chunk.subsection_title,
                        "bm25_score": round(item.bm25_score, 4),
                        "vector_score": round(item.vector_score, 4),
                        "rrf_score": round(item.rrf_score, 4),
                        "rerank_score": round(item.rerank_score, 4),
                        "final_score": round(item.final_score, 4),
                    }
                    for item in retrieved_chunks
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_profile_test(runtime: dict[str, object], user_id: str, question: str) -> None:
    knowledge_repository = runtime["knowledge_repository"]
    user_repository = runtime["user_repository"]
    llm_service = build_llm_service(enable=True)
    ensure_llm_ready(llm_service)
    profile_llm_service = UserProfileLLMService(
        llm_service=llm_service,
        known_projects=knowledge_repository.list_known_projects(),
    )
    base_user = user_repository.get_by_user_id(user_id)
    result = profile_llm_service.understand(question, base_user)
    print("=== USER PROFILE LLM OUTPUT ===")
    print(result.model_dump_json(indent=2))


def run_answer_test(
    runtime: dict[str, object],
    user_id: str,
    question: str,
    top_k: int,
    print_context: bool,
) -> None:
    user_repository = runtime["user_repository"]
    permission_service = runtime["permission_service"]
    knowledge_repository = runtime["knowledge_repository"]
    version_service = runtime["version_service"]
    version_diff_service = runtime["version_diff_service"]
    answer_strategy_router_service = runtime["answer_strategy_router_service"]
    intent_parser_service = runtime["intent_parser_service"]
    retrieval_service = runtime["retrieval_service"]

    llm_service = build_llm_service(enable=True)
    ensure_llm_ready(llm_service)
    answer_service = AnswerService(version_service=version_service, llm_service=llm_service)

    _, profile, intent = build_profile_and_intent(
        user_repository=user_repository,
        intent_parser_service=intent_parser_service,
        user_id=user_id,
        question=question,
    )
    accessible_documents = permission_service.filter_accessible(
        profile, knowledge_repository.list_documents()
    )
    accessible_chunks = knowledge_repository.list_chunks_for_documents(accessible_documents)
    retrieved_chunks = retrieval_service.search(
        question=question,
        intent=intent,
        documents=accessible_documents,
        top_k=top_k,
        chunks=accessible_chunks,
    )
    if not retrieved_chunks:
        raise RuntimeError("当前权限范围内没有命中任何候选文档，无法构造主Agent答案测试上下文。")

    print_retrieval_summary(
        retrieval_service=retrieval_service,
        retrieved_chunks=retrieved_chunks,
    )

    version_checks = version_service.check_versions(profile, retrieved_chunks)
    version_diffs = []
    for check in version_checks:
        if not check.has_newer_version or not check.source_chunk_id:
            continue
        source_chunk = next(
            (item.chunk for item in retrieved_chunks if item.chunk.chunk_id == check.source_chunk_id),
            None,
        )
        if source_chunk is None:
            continue
        version_diffs.append(
            version_diff_service.compare(
                old_chunk=source_chunk,
                new_chunk=version_service.latest_aligned_chunk_for_check(check),
                question=question,
            )
        )
    answer_strategy = answer_strategy_router_service.route(
        question=question,
        intent=intent,
        retrieved_chunks=retrieved_chunks,
        version_checks=version_checks,
        version_diffs=version_diffs,
    )
    lead_chunk = answer_service._select_lead_chunk(retrieved_chunks)
    lead_document = version_service.document_for_chunk(lead_chunk)
    strategy_chunks = answer_service._select_strategy_chunks(
        retrieved_chunks=retrieved_chunks,
        answer_strategy=answer_strategy,
    )
    lead_chunk = answer_service._select_lead_chunk(strategy_chunks)
    lead_document = version_service.document_for_chunk(lead_chunk)
    key_points = answer_service._extract_key_points(
        question,
        intent,
        strategy_chunks,
        answer_strategy,
        version_diffs,
    )
    version_notice = answer_service._build_version_notice(
        lead_document,
        version_checks,
        answer_strategy,
    )
    citations = answer_service._build_citations(strategy_chunks)

    if print_context:
        print_answer_context(
            question=question,
            profile=profile,
            intent=intent,
            retrieval_service=retrieval_service,
            retrieved_chunks=strategy_chunks,
            version_checks=version_checks,
            version_diffs=version_diffs,
            answer_strategy=answer_strategy,
            citations=citations,
            lead_chunk=lead_chunk,
            lead_document=lead_document,
            version_notice=version_notice,
            key_points=key_points,
        )

    llm_output = answer_service._compose_with_llm(
        question=question,
        profile=profile,
        intent=intent,
        lead_chunk=lead_chunk,
        lead_document=lead_document,
        key_points=key_points,
        version_notice=version_notice,
        citations=citations,
        version_diffs=version_diffs,
        answer_strategy=answer_strategy,
    )
    print("=== ANSWER LLM STRUCTURED OUTPUT ===")
    print(llm_output.model_dump_json(indent=2))


def main() -> None:
    args = build_parser().parse_args()
    runtime = build_runtime()
    print_runtime_summary(runtime)

    if args.target in {"profile", "both"}:
        print()
        run_profile_test(runtime, args.user_id, args.question)

    if args.target in {"answer", "both"}:
        print()
        run_answer_test(runtime, args.user_id, args.question, args.top_k, args.print_context)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"LLM smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
