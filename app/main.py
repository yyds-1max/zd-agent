from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.pipelines.query_pipeline import build_query_pipeline
from app.repositories.conversation_session_repository import ConversationSessionRepository
from app.schemas.query import QueryRequest
from app.services.conversation_session_service import ConversationSessionService
from app.services.feishu_bot_service import FeishuBotMessageClient, FeishuBotService

pipeline = build_query_pipeline()
settings = get_settings()


def create_app():
    from fastapi import FastAPI

    from app.api.routes.feishu_bot import create_feishu_bot_router
    from app.api.routes.query import create_query_router

    app = FastAPI(title="知达Agent", version="0.1.0")
    app.include_router(create_query_router(pipeline))
    if settings.feishu_bot_enabled and settings.feishu_app_id and settings.feishu_app_secret:
        bot_service = FeishuBotService(
            pipeline=pipeline,
            message_client=FeishuBotMessageClient(
                app_id=settings.feishu_app_id,
                app_secret=settings.feishu_app_secret,
                base_url=settings.feishu_api_base_url,
            ),
            verification_token=settings.feishu_verification_token,
            encrypt_key=settings.feishu_encrypt_key,
            default_top_k=settings.default_top_k,
            conversation_session_service=ConversationSessionService(
                ConversationSessionRepository(settings.conversation_session_store_path)
            ),
            new_conversation_menu_key=settings.feishu_new_conversation_menu_key,
        )
        app.include_router(
            create_feishu_bot_router(
                bot_service,
                path=settings.feishu_bot_event_path,
            )
        )
    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 知达Agent main flow.")
    parser.add_argument(
        "--user-id",
        required=True,
        help="User identifier from Feishu or data/feishu_users.json",
    )
    parser.add_argument(
        "--user-id-type",
        default="open_id",
        help="Identity type such as open_id, user_id or union_id",
    )
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=4, help="Number of citations to return")
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Conversation id for multi-turn memory. Defaults to the user id.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable loading and writing conversation history for this request.",
    )
    args = parser.parse_args()

    response = pipeline.run(
        QueryRequest(
            user_id=args.user_id,
            user_id_type=args.user_id_type,
            question=args.question,
            top_k=args.top_k,
            conversation_id=args.conversation_id,
            use_history=not args.no_history,
        )
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
