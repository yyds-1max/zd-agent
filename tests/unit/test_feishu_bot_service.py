from __future__ import annotations

import base64
import hashlib
import json

from app.repositories.conversation_session_repository import ConversationSessionRepository
from app.schemas.answer_strategy import AnswerStrategyResult
from app.schemas.intent import IntentResult
from app.schemas.knowledge import Citation
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.user import UserProfile
from app.services.conversation_session_service import ConversationSessionService
from app.services.feishu_bot_service import FeishuBotService


class _FakePipeline:
    def __init__(self) -> None:
        self.requests: list[QueryRequest] = []

    def run(self, request: QueryRequest) -> QueryResponse:
        self.requests.append(request)
        return QueryResponse(
            question=request.question,
            answer="根据当前权限范围，出差报销时限为 10 天。",
            user_profile=UserProfile(
                user_id=request.user_id,
                user_id_type=request.user_id_type,
                name="李然",
                department="市场部",
                title="运营专员",
                level="P2",
                role="employee",
            ),
            intent=IntentResult(name="policy_lookup", confidence=0.9),
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    chunk_index=0,
                    doc_id="doc-1",
                    title="字节跳动员工差旅与报销制度",
                    doc_type="policy",
                    version="V2.0",
                    permission_level="全员可见",
                    source_path="data/fixtures/02.txt",
                )
            ],
            answer_strategy=AnswerStrategyResult(
                mode="current_policy_mode",
                reason="命中制度文档",
            ),
            version_notice="当前推荐版本为 V2.0。",
        )


class _FakeMessageClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def send_text(self, *, receive_id: str, receive_id_type: str, text: str) -> None:
        self.messages.append(
            {
                "receive_id": receive_id,
                "receive_id_type": receive_id_type,
                "text": text,
            }
        )


def _encrypt_payload(plaintext: dict, encrypt_key: str) -> str:
    from Crypto.Cipher import AES

    raw = json.dumps(plaintext, ensure_ascii=False).encode("utf-8")
    padding = AES.block_size - (len(raw) % AES.block_size)
    padded = raw + bytes([padding]) * padding
    iv = b"0123456789abcdef"
    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = iv + cipher.encrypt(padded)
    return base64.b64encode(encrypted).decode("utf-8")


def test_feishu_bot_service_returns_challenge_on_url_verification() -> None:
    service = FeishuBotService(
        pipeline=_FakePipeline(),
        message_client=_FakeMessageClient(),
        verification_token="token-123",
    )

    response = service.handle_callback(
        {
            "type": "url_verification",
            "token": "token-123",
            "challenge": "challenge-value",
        }
    )

    assert response == {"challenge": "challenge-value"}


def test_feishu_bot_service_decrypts_challenge_payload() -> None:
    service = FeishuBotService(
        pipeline=_FakePipeline(),
        message_client=_FakeMessageClient(),
        verification_token="token-123",
        encrypt_key="encrypt-key-123",
    )
    inner_payload = {
        "token": "token-123",
        "challenge": "challenge-from-encrypted-payload",
    }

    response = service.handle_callback(
        {
            "encrypt": _encrypt_payload(inner_payload, "encrypt-key-123"),
        }
    )

    assert response == {"challenge": "challenge-from-encrypted-payload"}


def test_feishu_bot_service_processes_message_event_and_replies() -> None:
    pipeline = _FakePipeline()
    message_client = _FakeMessageClient()
    service = FeishuBotService(
        pipeline=pipeline,
        message_client=message_client,
        verification_token="token-123",
        default_top_k=5,
    )

    response = service.handle_callback(
        {
            "schema": "2.0",
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "token-123",
            },
            "event": {
                "sender": {
                    "sender_type": "user",
                    "sender_id": {
                        "open_id": "ou_test_user",
                    },
                },
                "message": {
                    "message_type": "text",
                    "chat_type": "p2p",
                    "chat_id": "oc_chat",
                    "content": "{\"text\":\"请问差旅报销最新标准是什么？\"}",
                },
            },
        }
    )

    assert response == {"code": 0}
    assert pipeline.requests[0].user_id == "ou_test_user"
    assert pipeline.requests[0].user_id_type == "open_id"
    assert pipeline.requests[0].question == "请问差旅报销最新标准是什么？"
    assert pipeline.requests[0].top_k == 5
    assert pipeline.requests[0].conversation_id == "feishu:oc_chat"
    assert message_client.messages[0]["receive_id"] == "ou_test_user"
    assert message_client.messages[0]["receive_id_type"] == "open_id"
    assert "10 天" in message_client.messages[0]["text"]
    assert "V2.0" in message_client.messages[0]["text"]


def test_feishu_bot_service_ignores_duplicate_message_event() -> None:
    pipeline = _FakePipeline()
    message_client = _FakeMessageClient()
    service = FeishuBotService(
        pipeline=pipeline,
        message_client=message_client,
        verification_token="token-123",
    )
    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "event-duplicate-1",
            "event_type": "im.message.receive_v1",
            "token": "token-123",
        },
        "event": {
            "sender": {
                "sender_type": "user",
                "sender_id": {
                    "open_id": "ou_test_user",
                },
            },
            "message": {
                "message_id": "om_duplicate_1",
                "message_type": "text",
                "chat_type": "p2p",
                "chat_id": "oc_chat",
                "content": "{\"text\":\"请问差旅报销最新标准是什么？\"}",
            },
        },
    }

    first_response = service.handle_callback(payload)
    duplicate_response = service.handle_callback(payload)

    assert first_response == {"code": 0}
    assert duplicate_response == {"code": 0}
    assert len(pipeline.requests) == 1
    assert len(message_client.messages) == 1


def test_feishu_bot_menu_starts_new_conversation(tmp_path) -> None:
    pipeline = _FakePipeline()
    message_client = _FakeMessageClient()
    session_service = ConversationSessionService(
        ConversationSessionRepository(tmp_path / "sessions.json")
    )
    service = FeishuBotService(
        pipeline=pipeline,
        message_client=message_client,
        verification_token="token-123",
        conversation_session_service=session_service,
        new_conversation_menu_key="start_new_conversation",
    )

    response = service.handle_callback(
        {
            "schema": "2.0",
            "header": {
                "event_id": "menu-event-1",
                "event_type": "application.bot.menu_v6",
                "token": "token-123",
            },
            "event": {
                "event_key": "start_new_conversation",
                "operator": {
                    "operator_id": {
                        "open_id": "ou_test_user",
                    }
                },
            },
        }
    )

    assert response == {"code": 0}
    assert pipeline.requests == []
    assert len(message_client.messages) == 1
    assert message_client.messages[0]["receive_id"] == "ou_test_user"
    assert "已开始一个新会话" in message_client.messages[0]["text"]
    active_id = session_service.active_conversation_id(
        channel_key="feishu:p2p:ou_test_user"
    )
    assert active_id in message_client.messages[0]["text"]


def test_feishu_bot_message_uses_active_conversation_from_menu(tmp_path) -> None:
    pipeline = _FakePipeline()
    message_client = _FakeMessageClient()
    session_service = ConversationSessionService(
        ConversationSessionRepository(tmp_path / "sessions.json")
    )
    active_id = session_service.start_new_conversation(
        channel_key="feishu:oc_chat",
    )
    service = FeishuBotService(
        pipeline=pipeline,
        message_client=message_client,
        verification_token="token-123",
        conversation_session_service=session_service,
    )

    response = service.handle_callback(
        {
            "schema": "2.0",
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "token-123",
            },
            "event": {
                "sender": {
                    "sender_type": "user",
                    "sender_id": {
                        "open_id": "ou_test_user",
                    },
                },
                "message": {
                    "message_type": "text",
                    "chat_type": "p2p",
                    "chat_id": "oc_chat",
                    "content": "{\"text\":\"出差报销最新标准是什么？\"}",
                },
            },
        }
    )

    assert response == {"code": 0}
    assert pipeline.requests[0].conversation_id == active_id


def test_feishu_bot_reply_deduplicates_document_sources() -> None:
    service = FeishuBotService(
        pipeline=_FakePipeline(),
        message_client=_FakeMessageClient(),
    )
    response = _FakePipeline().run(
        QueryRequest(
            user_id="ou_test_user",
            question="出差报销最新标准是什么？",
        )
    )
    response.citations.append(response.citations[0].model_copy(update={"chunk_id": "chunk-2"}))

    text = service._build_reply_text(response)

    assert text.count("字节跳动员工差旅与报销制度") == 1
    assert "（V2.0）（V2.0）" not in text
