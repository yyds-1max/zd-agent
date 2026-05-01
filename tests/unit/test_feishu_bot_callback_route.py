from __future__ import annotations

import asyncio
import base64
import hashlib
import json

from starlette.requests import Request

from app.api.routes.feishu_bot import create_feishu_bot_router
from app.services.feishu_bot_service import FeishuBotService


class _NoopPipeline:
    def run(self, request):  # pragma: no cover
        raise NotImplementedError


class _NoopMessageClient:
    def send_text(self, *, receive_id: str, receive_id_type: str, text: str) -> None:  # pragma: no cover
        raise NotImplementedError


def _encrypt_payload(plaintext: dict, encrypt_key: str) -> str:
    from Crypto.Cipher import AES

    raw = json.dumps(plaintext, ensure_ascii=False).encode("utf-8")
    padding = AES.block_size - (len(raw) % AES.block_size)
    padded = raw + bytes([padding]) * padding
    iv = b"fedcba9876543210"
    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = iv + cipher.encrypt(padded)
    return base64.b64encode(encrypted).decode("utf-8")


def _build_request(body: bytes, headers: dict[str, str] | None = None) -> Request:
    header_items = [(key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in (headers or {}).items()]

    async def receive() -> dict:
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/feishu/events",
            "headers": header_items,
        },
        receive=receive,
    )


def test_challenge_callback_does_not_require_type_field() -> None:
    service = FeishuBotService(
        pipeline=_NoopPipeline(),
        message_client=_NoopMessageClient(),
        verification_token="token-123",
    )
    router = create_feishu_bot_router(service)
    request = _build_request(
        json.dumps(
            {
                "token": "token-123",
                "challenge": "challenge-without-type",
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    response = asyncio.run(router.routes[0].endpoint(request))

    assert response == {"challenge": "challenge-without-type"}


def test_encrypted_challenge_callback_returns_decrypted_challenge() -> None:
    encrypt_key = "encrypt-key-123"
    plaintext = {
        "token": "token-123",
        "challenge": "challenge-encrypted",
    }
    body = json.dumps(
        {"encrypt": _encrypt_payload(plaintext, encrypt_key)},
        ensure_ascii=False,
    ).encode("utf-8")
    timestamp = "1710000000"
    nonce = "nonce-123"
    signature = hashlib.sha256((timestamp + nonce + encrypt_key).encode("utf-8") + body).hexdigest()
    service = FeishuBotService(
        pipeline=_NoopPipeline(),
        message_client=_NoopMessageClient(),
        verification_token="token-123",
        encrypt_key=encrypt_key,
    )
    router = create_feishu_bot_router(service)
    request = _build_request(
        body,
        headers={
            "Content-Type": "application/json",
            "X-Lark-Request-Timestamp": timestamp,
            "X-Lark-Request-Nonce": nonce,
            "X-Lark-Signature": signature,
        },
    )
    response = asyncio.run(router.routes[0].endpoint(request))

    assert response == {"challenge": "challenge-encrypted"}
