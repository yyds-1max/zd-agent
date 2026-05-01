# 飞书回调处理


from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from typing import Any, Protocol

import requests

from app.schemas.query import QueryRequest, QueryResponse
from app.services.conversation_session_service import ConversationSessionService


class QueryPipelineLike(Protocol):
    def run(self, request: QueryRequest) -> QueryResponse:
        ...


class MessageClientLike(Protocol):
    def send_text(self, *, receive_id: str, receive_id_type: str, text: str) -> None:
        ...


class FeishuBotVerificationError(RuntimeError):
    pass


class FeishuBotDecryptError(RuntimeError):
    pass


class FeishuBotMessageClient:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        base_url: str = "https://open.feishu.cn",
        timeout_seconds: int = 15,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._access_token: str | None = None
        self._access_token_expire_at = 0.0

    def send_text(self, *, receive_id: str, receive_id_type: str, text: str) -> None:
        access_token = self._tenant_access_token()
        response = requests.post(
            f"{self.base_url}/open-apis/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json={
                "receive_id": receive_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code", 0) != 0:
            raise RuntimeError(
                f"Feishu send message failed, code={payload.get('code')}, msg={payload.get('msg')}"
            )

    def _tenant_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._access_token_expire_at:
            return self._access_token

        response = requests.post(
            f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal",
            headers={"Content-Type": "application/json; charset=utf-8"},
            json={
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        access_token = payload.get("tenant_access_token")
        if not access_token:
            raise RuntimeError(
                f"Feishu tenant_access_token missing, code={payload.get('code')}, msg={payload.get('msg')}"
            )

        expire = int(payload.get("expire", 7200))
        self._access_token = access_token
        self._access_token_expire_at = now + max(expire - 60, 60)
        return access_token


class FeishuBotService:
    def __init__(
        self,
        *,
        pipeline: QueryPipelineLike,
        message_client: MessageClientLike,
        verification_token: str | None = None,
        encrypt_key: str | None = None,
        default_top_k: int = 4,
        dedup_ttl_seconds: int = 600,
        conversation_session_service: ConversationSessionService | None = None,
        new_conversation_menu_key: str = "start_new_conversation",
    ):
        self.pipeline = pipeline
        self.message_client = message_client
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.default_top_k = default_top_k
        self.dedup_ttl_seconds = dedup_ttl_seconds
        self.conversation_session_service = conversation_session_service
        self.new_conversation_menu_key = new_conversation_menu_key
        self._processed_events: dict[str, float] = {}
        self._processed_events_lock = threading.Lock()

    def handle_callback(
        self,
        payload: dict[str, Any],
        *,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = self._decode_payload(payload, raw_body=raw_body, headers=headers)
        if payload.get("challenge"):
            self._verify_token(payload)
            return {"challenge": payload["challenge"]}

        self._verify_token(payload)
        event_type = self._event_type(payload)
        if event_type == "application.bot.menu_v6":
            return self._handle_bot_menu_event(payload)

        if event_type not in {"im.message.receive_v1", "message"}:
            return {"code": 0}

        if self._sender_type(payload) not in {"user", ""}:
            return {"code": 0}

        question = self._extract_question(payload)
        if not question:
            return {"code": 0}

        event_key = self._event_dedup_key(payload)
        if not self._mark_event_processing(event_key):
            return {"code": 0}

        user_id, user_id_type = self._extract_sender_identity(payload)
        try:
            query = QueryRequest(
                user_id=user_id,
                user_id_type=user_id_type,
                question=question,
                top_k=self.default_top_k,
                conversation_id=self._conversation_id(payload, user_id),
            )
            response = self.pipeline.run(query)
            receive_id, receive_id_type = self._resolve_reply_target(payload, user_id)
            self.message_client.send_text(
                receive_id=receive_id,
                receive_id_type=receive_id_type,
                text=self._build_reply_text(response),
            )
        except Exception:
            self._forget_event(event_key)
            raise
        return {"code": 0}

    def _handle_bot_menu_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_key = self._bot_menu_event_key(payload)
        if event_key != self.new_conversation_menu_key:
            return {"code": 0}

        dedup_key = self._event_dedup_key(payload)
        if not self._mark_event_processing(dedup_key):
            return {"code": 0}

        user_id, _ = self._extract_menu_operator_identity(payload)
        channel_key = self._conversation_channel_key(payload, user_id)
        conversation_id = self._start_new_conversation(channel_key)
        receive_id, receive_id_type = self._resolve_menu_reply_target(payload, user_id)
        self.message_client.send_text(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            text=(
                "已开始一个新会话。接下来我不会再参考刚才的上下文。\n"
                f"会话 ID：{conversation_id}"
            ),
        )
        return {"code": 0}

    def _decode_payload(
        self,
        payload: dict[str, Any],
        *,
        raw_body: bytes | None,
        headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        encrypted = payload.get("encrypt")
        if not encrypted:
            return payload
        if not self.encrypt_key:
            raise FeishuBotDecryptError("Feishu encrypt payload received but FEISHU_ENCRYPT_KEY is not configured.")
        normalized_headers = {key.lower(): value for key, value in (headers or {}).items()}
        self._verify_signature(raw_body=raw_body, headers=normalized_headers)
        try:
            decrypted_text = self._decrypt_string(str(encrypted))
            decrypted_payload = json.loads(decrypted_text)
        except Exception as exc:
            raise FeishuBotDecryptError("Failed to decrypt Feishu callback payload.") from exc
        if not isinstance(decrypted_payload, dict):
            raise FeishuBotDecryptError("Decrypted Feishu payload is not a JSON object.")
        return decrypted_payload

    def _verify_signature(self, *, raw_body: bytes | None, headers: dict[str, str]) -> None:
        if not self.encrypt_key:
            return
        signature = headers.get("x-lark-signature", "")
        timestamp = headers.get("x-lark-request-timestamp", "")
        nonce = headers.get("x-lark-request-nonce", "")
        if not any([signature, timestamp, nonce]):
            return
        if not all([signature, timestamp, nonce]):
            raise FeishuBotVerificationError("Incomplete Feishu signature headers.")
        body = raw_body or b""
        sign_bytes = (timestamp + nonce + self.encrypt_key).encode("utf-8") + body
        expected_signature = hashlib.sha256(sign_bytes).hexdigest()
        if expected_signature != signature:
            raise FeishuBotVerificationError("Invalid Feishu request signature.")

    def _decrypt_string(self, encrypted: str) -> str:
        try:
            from Crypto.Cipher import AES
        except ModuleNotFoundError as exc:
            raise FeishuBotDecryptError(
                "Missing dependency `pycryptodome`, cannot decrypt Feishu callback payload."
            ) from exc

        decoded = base64.b64decode(encrypted)
        if len(decoded) < AES.block_size:
            raise FeishuBotDecryptError("Encrypted Feishu payload is too short.")
        key_bytes = hashlib.sha256(self.encrypt_key.encode("utf-8")).digest()
        iv = decoded[: AES.block_size]
        ciphertext = decoded[AES.block_size :]
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv)
        plaintext = cipher.decrypt(ciphertext)
        return self._pkcs7_unpad(plaintext).decode("utf-8")

    def _pkcs7_unpad(self, data: bytes) -> bytes:
        if not data:
            raise FeishuBotDecryptError("Encrypted Feishu payload is empty after decryption.")
        padding = data[-1]
        if padding < 1 or padding > 16:
            raise FeishuBotDecryptError("Invalid Feishu payload padding.")
        return data[:-padding]

    def _verify_token(self, payload: dict[str, Any]) -> None:
        if not self.verification_token:
            return
        token = payload.get("token") or payload.get("header", {}).get("token")
        if token != self.verification_token:
            raise FeishuBotVerificationError("Invalid Feishu verification token.")

    def _event_type(self, payload: dict[str, Any]) -> str:
        return str(payload.get("header", {}).get("event_type") or payload.get("type") or "")

    def _event_dedup_key(self, payload: dict[str, Any]) -> str | None:
        header = payload.get("header", {})
        event = payload.get("event", {})
        message = event.get("message", {})
        event_id = header.get("event_id") or event.get("event_id")
        if event_id:
            return f"event:{event_id}"
        message_id = message.get("message_id") or event.get("message_id")
        if message_id:
            return f"message:{message_id}"
        return None

    def _bot_menu_event_key(self, payload: dict[str, Any]) -> str:
        event = payload.get("event", {})
        return str(
            event.get("event_key")
            or event.get("menu_key")
            or event.get("key")
            or event.get("action")
            or ""
        )

    def _mark_event_processing(self, event_key: str | None) -> bool:
        if not event_key:
            return True
        now = time.time()
        with self._processed_events_lock:
            expired_keys = [
                key for key, expire_at in self._processed_events.items()
                if expire_at <= now
            ]
            for key in expired_keys:
                self._processed_events.pop(key, None)
            if event_key in self._processed_events:
                return False
            self._processed_events[event_key] = now + self.dedup_ttl_seconds
            return True

    def _forget_event(self, event_key: str | None) -> None:
        if not event_key:
            return
        with self._processed_events_lock:
            self._processed_events.pop(event_key, None)

    def _sender_type(self, payload: dict[str, Any]) -> str:
        return str(payload.get("event", {}).get("sender", {}).get("sender_type") or "")

    def _extract_sender_identity(self, payload: dict[str, Any]) -> tuple[str, str]:
        sender_id = payload.get("event", {}).get("sender", {}).get("sender_id", {})
        if sender_id.get("open_id"):
            return str(sender_id["open_id"]), "open_id"
        if sender_id.get("user_id"):
            return str(sender_id["user_id"]), "user_id"
        if sender_id.get("union_id"):
            return str(sender_id["union_id"]), "union_id"

        legacy_event = payload.get("event", {})
        if legacy_event.get("open_id"):
            return str(legacy_event["open_id"]), "open_id"
        if legacy_event.get("user_id"):
            return str(legacy_event["user_id"]), "user_id"

        raise RuntimeError("Unable to resolve Feishu sender identity.")

    def _extract_menu_operator_identity(self, payload: dict[str, Any]) -> tuple[str, str]:
        event = payload.get("event", {})
        operator_id = event.get("operator", {}).get("operator_id", {})
        if operator_id.get("open_id"):
            return str(operator_id["open_id"]), "open_id"
        if operator_id.get("user_id"):
            return str(operator_id["user_id"]), "user_id"
        if operator_id.get("union_id"):
            return str(operator_id["union_id"]), "union_id"
        return self._extract_sender_identity(payload)

    def _resolve_reply_target(self, payload: dict[str, Any], user_id: str) -> tuple[str, str]:
        message = payload.get("event", {}).get("message", {})
        chat_type = str(message.get("chat_type") or "")
        chat_id = message.get("chat_id")
        if chat_type and chat_type != "p2p" and chat_id:
            return str(chat_id), "chat_id"
        return user_id, "open_id"

    def _resolve_menu_reply_target(self, payload: dict[str, Any], user_id: str) -> tuple[str, str]:
        event = payload.get("event", {})
        chat_id = event.get("chat_id") or event.get("chat", {}).get("chat_id")
        if chat_id:
            return str(chat_id), "chat_id"
        return user_id, "open_id"

    def _conversation_id(self, payload: dict[str, Any], user_id: str) -> str:
        channel_key = self._conversation_channel_key(payload, user_id)
        if self.conversation_session_service is not None:
            return self.conversation_session_service.active_conversation_id(
                channel_key=channel_key,
            )
        return channel_key

    def _start_new_conversation(self, channel_key: str) -> str:
        if self.conversation_session_service is not None:
            return self.conversation_session_service.start_new_conversation(
                channel_key=channel_key,
            )
        return channel_key

    def _conversation_channel_key(self, payload: dict[str, Any], user_id: str) -> str:
        message = payload.get("event", {}).get("message", {})
        event = payload.get("event", {})
        chat_id = (
            message.get("chat_id")
            or event.get("chat_id")
            or event.get("chat", {}).get("chat_id")
        )
        if chat_id:
            return f"feishu:{chat_id}"
        return f"feishu:p2p:{user_id}"

    def _extract_question(self, payload: dict[str, Any]) -> str:
        event = payload.get("event", {})
        legacy_text = event.get("text_without_at_bot")
        if legacy_text:
            return self._normalize_question(str(legacy_text))

        message = event.get("message", {})
        if str(message.get("message_type") or "") not in {"", "text"}:
            return ""

        raw_content = message.get("content")
        if isinstance(raw_content, dict):
            text = raw_content.get("text", "")
        else:
            text = self._extract_text_from_content(raw_content)
        return self._normalize_question(text)

    def _extract_text_from_content(self, raw_content: Any) -> str:
        if not raw_content:
            return ""
        if isinstance(raw_content, str):
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError:
                return raw_content
            if isinstance(content, dict):
                return str(content.get("text") or "")
            return ""
        return ""

    def _normalize_question(self, question: str) -> str:
        text = re.sub(r"<at\s+user_id=\"[^\"]+\">.*?</at>", " ", question)
        text = re.sub(r"@[^\s]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _build_reply_text(self, response: QueryResponse) -> str:
        parts = [response.answer.strip()]
        if response.version_notice:
            parts.append(response.version_notice)
        if response.citations:
            titles: list[str] = []
            seen_doc_ids: set[str] = set()
            for citation in response.citations:
                if citation.doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(citation.doc_id)
                title = citation.title
                if citation.version and citation.version not in title:
                    title = f"{title}（{citation.version}）"
                titles.append(f"- {title}")
                if len(titles) >= 2:
                    break
            parts.append("参考文档：\n" + "\n".join(titles))

        text = "\n\n".join(part for part in parts if part)
        return text[:1800]
