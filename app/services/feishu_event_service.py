import base64
import json
from typing import Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from app.core.config import settings


class FeishuEventService:
    """飞书事件回调解析：支持 encrypt 解密与 token 校验。"""

    @classmethod
    def parse_event(cls, payload: dict[str, Any]) -> dict[str, Any]:
        encrypted = payload.get("encrypt")
        if isinstance(encrypted, str) and encrypted.strip():
            return cls._decrypt_payload(encrypted.strip())
        return payload

    @classmethod
    def validate_token(cls, payload: dict[str, Any]) -> bool:
        configured = settings.feishu_verification_token.strip()
        if not configured:
            return True
        incoming = str(payload.get("token") or "").strip()
        if not incoming:
            # 一些事件协议把 token 放在 header 中
            header = payload.get("header")
            if isinstance(header, dict):
                incoming = str(header.get("token") or "").strip()
        if not incoming:
            return False
        return incoming == configured

    @classmethod
    def _decrypt_payload(cls, encrypted: str) -> dict[str, Any]:
        key = settings.feishu_encrypt_key.encode("utf-8")
        if len(key) != 32:
            raise ValueError("FEISHU_ENCRYPT_KEY 必须为 32 字节。")

        cipher_bytes = base64.b64decode(encrypted)
        iv = key[:16]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        plain = decryptor.update(cipher_bytes) + decryptor.finalize()
        plain = cls._pkcs7_unpad(plain)

        # 模式 A：直接是 JSON
        parsed = cls._try_parse_json(plain)
        if parsed is not None:
            return parsed

        # 模式 B：random(16) + msg_len(4, big-endian) + msg + app_id
        if len(plain) > 20:
            msg_len = int.from_bytes(plain[16:20], byteorder="big", signed=False)
            if 0 < msg_len <= len(plain) - 20:
                msg = plain[20 : 20 + msg_len]
                parsed = cls._try_parse_json(msg)
                if parsed is not None:
                    return parsed

        raise ValueError("飞书回调解密成功，但未解析出有效 JSON。")

    @staticmethod
    def _pkcs7_unpad(data: bytes) -> bytes:
        if not data:
            raise ValueError("空密文。")
        pad = data[-1]
        if pad <= 0 or pad > 16:
            raise ValueError("无效的 PKCS7 padding。")
        if data[-pad:] != bytes([pad]) * pad:
            raise ValueError("无效的 PKCS7 padding。")
        return data[:-pad]

    @staticmethod
    def _try_parse_json(raw: bytes) -> dict[str, Any] | None:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
