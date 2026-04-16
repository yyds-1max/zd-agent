import base64
import json

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def _encrypt_json(payload: dict, key_text: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    pad = 16 - (len(data) % 16)
    padded = data + bytes([pad]) * pad
    key = key_text.encode("utf-8")
    cipher = Cipher(algorithms.AES(key), modes.CBC(key[:16]))
    encryptor = cipher.encryptor()
    cipher_bytes = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(cipher_bytes).decode("utf-8")


def test_feishu_url_verification_plain_should_return_challenge() -> None:
    client = TestClient(app)
    resp = client.post("/feishu/events", json={"type": "url_verification", "challenge": "abc123"})
    assert resp.status_code == 200
    assert resp.json() == {"challenge": "abc123"}


def test_feishu_url_verification_encrypted_should_return_challenge() -> None:
    client = TestClient(app)
    old_key = settings.feishu_encrypt_key
    settings.feishu_encrypt_key = "0123456789abcdef0123456789abcdef"
    encrypted = _encrypt_json({"type": "url_verification", "challenge": "xyz789"}, settings.feishu_encrypt_key)
    try:
        resp = client.post("/feishu/events", json={"encrypt": encrypted})
    finally:
        settings.feishu_encrypt_key = old_key
    assert resp.status_code == 200
    assert resp.json() == {"challenge": "xyz789"}


def test_feishu_event_should_validate_verification_token() -> None:
    client = TestClient(app)
    old_token = settings.feishu_verification_token
    settings.feishu_verification_token = "token-1"
    try:
        bad = client.post("/feishu/events", json={"type": "event_callback", "token": "token-2"})
        ok = client.post("/feishu/events", json={"type": "event_callback", "token": "token-1"})
    finally:
        settings.feishu_verification_token = old_token

    assert bad.status_code == 403
    assert ok.status_code == 200
    assert ok.json()["code"] == 0
