from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.feishu_event_service import FeishuEventService

router = APIRouter()


@router.post("/events")
async def receive_event(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"code": 400, "msg": "invalid json"})
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"code": 400, "msg": "invalid payload"})

    try:
        event_payload = FeishuEventService.parse_event(payload)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"code": 400, "msg": f"decrypt failed: {exc}"})

    event_type = str(event_payload.get("type") or "").strip()
    if event_type == "url_verification":
        challenge = event_payload.get("challenge")
        if not isinstance(challenge, str) or not challenge:
            return JSONResponse(status_code=400, content={"code": 400, "msg": "missing challenge"})
        return JSONResponse(content={"challenge": challenge})

    if not FeishuEventService.validate_token(event_payload):
        return JSONResponse(status_code=403, content={"code": 403, "msg": "verification token mismatch"})

    # 先保证飞书回调成功，后续可在这里分发事件到 query/recommendation 等业务处理链路。
    return JSONResponse(content={"code": 0, "msg": "success"})
