from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from app.services.feishu_bot_service import (
    FeishuBotDecryptError,
    FeishuBotService,
    FeishuBotVerificationError,
)


def create_feishu_bot_router(
    service: FeishuBotService,
    *,
    path: str = "/feishu/events",
):
    router = APIRouter()

    @router.post(path)
    async def callback(request: Request) -> dict:
        raw_body = await request.body()
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        try:
            return service.handle_callback(
                payload,
                raw_body=raw_body,
                headers=dict(request.headers),
            )
        except FeishuBotVerificationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except FeishuBotDecryptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
