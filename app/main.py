import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from fastapi import FastAPI

from app.api.routes import feedback, feishu, health, ingest, query, recommendation
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.include_router(health.router, tags=["健康检查"])
app.include_router(ingest.router, prefix="/ingest", tags=["知识入库"])
app.include_router(query.router, prefix="/query", tags=["问答查询"])
app.include_router(recommendation.router, prefix="/recommendation", tags=["个性化推荐"])
app.include_router(feedback.router, prefix="/feedback", tags=["反馈记录"])
app.include_router(feishu.router, prefix="/feishu", tags=["飞书回调"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.app_port, reload=True)
