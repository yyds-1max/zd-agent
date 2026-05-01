from __future__ import annotations

from app.schemas.query import QueryRequest, QueryResponse


def create_query_router(pipeline):
    from fastapi import APIRouter

    router = APIRouter()

    @router.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        return pipeline.run(request)

    return router
