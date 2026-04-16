from app.services.rerank_service import RerankService


def test_rerank_should_prefer_latest_when_relevance_is_close() -> None:
    service = RerankService(enable_model_rerank=False)
    rows = [
        {
            "document": "住宿标准 500 元/晚，10 天内报销。",
            "distance": 0.10,
            "metadata": {"is_latest": False, "version": "V1.0", "effective_date": "2025-01-01"},
        },
        {
            "document": "请按当前制度执行。",
            "distance": 0.25,
            "metadata": {"is_latest": True},
        },
    ]

    ranked = service.rerank("报销标准", rows, top_n=2)

    assert ranked[0]["metadata"]["is_latest"] is True


def test_rerank_should_use_model_scores_when_available() -> None:
    service = RerankService(enable_model_rerank=True)
    rows = [
        {"document": "A", "distance": 0.05, "metadata": {"is_latest": True}},
        {"document": "B", "distance": 0.90, "metadata": {"is_latest": False}},
    ]
    service._model_scores = lambda query, rs: {0: 0.10, 1: 0.99}  # type: ignore[method-assign]

    ranked = service.rerank("test", rows, top_n=2)

    assert ranked[0]["document"] == "B"
