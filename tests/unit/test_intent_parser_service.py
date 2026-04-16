from app.schemas.intent import QueryIntent
from app.schemas.user import UserProfile
from app.services.intent_parser_service import IntentParserService


def test_parse_should_fallback_to_rule_when_llm_unavailable() -> None:
    service = IntentParserService()
    service._parse_with_llm = lambda question, user: None  # type: ignore[method-assign]
    user = UserProfile(user_id="u1", role="employee", department="hr", projects=["A"])

    intent = service.parse("出差报销制度最新标准是什么？", user)

    assert intent.source == "rule"
    assert intent.intent_type == "policy"
    assert intent.need_latest is True
    assert intent.retrieval_query == "出差报销制度最新标准是什么？"


def test_parse_should_use_llm_result_when_available() -> None:
    service = IntentParserService()
    expected = QueryIntent(
        original_question="原问题",
        retrieval_query="改写后的检索词",
        intent_type="project",
        keywords=["项目北极星", "里程碑"],
        need_latest=False,
        source="llm",
    )
    service._parse_with_llm = lambda question, user: expected  # type: ignore[method-assign]
    user = UserProfile(user_id="u1", role="pm", department="delivery", projects=["A"])

    intent = service.parse("原问题", user)

    assert intent is expected


def test_normalize_llm_payload_should_sanitize_fields() -> None:
    service = IntentParserService()
    payload = {
        "intent_type": "UNKNOWN",
        "retrieval_query": "",
        "keywords": "报销, 制度, 最新",
        "need_latest": False,
    }

    intent = service._normalize_llm_payload("报销最新制度是什么", payload)

    assert intent.intent_type == "general"
    assert intent.retrieval_query == "报销最新制度是什么"
    assert intent.keywords == ["报销", "制度", "最新"]
    assert intent.need_latest is True
    assert intent.source == "llm"


def test_parse_with_rules_should_identify_recommendation_request() -> None:
    service = IntentParserService()
    intent = service._parse_with_rules("我最近负责采购审批，有什么相关更新可以推荐？")

    assert intent.intent_type == "recommendation"
