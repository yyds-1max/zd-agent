from datetime import date, datetime

from app.schemas.citation import Citation
from app.services.answer_service import AnswerService


def test_answer_service_should_include_required_sections() -> None:
    service = AnswerService(enable_llm_generation=False)
    answer = service.compose(
        question="出差报销最新标准是什么？",
        citations=[
            Citation(
                doc_id="fixture-02:0",
                title="字节跳动员工差旅与报销制度（V2.0）",
                source_type="policy",
                chunk_text="国内出差餐补为 100 元/天，当天往返为 50 元/天。",
                version="V2.0",
                is_latest=True,
                effective_date=date(2026, 3, 15),
                updated_at=datetime(2026, 3, 15, 0, 0, 0),
                role_scope=["*"],
                department_scope=["*"],
                project_scope=["*"],
            )
        ],
    )

    assert "【答案摘要】" in answer
    assert "【引用来源】" in answer
    assert "【版本】" in answer
    assert "【生效时间】" in answer
    assert "【适用范围】" in answer
    assert "【制度类答复】" in answer
    assert "【制度口径】" in answer


def test_answer_service_should_include_required_sections_when_empty() -> None:
    service = AnswerService(enable_llm_generation=False)
    answer = service.compose(question="没有命中的问题", citations=[])

    assert "【答案摘要】" in answer
    assert "【引用来源】" in answer
    assert "【版本】" in answer
    assert "【生效时间】" in answer
    assert "【适用范围】" in answer


def test_answer_service_should_use_general_template_for_non_policy() -> None:
    service = AnswerService(enable_llm_generation=False)
    answer = service.compose(
        question="项目北极星本周进展是什么？",
        citations=[
            Citation(
                doc_id="fixture-06:0",
                title="项目北极星_周报_2026W14",
                source_type="project",
                chunk_text="本周已完成需求评审与接口联调。",
                version="V1.0",
                is_latest=True,
                role_scope=["pm"],
                department_scope=["delivery"],
                project_scope=["A"],
            )
        ],
    )

    assert "【答案摘要】" in answer
    assert "【引用来源】" in answer
    assert "【版本】" in answer
    assert "【生效时间】" in answer
    assert "【适用范围】" in answer
    assert "【制度类答复】" not in answer
    assert "【制度口径】" not in answer


def test_answer_service_should_append_version_notice_when_old_version_hit() -> None:
    service = AnswerService(enable_llm_generation=False)
    answer = service.compose(
        question="旧制度内容是什么？",
        citations=[
            Citation(
                doc_id="fixture-01:0",
                title="字节跳动员工差旅与报销制度（V1.0）",
                source_type="policy",
                chunk_text="旧版报销时效为 15 天。",
                version="V1.0",
                is_latest=False,
                role_scope=["*"],
                department_scope=["*"],
                project_scope=["*"],
            )
        ],
        intent_type="policy",
        version_hint="当前有更新版本：V2.0。你当前命中的是旧版本：V1.0。",
    )

    assert "【版本更新提醒】" in answer
    assert "当前有更新版本：V2.0" in answer


def test_answer_service_should_use_llm_output_when_available() -> None:
    service = AnswerService(enable_llm_generation=True)
    service._compose_with_llm = lambda q, c, i, v: (  # type: ignore[method-assign]
        "【答案摘要】\nLLM答案\n\n"
        "【引用来源】\n1. 文档（doc_id=fixture-02:0）\n\n"
        "【版本】\nV2.0\n\n"
        "【生效时间】\n2026-03-15\n\n"
        "【适用范围】\n全员"
    )
    answer = service.compose(
        question="出差报销最新标准是什么？",
        citations=[
            Citation(
                doc_id="fixture-02:0",
                title="字节跳动员工差旅与报销制度（V2.0）",
                source_type="policy",
                chunk_text="国内出差餐补为 100 元/天，当天往返为 50 元/天。",
                version="V2.0",
                is_latest=True,
                role_scope=["*"],
                department_scope=["*"],
                project_scope=["*"],
            )
        ],
        intent_type="policy",
    )

    assert "LLM答案" in answer


def test_answer_service_should_explain_old_and_latest_when_question_targets_old_version() -> None:
    service = AnswerService(enable_llm_generation=False)
    answer = service.compose(
        question="字节跳动员工差旅与报销制度V1.0里报销时限是多少？",
        citations=[
            Citation(
                doc_id="fixture-02:0",
                title="字节跳动员工差旅与报销制度（V2.0）",
                source_type="policy",
                chunk_text="报销需在出差结束后 10 个自然日内发起。",
                version="V2.0",
                is_latest=True,
                effective_date=date(2026, 3, 15),
                updated_at=datetime(2026, 3, 15, 0, 0, 0),
                role_scope=["*"],
                department_scope=["*"],
                project_scope=["*"],
            ),
            Citation(
                doc_id="fixture-01:0",
                title="字节跳动员工差旅与报销制度（V1.0）",
                source_type="policy",
                chunk_text="报销需在出差结束后 5 个工作日内发起。",
                version="V1.0",
                is_latest=False,
                effective_date=date(2026, 2, 10),
                updated_at=datetime(2026, 2, 10, 0, 0, 0),
                role_scope=["*"],
                department_scope=["*"],
                project_scope=["*"],
            ),
        ],
        intent_type="policy",
        version_hint="检测到多个版本，已优先返回最新版本。当前版本：V2.0",
    )

    assert "指定查询版本：V1.0；当前生效版本：V2.0" in answer
    assert "【版本更新提醒】" in answer
    assert "【新旧差异】" in answer
