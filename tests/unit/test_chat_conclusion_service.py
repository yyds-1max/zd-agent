from textwrap import dedent

from app.services.chat_conclusion_service import ChatConclusionService


def test_refine_should_extract_structured_conclusions_from_raw_chat() -> None:
    service = ChatConclusionService(enable_llm=False)
    raw_chat = dedent(
        """
        2026-04-02 10:12 张三：这周我们先确定首期接入范围。
        2026-04-02 10:13 李四：确认，首期只接制度文档、FAQ 和项目周报。
        2026-04-02 10:16 王五：权限策略统一为三类角色：员工、财务、项目经理。
        2026-04-02 10:20 张三：版本策略决定优先最新版本，命中旧版时提示差异。
        2026-04-02 10:22 李四：后续需要补充推送规则文案，下周跟进。
        """
    ).strip()

    refined = service.refine(raw_chat, title="项目北极星群聊记录")

    assert "关键结论" in refined
    assert "自动提炼" in refined
    assert "首期只接制度文档、FAQ 和项目周报" in refined
    assert "优先最新版本" in refined
    assert "待跟进事项" in refined


def test_refine_should_keep_existing_structured_summary() -> None:
    service = ChatConclusionService(enable_llm=False)
    structured = dedent(
        """
        项目群结论汇总（2026 年 4 月）

        一、关于接入范围
        经讨论确认，首期只接入制度文档和 FAQ。

        二、关于权限策略
        经讨论确认，采用三角色权限模型。
        """
    ).strip()

    refined = service.refine(structured, title="项目群结论汇总")

    assert refined == structured


def test_refine_should_fallback_when_chat_format_is_not_detected() -> None:
    service = ChatConclusionService(enable_llm=False)
    plain_text = "这是一次非标准格式的聊天导出文本，没有明确发言人格式。"

    refined = service.refine(plain_text, title="导出文本")

    assert "关键结论" in refined
    assert "自动提炼" in refined
