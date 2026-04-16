from pathlib import Path
from textwrap import dedent

from app.services.document_parser_service import DocumentParserService


def test_parse_file_should_refine_raw_chat_before_chunking(tmp_path: Path) -> None:
    chat_file = tmp_path / "14_项目北极星_项目群聊天记录_2026年4月.txt"
    chat_file.write_text(
        dedent(
            """
            文档类型： 聊天记录
            权限级别： 项目组成员、项目经理 可见
            版本号： 2026-04
            发布日期： 2026-04-08

            正文

            2026-04-02 10:12 张三：这周我们先确定首期接入范围。
            2026-04-02 10:13 李四：确认，首期只接制度文档、FAQ 和项目周报。
            2026-04-02 10:16 王五：权限策略统一为三类角色：员工、财务、项目经理。
            2026-04-02 10:20 张三：版本策略决定优先最新版本，命中旧版时提示差异。
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = DocumentParserService()
    chunks = parser.parse_directory(str(tmp_path))

    assert chunks
    assert chunks[0]["source_type"] == "chat_summary"
    assert "关键结论" in chunks[0]["content_chunk"]
    assert "优先最新版本" in chunks[0]["content_chunk"]
