from datetime import date

from app.schemas.citation import Citation
from app.services.version_service import VersionService


class _FakeMetadataRepo:
    def __init__(self, docs):
        self._docs = docs

    def list_document_versions(self, source_type=None):
        if source_type is None:
            return self._docs
        return [doc for doc in self._docs if doc.get("source_type") == source_type]


def test_build_version_hint_should_warn_when_hit_old_version() -> None:
    old_hit = Citation(
        doc_id="old-doc",
        title="字节跳动员工差旅与报销制度（V1.0）",
        source_type="policy",
        chunk_text="旧版要求报销在 15 天内提交。",
        version="V1.0",
        is_latest=False,
        effective_date=date(2025, 1, 1),
        role_scope=["*"],
        department_scope=["*"],
        project_scope=["*"],
    )
    repo = _FakeMetadataRepo(
        [
            {
                "doc_id": "new-doc",
                "title": "字节跳动员工差旅与报销制度（V2.0）",
                "source_type": "policy",
                "version": "V2.0",
                "is_latest": 1,
                "effective_date": "2026-03-15",
                "updated_at": "2026-03-15T00:00:00",
                "summary": "新版要求报销在 10 天内提交，并补充超标审批说明。",
                "content_chunk": "新版要求报销在 10 天内提交，并补充超标审批说明。",
            }
        ]
    )
    service = VersionService(metadata_repo=repo)  # type: ignore[arg-type]

    hint = service.build_version_hint([old_hit])

    assert hint is not None
    assert "当前有更新版本" in hint
    assert "更新时间" in hint
    assert "主要差异" in hint
    assert "V2.0" in hint


def test_build_version_hint_should_show_current_when_latest() -> None:
    latest_hit = Citation(
        doc_id="new-doc",
        title="制度文档",
        source_type="policy",
        chunk_text="当前版本内容",
        version="V2.0",
        is_latest=True,
        role_scope=["*"],
        department_scope=["*"],
        project_scope=["*"],
    )
    service = VersionService(metadata_repo=_FakeMetadataRepo([]))  # type: ignore[arg-type]

    hint = service.build_version_hint([latest_hit])

    assert hint == "当前版本：V2.0"


def test_build_version_hint_should_use_llm_diff_when_available() -> None:
    old_hit = Citation(
        doc_id="old-doc",
        title="字节跳动员工差旅与报销制度（V1.0）",
        source_type="policy",
        chunk_text="旧版要求报销在 15 天内提交。",
        version="V1.0",
        is_latest=False,
        role_scope=["*"],
        department_scope=["*"],
        project_scope=["*"],
    )
    repo = _FakeMetadataRepo(
        [
            {
                "doc_id": "new-doc",
                "title": "字节跳动员工差旅与报销制度（V2.0）",
                "source_type": "policy",
                "version": "V2.0",
                "is_latest": 1,
                "effective_date": "2026-03-15",
                "updated_at": "2026-03-15T00:00:00",
                "summary": "新版要求报销在 10 天内提交，并补充超标审批说明。",
                "content_chunk": "新版要求报销在 10 天内提交，并补充超标审批说明。",
            }
        ]
    )
    service = VersionService(metadata_repo=repo, enable_llm_diff=True)  # type: ignore[arg-type]
    service._build_diff_with_llm = lambda old_text, new_text: "LLM总结：报销时效由15天调整为10天，并新增超标审批要求。"  # type: ignore[method-assign]

    hint = service.build_version_hint([old_hit])

    assert hint is not None
    assert "LLM总结" in hint
    assert "当前有更新版本" in hint


def test_build_version_hint_should_warn_when_question_targets_old_version() -> None:
    latest_hit = Citation(
        doc_id="new-doc",
        title="字节跳动员工差旅与报销制度（V2.0）",
        source_type="policy",
        chunk_text="新版要求报销在 10 天内提交。",
        version="V2.0",
        is_latest=True,
        role_scope=["*"],
        department_scope=["*"],
        project_scope=["*"],
    )
    old_hit = Citation(
        doc_id="old-doc",
        title="字节跳动员工差旅与报销制度（V1.0）",
        source_type="policy",
        chunk_text="旧版要求报销在 15 天内提交。",
        version="V1.0",
        is_latest=False,
        role_scope=["*"],
        department_scope=["*"],
        project_scope=["*"],
    )
    repo = _FakeMetadataRepo(
        [
            {
                "doc_id": "new-doc",
                "title": "字节跳动员工差旅与报销制度（V2.0）",
                "source_type": "policy",
                "version": "V2.0",
                "is_latest": 1,
                "effective_date": "2026-03-15",
                "updated_at": "2026-03-15T00:00:00",
                "summary": "新版要求报销在 10 天内提交。",
                "content_chunk": "新版要求报销在 10 天内提交。",
            }
        ]
    )
    service = VersionService(metadata_repo=repo)  # type: ignore[arg-type]

    hint = service.build_version_hint(
        [latest_hit, old_hit],
        question="字节跳动员工差旅与报销制度V1.0里报销时限是多少？",
    )

    assert hint is not None
    assert "当前有更新版本" in hint
    assert "旧版本：V1.0" in hint
