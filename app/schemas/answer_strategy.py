from __future__ import annotations

from pydantic import BaseModel


class AnswerStrategyResult(BaseModel):
    mode: str
    reason: str
    preferred_doc_id: str | None = None
    compare_doc_id: str | None = None
    use_latest_as_primary: bool = False
    include_version_notice: bool = True
    include_diff_summary: bool = False
    answer_old_first: bool = False
