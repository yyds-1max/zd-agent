from pydantic import BaseModel, Field


class QueryIntent(BaseModel):
    original_question: str
    retrieval_query: str
    intent_type: str = "general"
    keywords: list[str] = Field(default_factory=list)
    need_latest: bool = False
    source: str = "rule"
