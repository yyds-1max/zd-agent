from pydantic import BaseModel, Field

from app.schemas.citation import Citation
from app.schemas.user import UserProfile


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user_id: str | None = Field(default=None, description="可选飞书用户标识（类型由 FEISHU_USER_ID_TYPE 控制）")
    user_profile: UserProfile | None = Field(default=None, description="可选用户画像；不传则由系统识别")


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    version_hint: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class QueryClickRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    title: str | None = None
    position: int | None = None
    action: str = Field(default="open_citation")
    user_id: str | None = None
