from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    helpful: bool
    is_obsolete: bool = False
    note: str | None = None
