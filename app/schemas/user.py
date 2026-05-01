from __future__ import annotations

from pydantic import BaseModel, Field


class DirectoryUser(BaseModel):
    user_id: str
    user_id_type: str = "user_id"
    name: str
    department: str
    department_id: str | None = None
    title: str
    level: str
    job_level_id: str | None = None
    role: str
    projects: list[str] = Field(default_factory=list)
    managed_projects: list[str] = Field(default_factory=list)
    is_new_hire: bool = False
    source: str = "mock_feishu_directory"


class UserProfile(DirectoryUser):
    project_mentions: list[str] = Field(default_factory=list)
    active_projects: list[str] = Field(default_factory=list)
    intent_hint: str | None = None

    @property
    def accessible_projects(self) -> set[str]:
        return set(self.projects) | set(self.managed_projects)
