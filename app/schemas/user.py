from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    role: str = Field(..., description="角色名称，例如员工/财务/项目经理")
    department: str = Field(..., description="部门名称")
    projects: list[str] = Field(default_factory=list, description="关联项目列表")
