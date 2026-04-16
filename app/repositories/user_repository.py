class UserRepository:
    def get_profile(self, user_id: str) -> dict:
        # MVP: 用固定映射模拟身份系统；找不到时返回默认员工画像。
        fixtures = {
            "u-employee": {"role": "employee", "department": "hr", "projects": []},
            "u-finance": {"role": "finance", "department": "finance", "projects": []},
            "u-pm-a": {"role": "pm", "department": "delivery", "projects": ["A"]},
        }
        profile = fixtures.get(user_id, {"role": "employee", "department": "general", "projects": []})
        return {"user_id": user_id, **profile}
