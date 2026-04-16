from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipelines.query_pipeline import QueryPipeline
from app.repositories.query_log_repository import QueryLogRepository
from app.schemas.feedback import FeedbackRequest
from app.schemas.query import QueryRequest
from app.schemas.user import UserProfile
from app.services.feedback_service import FeedbackService
from app.services.recommendation_service import RecommendationService

INPUT_PATTERN = re.compile(
    r"^\s*问题[:：]\s*(?P<question>.+?)\s*[。.]?\s*职级[:：]\s*(?P<role>.+?)\s*[。.]?\s*部门[:：]\s*(?P<department>.+?)\s*[。.]?\s*$"
)


def normalize_role(raw: str) -> str:
    text = raw.strip().lower()
    if any(token in text for token in ("财务", "finance")):
        return "finance"
    if any(token in text for token in ("pm", "项目经理", "负责人", "manager", "管理", "pmo", "交付")):
        return "pm"
    return "employee"


def normalize_department(raw: str) -> str:
    text = raw.strip().lower()
    if any(token in text for token in ("财务", "finance")):
        return "finance"
    if any(token in text for token in ("hr", "人力")):
        return "hr"
    if any(token in text for token in ("运营", "operations")):
        return "operations"
    if any(token in text for token in ("交付", "delivery")):
        return "delivery"
    if any(token in text for token in ("研发", "工程", "开发", "engineering")):
        return "engineering"
    return text or "unknown"


def extract_projects(question: str) -> list[str]:
    projects: set[str] = set()
    if "北极星" in question:
        projects.add("A")
    for match in re.findall(r"(?:项目|负责)\s*([A-Za-z]\w*)", question):
        projects.add(match.upper())
    for match in re.findall(r"\b([A-Za-z])项目\b", question):
        projects.add(match.upper())
    return sorted(projects) if projects else ["unknown"]


def build_user_profile(question: str, role_raw: str, dept_raw: str) -> UserProfile:
    role = normalize_role(role_raw)
    department = normalize_department(dept_raw)
    projects = extract_projects(question)
    user_id = f"cli-{role}-{department}"
    return UserProfile(
        user_id=user_id,
        role=role,
        department=department,
        projects=projects,
    )


def prompt_user_input() -> tuple[str, str, str]:
    while True:
        line = input("\n请输入：").strip()
        match = INPUT_PATTERN.match(line)
        if not match:
            print("输入格式不正确，请按：问题：***。职级：***。部门：***。")
            continue
        question = match.group("question").strip()
        role_raw = match.group("role").strip()
        department_raw = match.group("department").strip()
        if not question or not role_raw or not department_raw:
            print("问题、职级、部门都不能为空，请重新输入。")
            continue
        return question, role_raw, department_raw


def print_answer(resp) -> None:
    print("\n================ 回答结果 ================")
    print(f"query_id: {resp.query_id}")
    if resp.version_hint:
        print(f"version_hint: {resp.version_hint}")
    print("\n【答案】")
    print(resp.answer)

    print("\n【引用】")
    if not resp.citations:
        print("无")
        return
    for idx, item in enumerate(resp.citations, start=1):
        updated = item.updated_at.isoformat() if item.updated_at else "未知"
        snippet = re.sub(r"\s+", " ", item.content_chunk).strip()
        if len(snippet) > 120:
            snippet = f"{snippet[:120]}..."
        print(
            f"{idx}. [{item.doc_id}] {item.title} | source={item.source_type} | "
            f"version={item.version} | updated_at={updated}"
        )
        print(f"   片段: {snippet}")


def prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{label} ({suffix}): ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true", "是"}


def print_recommendations(items) -> None:
    print("\n================ 推荐结果 ================")
    if not items:
        print("暂无推荐内容。")
        return
    for idx, item in enumerate(items, start=1):
        reasons = "；".join(item.reasons) if item.reasons else "无"
        updated = item.updated_at or "未知"
        summary = item.summary or "无摘要"
        print(
            f"{idx}. [{item.doc_id}] {item.title} | source={item.source_type} | "
            f"version={item.version} | score={item.score} | updated_at={updated}"
        )
        print(f"   reasons: {reasons}")
        print(f"   summary: {summary}")


def handle_post_actions(
    *,
    user: UserProfile,
    query_id: str,
    citations,
    query_log_repo: QueryLogRepository,
    feedback_service: FeedbackService,
    recommendation_service: RecommendationService,
) -> bool:
    print(
        "\n可选操作：\n"
        "  1) click <引用序号>   记录引用点击\n"
        "  2) feedback         提交反馈\n"
        "  3) recommend        查看推荐\n"
        "  4) push             触发推送\n"
        "  5) next             继续下一个问题\n"
        "  6) exit             退出\n"
    )
    while True:
        command = input("请输入操作: ").strip()
        if not command:
            continue

        lower = command.lower()
        if lower == "next":
            return True
        if lower == "exit":
            return False
        if lower == "feedback":
            helpful = prompt_yes_no("这条回答是否有帮助", default=True)
            is_obsolete = prompt_yes_no("是否命中旧文档/信息过期", default=False)
            note = input("补充说明（可留空）: ").strip() or None
            result = feedback_service.save(
                FeedbackRequest(
                    query_id=query_id,
                    helpful=helpful,
                    is_obsolete=is_obsolete,
                    note=note,
                )
            )
            print(f"反馈提交结果: {result}")
            continue
        if lower == "recommend":
            items = recommendation_service.recommend(user, top_k=5)
            print_recommendations(items)
            continue
        if lower == "push":
            result = recommendation_service.trigger_push(user=user, top_k=5, channel="cli-manual")
            print(f"推送触发结果: status={result.status}, push_id={result.push_id}, item_count={result.item_count}")
            print_recommendations(result.items)
            continue
        if lower.startswith("click"):
            parts = command.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("click 命令格式错误，应为：click <引用序号>")
                continue
            idx = int(parts[1]) - 1
            if idx < 0 or idx >= len(citations):
                print("引用序号超出范围，请重新输入。")
                continue
            citation = citations[idx]
            click_id = query_log_repo.save_click_log(
                {
                    "query_id": query_id,
                    "doc_id": citation.doc_id,
                    "title": citation.title,
                    "position": idx + 1,
                    "action": "open_citation",
                    "user_id": user.user_id,
                }
            )
            print(f"点击已记录，click_id={click_id}")
            continue

        print("未知操作，请输入 click/feedback/recommend/push/next/exit。")


def main() -> None:
    print("知达 Agent CLI 演示")
    print("请输入固定格式：问题：***。职级：***。部门：***。")
    print("示例：问题：出差报销最新标准是什么？。职级：普通员工。部门：人力。")

    query_pipeline = QueryPipeline()
    query_log_repo = QueryLogRepository()
    feedback_service = FeedbackService()
    recommendation_service = RecommendationService()

    while True:
        question, role_raw, department_raw = prompt_user_input()
        user_profile = build_user_profile(question, role_raw, department_raw)
        print(
            f"\n用户画像识别结果: user_id={user_profile.user_id}, role={user_profile.role}, "
            f"department={user_profile.department}, projects={user_profile.projects}"
        )

        try:
            resp = query_pipeline.run(
                QueryRequest(
                    question=question,
                    user_profile=user_profile,
                )
            )
        except Exception as exc:
            print(f"\n问答执行失败: {exc}")
            print("请确认已执行知识入库、并正确配置模型与向量库依赖。")
            continue

        print_answer(resp)
        should_continue = handle_post_actions(
            user=user_profile,
            query_id=resp.query_id,
            citations=resp.citations,
            query_log_repo=query_log_repo,
            feedback_service=feedback_service,
            recommendation_service=recommendation_service,
        )
        if not should_continue:
            break

    print("\n已退出 CLI 演示。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，程序退出。")
