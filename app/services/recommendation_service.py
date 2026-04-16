import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.repositories.recommendation_repository import RecommendationRepository
from app.schemas.recommendation import RecommendationItem, PushTriggerResponse
from app.schemas.user import UserProfile


class RecommendationService:
    def __init__(self, repo: RecommendationRepository | None = None) -> None:
        self._repo = repo or RecommendationRepository()

    def recommend(self, user: UserProfile, top_k: int = 5) -> list[RecommendationItem]:
        candidates = self._repo.list_latest_knowledge(limit=300)
        if not candidates:
            return []

        queries = self._repo.list_recent_queries(user.user_id, limit=30)
        clicks = self._repo.list_recent_clicks(user.user_id, limit=30)
        tokens = self._build_interest_tokens(queries, clicks)
        clicked_doc_ids = {str(item.get("doc_id") or "").strip() for item in clicks if item.get("doc_id")}
        clicked_doc_ids.discard("")
        source_pref = self._build_source_preference(queries)

        ranked: list[RecommendationItem] = []
        for row in candidates:
            if not self._is_allowed(row, user):
                continue
            item = self._score_candidate(
                row=row,
                user=user,
                interest_tokens=tokens,
                clicked_doc_ids=clicked_doc_ids,
                source_pref=source_pref,
            )
            ranked.append(item)

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(1, int(top_k))]

    def trigger_push(self, *, user: UserProfile, top_k: int = 5, channel: str = "manual") -> PushTriggerResponse:
        items = self.recommend(user, top_k=top_k)
        if not items:
            return PushTriggerResponse(
                status="no_items",
                user_id=user.user_id,
                item_count=0,
                items=[],
                message="当前没有可推送的推荐内容。",
            )

        payload = json.dumps([item.model_dump() for item in items], ensure_ascii=False)
        push_id = self._repo.save_push_log(
            {
                "user_id": user.user_id,
                "channel": channel,
                "item_count": len(items),
                "payload": payload,
                "status": "triggered",
                "trigger_reason": "manual_trigger",
            }
        )
        return PushTriggerResponse(
            status="triggered",
            push_id=str(push_id),
            user_id=user.user_id,
            item_count=len(items),
            items=items,
            message=f"已触发推送，包含 {len(items)} 条推荐。",
        )

    def _score_candidate(
        self,
        *,
        row: dict[str, Any],
        user: UserProfile,
        interest_tokens: Counter[str],
        clicked_doc_ids: set[str],
        source_pref: Counter[str],
    ) -> RecommendationItem:
        doc_id = str(row.get("doc_id") or "").strip()
        title = str(row.get("title") or "未命名文档").strip() or "未命名文档"
        source_type = str(row.get("source_type") or "unknown").strip() or "unknown"
        summary = (str(row.get("summary") or "").strip() or None)
        version = str(row.get("version") or "unknown").strip() or "unknown"
        updated_at = str(row.get("updated_at") or "").strip() or None
        tags = self._parse_scope(row.get("tags"))

        reasons: list[str] = ["角色/部门/项目权限匹配"]
        score = 1.5

        freshness_score = self._freshness_score(row.get("updated_at"), row.get("effective_date"))
        if freshness_score > 0:
            score += 1.8 * freshness_score
            reasons.append("文档近期更新")

        token_match_score = self._token_match_score(
            title=title,
            summary=summary,
            tags=tags,
            interest_tokens=interest_tokens,
        )
        if token_match_score > 0:
            score += 1.4 * token_match_score
            reasons.append("匹配最近搜索主题")

        if doc_id and doc_id in clicked_doc_ids:
            score += 0.9
            reasons.append("你最近点击过相关文档")

        source_boost = min(1.0, 0.25 * source_pref.get(source_type, 0))
        if source_boost > 0:
            score += source_boost
            reasons.append("符合近期关注的知识类型")

        # 项目强匹配额外加权，保证项目场景优先。
        project_scope = self._parse_scope(row.get("project_scope"))
        if project_scope and "*" not in project_scope and any(project in project_scope for project in user.projects):
            score += 0.6
            reasons.append("与当前项目直接相关")

        return RecommendationItem(
            doc_id=doc_id,
            title=title,
            source_type=source_type,
            version=version,
            updated_at=updated_at,
            summary=summary,
            score=round(score, 4),
            reasons=self._dedupe(reasons),
        )

    def _is_allowed(self, row: dict[str, Any], user: UserProfile) -> bool:
        role_scope = self._parse_scope(row.get("role_scope"))
        department_scope = self._parse_scope(row.get("department_scope"))
        project_scope = self._parse_scope(row.get("project_scope"))
        return (
            self._scope_match(role_scope, [user.role])
            and self._scope_match(department_scope, [user.department])
            and self._scope_match(project_scope, user.projects)
        )

    @staticmethod
    def _build_interest_tokens(queries: list[dict[str, Any]], clicks: list[dict[str, Any]]) -> Counter[str]:
        tokens: Counter[str] = Counter()

        for idx, row in enumerate(queries):
            text = " ".join(
                [
                    str(row.get("question") or ""),
                    str(row.get("retrieval_query") or ""),
                ]
            )
            weight = max(1, 6 - idx)
            for token in RecommendationService._extract_tokens(text):
                tokens[token] += weight

        for idx, row in enumerate(clicks):
            text = " ".join([str(row.get("title") or ""), str(row.get("doc_id") or "")])
            weight = max(1, 8 - idx)
            for token in RecommendationService._extract_tokens(text):
                tokens[token] += weight

        return tokens

    @staticmethod
    def _build_source_preference(queries: list[dict[str, Any]]) -> Counter[str]:
        mapping = {
            "policy": "policy",
            "faq": "faq",
            "project": "project",
            "chat_summary": "chat_summary",
        }
        counter: Counter[str] = Counter()
        for row in queries:
            intent_type = str(row.get("intent_type") or "").strip()
            source_type = mapping.get(intent_type)
            if source_type:
                counter[source_type] += 1
        return counter

    @staticmethod
    def _extract_tokens(text: str) -> list[str]:
        raw = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
        stop_words = {
            "什么",
            "怎么",
            "可以",
            "需要",
            "我们",
            "你们",
            "一下",
            "这个",
            "那个",
            "请问",
            "最新",
            "当前",
            "制度",
            "文档",
            "项目",
            "问题",
        }
        tokens: list[str] = []
        for token in raw:
            normalized = token.lower()
            if normalized in stop_words:
                continue
            tokens.append(normalized)
        return tokens

    @staticmethod
    def _token_match_score(
        *, title: str, summary: str | None, tags: list[str], interest_tokens: Counter[str]
    ) -> float:
        if not interest_tokens:
            return 0.0
        haystack = " ".join([title.lower(), (summary or "").lower(), " ".join(tag.lower() for tag in tags)])
        if not haystack.strip():
            return 0.0
        score = 0.0
        for token, weight in interest_tokens.items():
            if token and token in haystack:
                score += min(1.2, 0.08 * weight)
        return min(2.0, score)

    @staticmethod
    def _freshness_score(updated_at: Any, effective_date: Any) -> float:
        date_text = str(updated_at or effective_date or "").strip()
        if not date_text:
            return 0.0
        parsed = RecommendationService._parse_datetime(date_text)
        if parsed is None:
            return 0.0
        now = datetime.now(timezone.utc)
        delta_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
        if delta_days <= 7:
            return 1.0
        if delta_days <= 30:
            return 0.75
        if delta_days <= 90:
            return 0.45
        if delta_days <= 180:
            return 0.25
        return 0.1

    @staticmethod
    def _parse_datetime(text: str) -> datetime | None:
        raw = text.strip()
        if not raw:
            return None
        if "T" not in raw:
            raw = f"{raw}T00:00:00"
        raw = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _parse_scope(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return [text]
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [text]
        return []

    @staticmethod
    def _scope_match(scope: list[str], user_values: list[str]) -> bool:
        if not scope:
            return False
        if "*" in scope:
            return True
        return any(value in scope for value in user_values if value)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            text = item.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result
