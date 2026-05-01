from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    fixtures_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "fixtures"
    )
    feishu_directory_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "data"
        / "feishu_users.json"
    )
    vector_store_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "CHROMA_PERSIST_DIR",
                str(Path(__file__).resolve().parents[2] / "storage" / "vector" / "chroma"),
            )
        )
    )
    conversation_store_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "CONVERSATION_STORE_PATH",
                str(Path(__file__).resolve().parents[2] / "storage" / "conversations.json"),
            )
        )
    )
    conversation_history_limit: int = Field(
        default_factory=lambda: int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6"))
    )
    conversation_session_store_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "CONVERSATION_SESSION_STORE_PATH",
                str(Path(__file__).resolve().parents[2] / "storage" / "conversation_sessions.json"),
            )
        )
    )
    chroma_collection_name: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "knowledge_documents")
    )
    default_top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    feishu_app_id: str | None = Field(default_factory=lambda: os.getenv("FEISHU_APP_ID"))
    feishu_app_secret: str | None = Field(default_factory=lambda: os.getenv("FEISHU_APP_SECRET"))
    feishu_contact_api_enabled: bool = Field(
        default_factory=lambda: os.getenv("FEISHU_CONTACT_API_ENABLED", "false").lower() == "true"
    )
    feishu_bot_enabled: bool = Field(
        default_factory=lambda: os.getenv("FEISHU_BOT_ENABLED", "false").lower() == "true"
    )
    feishu_default_user_id_type: str = Field(
        default_factory=lambda: os.getenv("FEISHU_DEFAULT_USER_ID_TYPE", "open_id")
    )
    feishu_log_level: str = Field(default_factory=lambda: os.getenv("FEISHU_LOG_LEVEL", "INFO"))
    feishu_use_directory_fallback: bool = Field(
        default_factory=lambda: os.getenv("FEISHU_USE_DIRECTORY_FALLBACK", "true").lower() == "true"
    )
    feishu_api_base_url: str = Field(
        default_factory=lambda: os.getenv("FEISHU_API_BASE_URL", "https://open.feishu.cn")
    )
    feishu_bot_event_path: str = Field(
        default_factory=lambda: os.getenv("FEISHU_BOT_EVENT_PATH", "/feishu/events")
    )
    feishu_new_conversation_menu_key: str = Field(
        default_factory=lambda: os.getenv("FEISHU_NEW_CONVERSATION_MENU_KEY", "start_new_conversation")
    )
    feishu_verification_token: str | None = Field(
        default_factory=lambda: os.getenv("FEISHU_VERIFICATION_TOKEN")
    )
    feishu_encrypt_key: str | None = Field(
        default_factory=lambda: os.getenv("FEISHU_ENCRYPT_KEY")
    )
    dashscope_api_key: str | None = Field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY"))
    enable_user_profile_llm: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_USER_PROFILE_LLM", "false").lower() == "true"
    )
    enable_conversation_router_llm: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_CONVERSATION_ROUTER_LLM", "false").lower() == "true"
    )
    enable_answer_llm: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_ANSWER_LLM", "false").lower() == "true"
    )
    enable_version_diff_llm: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_VERSION_DIFF_LLM", "false").lower() == "true"
    )
    dashscope_llm_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    dashscope_llm_model: str = Field(
        default_factory=lambda: os.getenv("DASHSCOPE_LLM_MODEL", "qwen3-max")
    )
    dashscope_llm_temperature: float = Field(
        default_factory=lambda: float(os.getenv("DASHSCOPE_LLM_TEMPERATURE", "0.2"))
    )
    dashscope_llm_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("DASHSCOPE_LLM_TIMEOUT_SECONDS", "40"))
    )
    dashscope_embedding_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    dashscope_embedding_model: str = Field(
        default_factory=lambda: os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    )
    dashscope_embedding_dimensions: int = Field(
        default_factory=lambda: int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024"))
    )
    dashscope_rerank_url: str = Field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_RERANK_URL",
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        )
    )
    dashscope_rerank_model: str = Field(
        default_factory=lambda: os.getenv("DASHSCOPE_RERANK_MODEL", "gte-rerank-v2")
    )
    retrieval_candidate_limit: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "8"))
    )


def _load_dotenv(root_dir: Path) -> None:
    env_path = root_dir / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[2]
    _load_dotenv(root_dir)
    return Settings()
