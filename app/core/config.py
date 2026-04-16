from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "知达 Agent Demo"
    app_env: str = "dev"
    app_port: int = 8000

    vector_backend: str = "chroma"
    vector_dir: str = "storage/vector"
    chroma_collection: str = "knowledge_chunks"
    metadata_db: str = "storage/sqlite/metadata.db"
    feishu_base_url: str = "https://open.feishu.cn"
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_tenant_access_token: str = ""
    feishu_user_id_type: str = "open_id"
    feishu_department_id_type: str = "open_department_id"
    feishu_http_timeout: float = 8.0
    feishu_encrypt_key: str = ""
    feishu_verification_token: str = ""

    embedding_model: str = "text-embedding-v4"
    chat_model: str = "qwen3-max"
    answer_model: str = "qwen3-max"
    enable_llm_answer_generation: bool = True
    version_diff_model: str = "qwen3-max"
    enable_llm_version_diff: bool = True
    chat_conclusion_model: str = "qwen3-max"
    enable_llm_chat_conclusion: bool = False
    rerank_model: str = "gte-rerank"
    enable_model_rerank: bool = False
    text_chunk_size: int = 500
    text_chunk_overlap: int = 80
    enable_push: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
