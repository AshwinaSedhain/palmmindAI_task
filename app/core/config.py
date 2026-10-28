from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Groq LLM config
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"

    # vector db
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    sqlite_db_path: str = "./data/app.db"

    # embedding config
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    top_k_results: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
