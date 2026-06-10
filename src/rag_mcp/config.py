from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration loaded from .env via pydantic-settings."""

    # Embedding
    embedding_baseurl: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    embedding_api_key: str
    vector_dimension: int = 2048
    embedding_top_k: int = 10
    embedding_input_token: int = 131_072

    # Rerank
    rerank_baseurl: str = "https://ai.api.nvidia.com/v1"
    rerank_model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    rerank_api_key: str
    rerank_top_k: int = 5
    rerank_score_threshold: float = 0.0
    rerank_input_token: int = 8_192

    # Database
    database_path: str = "./database"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @classmethod
    def environment_key(cls) -> str:
        return "RAG_MCP"
