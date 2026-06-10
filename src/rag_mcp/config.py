import os
from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration loaded from .env via pydantic-settings."""

    # ── Essential ──
    embedding_baseurl: str
    embedding_model: str
    embedding_api_key: str

    # ── Rerank gate (all-or-nothing) ──
    rerank_baseurl: str | None = None
    rerank_model: str | None = None
    rerank_api_key: str | None = None

    # ── Non-essential (None = not used) ──
    embedding_top_k: int | None = None
    embedding_input_token: int | None = None
    rerank_top_k: int | None = None
    rerank_score_threshold: float | None = None
    rerank_input_token: int | None = None

    # ── Database ──
    database_path: str = "./database"

    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    model_config = {"env_file": _env_path, "env_file_encoding": "utf-8"}

    @model_validator(mode="after")
    def _validate_rerank_group(self):
        keys = [self.rerank_baseurl, self.rerank_model, self.rerank_api_key]
        filled = sum(1 for k in keys if k is not None)
        if filled not in (0, 3):
            raise ValueError(
                "RERANK_BASEURL, RERANK_MODEL, RERANK_API_KEY: "
                "must all be set together or all left empty"
            )
        return self
