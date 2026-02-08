"""Configuration loaded from settings.json."""

from pathlib import Path
from typing import Any
import json
from pydantic import BaseModel

_CONFIG_PATH = Path(__file__).parent / "settings.json"


def _load_json() -> dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class ChunkingConfig(BaseModel):
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    min_chunk_chars: int = 50
    max_chunk_chars: int = 2200


class LLMConfig(BaseModel):
    model: str = "mistral-small-latest"
    temperature: float = 0.3
    max_tokens: int = 1024


class RetrievalConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_dim: int = 384
    top_k: int = 10
    min_score: float = 0.0
    rerank_top_k: int = 20


_config: dict[str, Any] | None = None


def _get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = _load_json()
    return _config


def get_chunking_config() -> ChunkingConfig:
    return ChunkingConfig(**_get_config().get("chunking", {}))


def get_llm_config() -> LLMConfig:
    return LLMConfig(**_get_config().get("llm", {}))


def get_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(**_get_config().get("retrieval", {}))
