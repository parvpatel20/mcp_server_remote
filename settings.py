"""
Runtime settings loaded from environment variables.
"""
import os
from dataclasses import dataclass


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    freshdesk_domain_name: str = os.getenv("FRESHDESK_DOMAIN", "")
    rag_candidate_top_k: int = _get_env_int("RAG_CANDIDATE_TOP_K", 15)
    rag_rerank_top_k: int = _get_env_int("RAG_RERANK_TOP_K", 5)

    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "freshdesk-kb")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_dimension: int = _get_env_int("PINECONE_DIMENSION", 1024)


settings = Settings()
