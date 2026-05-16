"""
RAG agent for answering questions using Freshdesk KB articles.

Vector retrieve -> FIRST_LLM rerank -> top chunks for the main agent.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from llm_rerank import llm_rerank_candidates
from embeddings import embedding_model
from pinecone_client import pinecone_client
from settings import settings
from utils import format_article_url

logger = logging.getLogger(__name__)


def _normalize_article_id_for_url(article_id: Any) -> str:
    """Convert article IDs to stable string form for URL construction."""
    if isinstance(article_id, float) and article_id.is_integer():
        return str(int(article_id))

    value = str(article_id).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]

    return value


class RAGAgent:
    """RAG retrieval using LangChain embeddings with native Pinecone."""

    def __init__(self):
        self.initialized = False
        self._init_lock = asyncio.Lock()

    async def ensure_initialized(self) -> None:
        """Load embeddings and Pinecone once; safe under concurrent tool calls."""
        if self.initialized:
            return
        async with self._init_lock:
            if self.initialized:
                return
            await self.initialize()

    async def initialize(self) -> None:
        """Initialize heavy retrieval resources once at server startup."""
        try:
            logger.info("Loading Embedding Model...")
            await asyncio.to_thread(embedding_model.load_model)

            logger.info("Initializing Pinecone Index...")
            await asyncio.to_thread(pinecone_client.initialize_index)

            self.initialized = True
            logger.info("RAG retrieval resources initialized")

        except Exception as e:
            logger.error(f"Failed to initialize RAG retrieval resources: {e}")
            raise

    async def answer_question(
        self,
        question: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve KB chunks (candidate pool), rerank with FIRST_LLM, return top chunks and citations.
        """
        del conversation_history

        await self.ensure_initialized()

        try:
            logger.debug(f"Generating embedding for question: {question}")

            question_embedding = await embedding_model.aembed_query(question)

            pool_k = settings.rag_candidate_top_k
            if isinstance(top_k, int) and top_k > 0:
                pool_k = max(pool_k, top_k)

            logger.debug(f"Searching Pinecone for top {pool_k} matches")
            search_results = await asyncio.to_thread(
                pinecone_client.query,
                query_vector=question_embedding,
                top_k=pool_k,
                include_metadata=True,
            )

            matches_list = list(search_results.matches or [])
            candidates: List[Dict[str, Any]] = []
            rows: List[Tuple[Any, Dict[str, Any], str, str]] = []

            for match in matches_list:
                metadata = match.metadata or {}
                chunk_text = metadata.get("chunk_text", "")
                article_title = metadata.get("title", "Unknown")
                label = f"[{article_title}]\n{chunk_text}"
                rows.append((match, metadata, chunk_text, article_title))
                candidates.append(
                    {
                        "text": label,
                        "score": float(match.score) if match.score is not None else None,
                    }
                )

            rerank_k = settings.rag_rerank_top_k
            if candidates:
                order = await llm_rerank_candidates(
                    question,
                    candidates,
                    session_id=session_id,
                    rerank_top_k=rerank_k,
                )
            else:
                order = []

            context_chunks: List[str] = []
            sources: List[Dict[str, Any]] = []
            seen_source_keys: set[str] = set()

            for pos in order:
                if pos < 0 or pos >= len(rows):
                    continue
                match, metadata, chunk_text, article_title = rows[pos]
                block = f"[{article_title}]\n{chunk_text}"
                context_chunks.append(block)

                article_id = metadata.get("article_id")
                source_key = str(article_id) if article_id else article_title
                if source_key in seen_source_keys:
                    continue
                seen_source_keys.add(source_key)

                source_info: Dict[str, Any] = {
                    "article_id": article_id,
                    "title": article_title,
                    "score": float(match.score) if match.score is not None else None,
                }
                if article_id:
                    try:
                        normalized_article_id = _normalize_article_id_for_url(article_id)
                        source_info["url"] = format_article_url(
                            settings.freshdesk_domain_name, normalized_article_id
                        )
                    except (TypeError, ValueError):
                        pass
                sources.append(source_info)

            context = "\n\n---\n\n".join(context_chunks)

            source_text = ""
            if sources:
                source_text = "\n\n**Sources:**\n"
                for source in sources[: settings.rag_rerank_top_k]:
                    title = source.get("title", "Unknown Article")
                    url = source.get("url")
                    if url:
                        source_text += f"- [{title}]({url})\n"
                    else:
                        source_text += f"- {title}\n"

            logger.info(
                "Freshdesk RAG | session=%s | candidates=%d | reranked_chunks=%d",
                session_id,
                len(matches_list),
                len(context_chunks),
            )

            return {
                "answer": context,
                "formatted_citations": source_text,
                "sources": sources[: settings.rag_rerank_top_k],
                "session_id": session_id,
                "candidate_pool_size": pool_k,
                "top_k": rerank_k,
            }

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise


# Global instance
rag_agent = RAGAgent()
