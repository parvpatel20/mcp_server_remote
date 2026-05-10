"""
Lightweight reranker used when no external LLM is configured.
"""
from typing import Any, Dict, List


async def llm_rerank_candidates(
    question: str,
    candidates: List[Dict[str, Any]],
    session_id: str,
    rerank_top_k: int,
) -> List[int]:
    del question, session_id

    if not candidates:
        return []

    scored: List[tuple[int, float]] = []
    for idx, candidate in enumerate(candidates):
        score = candidate.get("score")
        if score is None:
            score = float("-inf")
        scored.append((idx, float(score)))

    scored.sort(key=lambda item: item[1], reverse=True)

    limit = rerank_top_k if rerank_top_k > 0 else len(scored)
    return [idx for idx, _ in scored[:limit]]
