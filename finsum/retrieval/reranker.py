"""Reranker - Client-agnostic reranking interface.

Reranks retrieved chunks using a provided rerank function.
Does not assume any specific reranking API or model.

# TODO: Add local reranking support (e.g., cross-encoder models)

Usage:
    # With a custom rerank function
    def my_rerank(query: str, documents: List[str]) -> List[Tuple[int, float]]:
        # Returns list of (index, score) tuples
        ...
    
    reranker = Reranker(rerank_fn=my_rerank)
    reranked = reranker.rerank(query, results)
"""
from __future__ import annotations

from typing import Callable, List, Tuple, TYPE_CHECKING

from config import get_retrieval_config

if TYPE_CHECKING:
    from retrieval.retriever import RetrievedChunk

# Type for rerank function: (query, documents) -> [(index, score), ...]
RerankFn = Callable[[str, List[str]], List[Tuple[int, float]]]


class Reranker:
    """Client-agnostic reranker.
    
    Takes a rerank function that scores query-document pairs.
    The function should return (index, score) tuples sorted by relevance.
    
    Example:
        def my_rerank_fn(query: str, docs: List[str]) -> List[Tuple[int, float]]:
            # Call your reranking API here
            response = my_api.rerank(query=query, documents=docs)
            return [(r.index, r.score) for r in response.results]
        
        reranker = Reranker(rerank_fn=my_rerank_fn)
        reranked = reranker.rerank("revenue growth", results)
    """
    
    def __init__(self, rerank_fn: RerankFn):
        """Initialize with a rerank function.
        
        Args:
            rerank_fn: Function that takes (query, documents) and returns
                       list of (index, score) tuples sorted by relevance.
        """
        if not callable(rerank_fn):
            raise ValueError("rerank_fn must be callable")
        self._rerank_fn = rerank_fn
    
    def rerank(
        self,
        query: str,
        results: "List[RetrievedChunk]",
        top_k: int | None = None,
    ) -> "List[RetrievedChunk]":
        """Rerank retrieved chunks.
        
        Args:
            query: Search query
            results: List of RetrievedChunk from retriever
            top_k: Max results to return
            
        Returns:
            Reranked list with updated scores
            
        Raises:
            RuntimeError: If reranking fails
        """
        if not results:
            return []
        
        cfg = get_retrieval_config()
        top_k = top_k if top_k is not None else cfg.rerank_top_k
        
        from retrieval.retriever import RetrievedChunk
        
        documents = [r.chunk.content for r in results]
        
        try:
            ranked = self._rerank_fn(query, documents)
        except Exception as e:
            raise RuntimeError(f"Reranking failed: {e}") from e
        
        # Build reranked list
        reranked = []
        for idx, score in ranked[:top_k]:
            if idx < len(results):
                original = results[idx]
                reranked.append(RetrievedChunk(
                    chunk=original.chunk,
                    score=float(score),
                ))
        
        return reranked


def make_zeroentropy_rerank_fn(api_key: str, model: str) -> RerankFn:
    """Create a rerank function using Zero Entropy API.
    
    Args:
        api_key: Zero Entropy API key
        model: Model name (e.g., "zerank-2")
        
    Returns:
        Rerank function compatible with Reranker
        
    Example:
        rerank_fn = make_zeroentropy_rerank_fn("your-key", "zerank-2")
        reranker = Reranker(rerank_fn=rerank_fn)
    """
    try:
        from zeroentropy import ZeroEntropy
    except ImportError:
        raise ImportError("zeroentropy package required: pip install zeroentropy")
    
    client = ZeroEntropy(api_key=api_key)
    
    def rerank_fn(query: str, documents: List[str]) -> List[Tuple[int, float]]:
        response = client.models.rerank(
            model=model,
            query=query,
            documents=documents,
        )
        return [(r.index, r.relevance_score) for r in response.results]
    
    return rerank_fn
