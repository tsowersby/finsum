"""SEC 10-K RAG Pipeline.

Summarize SEC 10-K filings with one function call.

Usage:
    from rag_project import summarize
    
    summary = summarize(
        ticker="AAPL",
        query="What are the main risk factors?",
        llm_api_key="your-mistral-key",
    )
"""
from facade import summarize
from retrieval.reranker import Reranker, make_zeroentropy_rerank_fn

__all__ = [
    "summarize",
    "Reranker",
    "make_zeroentropy_rerank_fn",
]
