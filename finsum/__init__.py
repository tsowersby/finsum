"""SEC 10-K RAG Pipeline.

Summarize SEC 10-K filings with one function call.

Usage:
    from finsum import summarize
    
    summary = summarize(
        ticker="AAPL",
        item="1a",
        query="What are the main risk factors?",
        llm_api_key="your-mistral-key",
    )
"""
from .facade import summarize

__all__ = ["summarize"]
