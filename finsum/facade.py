"""SEC 10-K RAG Pipeline.

One-liner API for summarizing SEC 10-K filings:

    summarize("AAPL", "1a", "What are the main risk factors?", llm_api_key="...", reranker_api_key="...")

That's it.
"""
from __future__ import annotations

from typing import List, Optional

from chunking.datatypes import Chunk
from chunking.pipeline import ChunkingPipeline
from ingestion.downloader import FilingDownloader
from storage.memory import ChunkStore
from retrieval.retriever import Retriever


def summarize(
    ticker: str,
    item: str,
    query: str,
    llm_api_key: str,
    reranker_api_key: Optional[str] = None,
    top_k: int = 10,
) -> str:
    """Summarize SEC 10-K filing content in response to a query.
    
    Downloads the latest 10-K, chunks the specified item section,
    retrieves relevant chunks, and generates a summary using an LLM.
    
    Args:
        ticker: Company ticker symbol (e.g., "AAPL")
        item: Item section to analyze (e.g., "1a", "7", "1")
        query: Question or topic to summarize
        llm_api_key: Mistral API key (required)
        reranker_api_key: Optional Zero Entropy API key for reranking
        top_k: Number of chunks to use for context
        
    Returns:
        String summary of the relevant information
        
    Raises:
        ValueError: If required params not provided
        RuntimeError: If filing download or processing fails
        
    Example:
        summary = summarize(
            ticker="MSFT",
            item="1a",
            query="What are the main risk factors?",
            llm_api_key="your-mistral-key",
        )
        print(summary)
    """
    if not llm_api_key:
        raise ValueError("llm_api_key is required")
    if not item:
        raise ValueError("item is required (e.g., '1a', '7')")
    
    # Normalize item key
    item_key = f"item{item}" if not item.lower().startswith("item") else item.lower()
    
    # 1. Download filing
    downloader = FilingDownloader()
    filing = downloader.download(ticker)
    if not filing:
        raise RuntimeError(f"Failed to download 10-K filing for {ticker}")
    
    # 2. Chunk the specified section only
    if item_key not in filing.sections:
        available = ", ".join(filing.sections.keys())
        raise ValueError(f"Item '{item_key}' not found. Available: {available}")
    
    section_info = filing.sections[item_key]
    chunker = ChunkingPipeline()
    chunks = chunker.process(
        text=section_info.content,
        source=f"{ticker}_10K_{item_key}",
        company=ticker,
    )
    
    if not chunks:
        raise RuntimeError(f"No chunks extracted from {item_key} for {ticker}")
    
    # 3. Store with embeddings
    store = ChunkStore()
    retriever = Retriever(store)
    
    texts = [c.content for c in chunks]
    embeddings = retriever.embed_batch(texts)
    store.add_batch(chunks, embeddings)
    
    # 4. Retrieve relevant chunks
    results = retriever.search(query, top_k=top_k)
    if not results:
        return "No relevant information found in the filing for this query."
    
    # 5. Rerank if API key provided
    if reranker_api_key and len(results) > 1:
        from retrieval.reranker import make_zeroentropy_rerank_fn, Reranker
        rerank_fn = make_zeroentropy_rerank_fn(reranker_api_key, "zerank-2")
        reranker = Reranker(rerank_fn=rerank_fn)
        results = reranker.rerank(query, results)
    
    # 6. Generate summary
    from inference.language_model import LLMClient
    llm = LLMClient(api_key=llm_api_key)
    
    context = "\n\n---\n\n".join([r.chunk.content for r in results])
    return llm.generate(query, context)
