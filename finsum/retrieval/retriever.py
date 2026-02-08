"""Retriever - Vector search for chunks.

Owns all search logic:
- Embeds queries with sentence-transformers
- Cosine similarity search on ChunkStore vectors
- Optional reranking with Zero Entropy API

Usage:
    retriever = Retriever(store)
    chunks = retriever.search(query, sections=["item1", "item7"], top_k=10)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from finsum.chunking.datatypes import Chunk
from finsum.config import get_retrieval_config
from finsum.storage.memory import ChunkStore


@dataclass
class RetrievedChunk:
    """Chunk with similarity score."""
    chunk: Chunk
    score: float


class Retriever:
    """Vector search over ChunkStore.
    
    Example:
        store = ChunkStore()
        # ... add chunks ...
        
        retriever = Retriever(store, embedding_model="BAAI/bge-large-en-v1.5")
        results = retriever.search("revenue growth", sections=["item7"], top_k=5)
    """
    
    def __init__(
        self,
        store: ChunkStore,
        embedding_model: str | None = None,
    ):
        cfg = get_retrieval_config()
        self.store = store
        self.embedding_model = embedding_model or cfg.embedding_model
        self._embedder = None
    
    @property
    def embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model, device="cpu")
        return self._embedder
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        result = self.embedder.encode(text, normalize_embeddings=True)
        return np.asarray(result)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        result = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(result)
    
    def search(
        self,
        query: str,
        sections: Optional[List[str]] = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> List[RetrievedChunk]:
        """Search for relevant chunks.
        
        Args:
            query: Search query
            sections: Filter to these sections (e.g., ["item1", "item7"])
            top_k: Max results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of RetrievedChunk sorted by score descending
        """
        cfg = get_retrieval_config()
        top_k = top_k if top_k is not None else cfg.top_k
        min_score = min_score if min_score is not None else cfg.min_score
        
        vectors = self.store.get_vectors()
        if vectors.size == 0:
            return []
        
        chunk_ids = self.store.get_chunk_ids()
        
        # Embed query
        query_vec = self.embed(query).reshape(1, -1)
        
        # Cosine similarity (vectors are already normalized)
        scores = (vectors @ query_vec.T).flatten()
        
        # Build results
        results: List[RetrievedChunk] = []
        for i, (chunk_id, score) in enumerate(zip(chunk_ids, scores)):
            if score < min_score:
                continue
            chunk = self.store.get(chunk_id)
            if chunk is None:
                continue
            if sections and chunk.metadata.section_path not in sections:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
        
        # Sort by score and take top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def get_by_section(self, section: str) -> List[Chunk]:
        """Direct lookup - get all chunks for a section."""
        return self.store.get_by_section(section)
