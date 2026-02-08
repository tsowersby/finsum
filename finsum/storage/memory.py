"""In-memory storage for chunks and embeddings.

Pure storage layer - no retrieval logic.
Stores chunks with their embeddings in dictionaries and NumPy arrays.

Usage:
    store = ChunkStore()
    store.add(chunk, embedding)
    store.add_batch(chunks, embeddings)
    
    chunks = store.get_by_section("item1")
    all_chunks = store.get_all()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..chunking.datatypes import Chunk
from ..config import get_retrieval_config


@dataclass
class StoredChunk:
    """Chunk with its embedding."""
    chunk: Chunk
    embedding: np.ndarray


class ChunkStore:
    """Simple in-memory storage for chunks and embeddings.
    
    Stores:
    - chunks: Dict[chunk_id, StoredChunk]
    - sections: Dict[section_path, List[chunk_id]] (index)
    
    Exposes raw data for retriever to use:
    - get_vectors() → NumPy array of all embeddings
    - get_chunk_ids() → List of chunk_ids in same order as vectors
    """
    
    def __init__(self, vector_dim: int | None = None):
        cfg = get_retrieval_config()
        self.vector_dim = vector_dim if vector_dim is not None else cfg.vector_dim
        self.chunks: Dict[str, StoredChunk] = {}
        self.sections: Dict[str, List[str]] = {}
        
        # Cache for vector matrix
        self._vectors: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []
        self._dirty: bool = False
    
    def add(self, chunk: Chunk, embedding: np.ndarray) -> bool:
        """Add a chunk with embedding. Returns False if duplicate."""
        if chunk.chunk_id in self.chunks:
            return False
        
        # Normalize embedding
        embedding = np.asarray(embedding).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self.chunks[chunk.chunk_id] = StoredChunk(chunk=chunk, embedding=embedding)
        
        # Update section index
        section = chunk.metadata.section_path
        if section not in self.sections:
            self.sections[section] = []
        self.sections[section].append(chunk.chunk_id)
        
        self._dirty = True
        return True
    
    def add_batch(self, chunks: List[Chunk], embeddings: np.ndarray) -> Tuple[int, int]:
        """Add multiple chunks. Returns (added, skipped)."""
        added = skipped = 0
        for i, chunk in enumerate(chunks):
            if self.add(chunk, embeddings[i]):
                added += 1
            else:
                skipped += 1
        return added, skipped
    
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        stored = self.chunks.get(chunk_id)
        return stored.chunk if stored else None
    
    def get_by_section(self, section: str) -> List[Chunk]:
        """Get all chunks for a section."""
        chunk_ids = self.sections.get(section, [])
        return [self.chunks[cid].chunk for cid in chunk_ids if cid in self.chunks]
    
    def get_all(self) -> List[Chunk]:
        """Get all chunks."""
        return [sc.chunk for sc in self.chunks.values()]
    
    def get_vectors(self) -> np.ndarray:
        """Get matrix of all embeddings (N x dim). Rebuilds if dirty."""
        if self._dirty or self._vectors is None:
            self._rebuild()
        return self._vectors if self._vectors is not None else np.array([]).reshape(0, self.vector_dim)
    
    def get_chunk_ids(self) -> List[str]:
        """Get chunk IDs in same order as get_vectors()."""
        if self._dirty or self._vectors is None:
            self._rebuild()
        return self._chunk_ids
    
    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding for a chunk."""
        stored = self.chunks.get(chunk_id)
        return stored.embedding if stored else None
    
    def list_sections(self) -> List[str]:
        """List all section paths."""
        return list(self.sections.keys())
    
    def count(self) -> int:
        """Total chunk count."""
        return len(self.chunks)
    
    def clear(self) -> None:
        """Clear all data."""
        self.chunks.clear()
        self.sections.clear()
        self._vectors = None
        self._chunk_ids = []
        self._dirty = False
    
    def _rebuild(self) -> None:
        """Rebuild vector matrix cache."""
        if not self.chunks:
            self._vectors = None
            self._chunk_ids = []
        else:
            self._chunk_ids = list(self.chunks.keys())
            self._vectors = np.vstack([self.chunks[cid].embedding for cid in self._chunk_ids])
        self._dirty = False
