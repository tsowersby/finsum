"""Storage module.

Pure in-memory storage for chunks and embeddings.
"""
from storage.memory import ChunkStore, StoredChunk

__all__ = [
    "ChunkStore",
    "StoredChunk",
]
