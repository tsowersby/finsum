"""Core Data Types - Standardized chunk output with content hashing."""
from dataclasses import dataclass
from typing import List, Optional
import hashlib


def content_hash(text: str) -> str:
    """Generate SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


@dataclass
class ChunkMetadata:
    """Per-chunk metadata for retrieval."""
    source: str
    section_path: str  # e.g. "item1", "item7"
    content_type: str  # "text", "table", or "heading"
    company: str  # ticker symbol
    

@dataclass
class Chunk:
    """
    Represents a single chunk of text ready for embedding/retrieval.
    
    Attributes:
        chunk_id: Unique ID based on content hash (for deduplication)
        content: The text to embed and search.
        metadata: Structured chunk metadata.
    """
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    
    @classmethod
    def create(cls, content: str, metadata: ChunkMetadata) -> "Chunk":
        """Create chunk with auto-generated content-based ID."""
        chunk_id = f"{metadata.company}_{metadata.section_path}_{content_hash(content)}"
        return cls(chunk_id=chunk_id, content=content, metadata=metadata)