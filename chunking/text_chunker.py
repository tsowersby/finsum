"""Text chunker with paragraph-based splitting.

Strategy:
1. Split by paragraphs (double newlines)
2. If paragraph too long → split by sentences with 1-sentence overlap
3. If sentence too long → split by words (fallback)
"""
from __future__ import annotations

import re
import logging
from typing import List, Optional

from chunking.datatypes import Chunk, ChunkMetadata
from config.settings import ChunkingConfig

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunks text content into appropriately sized pieces."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.max_chars = self.config.max_chunk_chars
        self.min_chars = self.config.min_chunk_chars
    
    def chunk(
        self,
        content: str,
        source: str,
        section_path: str = "",
        company: str = "",
    ) -> List[Chunk]:
        """Chunk text content into appropriately sized pieces.
        
        Args:
            content: Text content
            source: Source identifier
            section_path: Section path (e.g. "item1", "item7")
            company: Company ticker
            
        Returns:
            List of Chunk objects with metadata
        """
        if not content or not content.strip():
            return []
        
        content = content.strip()
        
        # Single chunk case: fits within limit
        if len(content) <= self.max_chars:
            if len(content) >= self.min_chars:
                return [self._create_chunk(
                    content, source, section_path, company
                )]
            return []
        
        # Split by paragraphs
        text_pieces = self._split_by_paragraphs(content)
        
        # Create chunks from text pieces
        chunks = []
        for text in text_pieces:
            if len(text) >= self.min_chars:
                chunks.append(self._create_chunk(
                    text, source, section_path, company
                ))
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph boundaries (double newlines)."""
        paragraphs = re.split(r'\n\s*\n', text)
        
        results = []
        current = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Oversized paragraph: split by sentences
            if len(para) > self.max_chars:
                if current:
                    results.append(current)
                    current = ""
                results.extend(self._split_by_sentences(para))
                continue
            
            # Try adding paragraph to current chunk
            if len(current) + len(para) + 2 <= self.max_chars:
                current = f"{current}\n\n{para}".strip()
            else:
                if current:
                    results.append(current)
                current = para
        
        if current:
            results.append(current)
        
        return results
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences with 1-sentence overlap."""
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text] if text.strip() else []
        
        results = []
        current = ""
        last_sentence = ""  # For overlap
        
        for sentence in sentences:
            # Oversized sentence: split by words
            if len(sentence) > self.max_chars:
                if current:
                    results.append(current)
                    current = ""
                    last_sentence = ""
                results.extend(self._split_by_words(sentence))
                # Set last_sentence to end of last word chunk for overlap
                if results:
                    words = results[-1].split()
                    if words:
                        last_sentence = words[-1] if len(words) == 1 else " ".join(words[-3:])
                continue
            
            # Try adding sentence to current chunk
            test_chunk = f"{current} {sentence}".strip() if current else sentence
            
            if len(test_chunk) <= self.max_chars:
                current = test_chunk
                last_sentence = sentence
            else:
                # Current chunk is full - save it
                if current:
                    results.append(current)
                
                # Start new chunk with overlap (last sentence + current sentence)
                if last_sentence and len(last_sentence) + len(sentence) + 1 <= self.max_chars:
                    current = f"{last_sentence} {sentence}"
                else:
                    current = sentence
                last_sentence = sentence
        
        if current:
            results.append(current)
        
        return results
    
    def _split_by_words(self, text: str) -> List[str]:
        """Split oversized text by words (last resort fallback)."""
        words = text.split()
        
        if not words:
            return []
        
        results = []
        current = ""
        
        for word in words:
            test = f"{current} {word}".strip() if current else word
            
            if len(test) <= self.max_chars:
                current = test
            else:
                if current:
                    results.append(current)
                # Handle single word longer than max_chars (truncate)
                current = word[:self.max_chars] if len(word) > self.max_chars else word
        
        if current:
            results.append(current)
        
        return results
    
    def _create_chunk(
        self,
        content: str,
        source: str,
        section_path: str,
        company: str,
    ) -> Chunk:
        """Create a text chunk with proper metadata."""
        metadata = ChunkMetadata(
            source=source,
            section_path=section_path,
            content_type="text",
            company=company,
        )
        return Chunk.create(content=content, metadata=metadata)
