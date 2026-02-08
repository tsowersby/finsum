"""Chunking pipeline.

Pipeline stages:
1. Block Segmentation - Identify typed blocks (heading, text)
2. Section-Aware Chunking - Attach section context and create chunks
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..config.settings import ChunkingConfig
from .block_segmenter import Block, BlockStream, segment_text
from .datatypes import Chunk, ChunkMetadata
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION TRACKING
# =============================================================================

@dataclass
class SectionContext:
    """Tracks current section hierarchy for section-aware chunking."""
    path: List[str] = field(default_factory=list)
    heading_level: int = 0
    
    def push_heading(self, text: str, level: int) -> None:
        """Push a heading onto the section stack."""
        # Pop headings at same or lower level
        while self.path and self.heading_level >= level:
            self.path.pop()
            self.heading_level = max(0, self.heading_level - 1)
        
        self.path.append(text)
        self.heading_level = level
    
    def get_path(self) -> List[str]:
        """Get current section path."""
        return list(self.path) if self.path else []


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

@dataclass 
class PipelineConfig:
    """Configuration for chunking pipeline."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class ChunkingPipeline:
    """Chunking pipeline with section tracking.
    
    Processes text through:
    1. Block Segmentation - Identify headings and text
    2. Chunking - Create chunks with section context
    
    Heading chunks are created but marked exclude_from_retrieval=True
    since section path is in all chunk metadata anyway.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.text_chunker = TextChunker(self.config.chunking)
    
    def process(self, text: str, source: str, company: str = "") -> List[Chunk]:
        """Process text through chunking pipeline.
        
        Args:
            text: Raw text content
            source: Source identifier
            company: Company ticker symbol
            
        Returns:
            List of Chunk objects with metadata
        """
        if not text or not text.strip():
            return []
        
        # Phase 1: Block Segmentation
        block_stream = segment_text(text, source)
        
        # Phase 2: Section-aware chunking
        chunks = self._chunk_with_sections(block_stream.blocks, source, company)
        
        return chunks
    
    def _chunk_with_sections(
        self,
        blocks: List[Block],
        source: str,
        company: str = "",
    ) -> List[Chunk]:
        """Chunk blocks with section context attached.
        
        - Headings: Update section context (no chunk created)
        - Text/Tables: Paragraph/sentence chunking
        """
        chunks = []
        section = SectionContext()
    
        for block in blocks:
            # Headings: update section path only
            if block.type == "heading" and block.heading_text:
                section.push_heading(block.heading_text, block.heading_level or 2)
                continue
        
            # Get section path as string (e.g. "item1" or "item7/revenue")
            section_path = "/".join(section.get_path()) if section.get_path() else ""
        
            # Text/Table: paragraph/sentence chunking
            text_chunks = self.text_chunker.chunk(
                block.content,
                source,
                section_path,
                company,
            )
            chunks.extend(text_chunks)
    
        return chunks


def chunk_text(
    text: str,
    source: str,
    company: str = "",
    config: Optional[PipelineConfig] = None,
) -> List[Chunk]:
    """Chunk text using the pipeline.
    
    Args:
        text: Raw text content
        source: Source identifier
        company: Company ticker
        config: Optional pipeline configuration
        
    Returns:
        List of chunks with metadata
    """
    pipeline = ChunkingPipeline(config)
    return pipeline.process(text, source, company)
