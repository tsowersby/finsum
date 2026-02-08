"""Chunking module - document segmentation.

Components:
- ChunkingPipeline: Structure-preserving chunking with section paths
- TextChunker: Paragraph/sentence splitting
- BlockSegmenter: Identifies typed blocks (heading, text, table)
"""
from .datatypes import Chunk, ChunkMetadata
from .block_segmenter import Block, BlockStream, segment_text
from .pipeline import ChunkingPipeline, PipelineConfig, chunk_text
from .text_chunker import TextChunker
from ..config.settings import ChunkingConfig

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ChunkingPipeline",
    "PipelineConfig",
    "chunk_text",
    "Block",
    "BlockStream",
    "segment_text",
    "TextChunker",
    "ChunkingConfig",
]
