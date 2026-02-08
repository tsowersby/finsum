"""Chunking module - document segmentation.

Components:
- ChunkingPipeline: Structure-preserving chunking with section paths
- TextChunker: Paragraph/sentence splitting
- BlockSegmenter: Identifies typed blocks (heading, text, table)
"""
from chunking.datatypes import Chunk, ChunkMetadata
from chunking.block_segmenter import Block, BlockStream, segment_text
from chunking.pipeline import ChunkingPipeline, PipelineConfig, chunk_text
from chunking.text_chunker import TextChunker
from config.settings import ChunkingConfig

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
