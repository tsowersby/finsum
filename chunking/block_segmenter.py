"""Block segmentation - identifies typed blocks in text before chunking.

Phase 1 of the chunking pipeline.
Converts raw text into ordered, typed blocks WITHOUT any chunking.

Block types:
- heading: Short lines (≤5 words, <100 chars) with blank lines before/after
- table: Markdown table lines (pipes)
- text: Everything else
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Iterator

BlockType = Literal["heading", "text", "table"]


@dataclass
class Block:
    """A single typed block from the document."""
    type: BlockType
    content: str
    start_line: int
    end_line: int
    
    # Heading-specific
    heading_level: Optional[int] = None
    heading_text: Optional[str] = None
    
    def __post_init__(self):
        self.content = self.content.strip() if self.content else ""
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    def is_empty(self) -> bool:
        return not self.content.strip()


@dataclass
class BlockStream:
    """Ordered collection of blocks from a document."""
    blocks: List[Block] = field(default_factory=list)
    source: str = ""
    
    def add(self, block: Block) -> None:
        if not block.is_empty():
            self.blocks.append(block)
    
    def __iter__(self) -> Iterator[Block]:
        return iter(self.blocks)
    
    def __len__(self) -> int:
        return len(self.blocks)
    
    def tables(self) -> List[Block]:
        """Get all table blocks."""
        return [b for b in self.blocks if b.type == "table"]
    
    def text_blocks(self) -> List[Block]:
        """Get all text blocks."""
        return [b for b in self.blocks if b.type == "text"]
    
    def headings(self) -> List[Block]:
        """Get all heading blocks."""
        return [b for b in self.blocks if b.type == "heading"]


class BlockParser:
    """Parses document text into typed blocks."""
    
    # Constants for heading detection
    MAX_HEADING_CHARS = 100
    MAX_HEADING_WORDS = 5
    
    def parse(self, text: str, source: str = "") -> BlockStream:
        """Parse text into ordered typed blocks."""
        if not text or not text.strip():
            return BlockStream(source=source)
        
        stream = BlockStream(source=source)
        lines = text.split('\n')
        n = len(lines)
        
        i = 0
        while i < n:
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                i += 1
                continue
            
            # Check for table
            if self._is_table_line(stripped):
                block, end_idx = self._parse_table_block(lines, i)
                stream.add(block)
                i = end_idx + 1
                continue
            
            # Check for heading (short line with blank lines around it)
            if self._is_heading_line(lines, i):
                stream.add(Block(
                    type="heading",
                    content=stripped,
                    start_line=i,
                    end_line=i,
                    heading_level=self._estimate_heading_level(stripped),
                    heading_text=stripped,
                ))
                i += 1
                continue
            
            # Default to text block
            block, end_idx = self._parse_text_block(lines, i)
            stream.add(block)
            i = end_idx + 1
        
        return stream
    
    def _is_heading_line(self, lines: List[str], idx: int) -> bool:
        """Check if line is a heading based on context.
        
        A heading is:
        - Short line (<100 chars, ≤5 words)
        - Preceded by at least one blank line (or start of doc)
        - Followed by at least one blank line (or end of doc)
        """
        line = lines[idx].strip()
        
        # Check length constraints
        if len(line) > self.MAX_HEADING_CHARS:
            return False
        
        word_count = len(line.split())
        if word_count > self.MAX_HEADING_WORDS or word_count == 0:
            return False
        
        # Don't treat table lines as headings
        if self._is_table_line(line):
            return False
        
        # Check for blank line before (or start of document)
        has_blank_before = (idx == 0) or (lines[idx - 1].strip() == "")
        
        # Check for blank line after (or end of document)
        has_blank_after = (idx == len(lines) - 1) or (lines[idx + 1].strip() == "")
        
        return has_blank_before and has_blank_after
    
    def _estimate_heading_level(self, text: str) -> int:
        """Estimate heading level based on content patterns."""
        text_upper = text.upper()
        
        # Item headers are top level
        if text_upper.startswith("ITEM "):
            return 1
        
        # Single words or very short = likely subsection
        word_count = len(text.split())
        if word_count == 1:
            return 3
        elif word_count <= 3:
            return 2
        else:
            return 2
    
    def _is_table_line(self, line: str) -> bool:
        """Check if line is part of a Markdown table."""
        stripped = line.strip()
        if not stripped or '|' not in stripped:
            return False
        
        # Split by pipes
        cells = [c.strip() for c in stripped.split('|')]
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        
        # Need at least 2 cells
        if len(cells) < 2:
            return False
        
        # Check if separator row
        sep_pattern = re.compile(r'^:?-{2,}:?$')
        if all(sep_pattern.match(c) for c in cells if c):
            return True
        
        # Data row - at least one cell should have content
        return any(len(c) > 0 for c in cells)
    
    def _parse_table_block(self, lines: List[str], start: int) -> tuple[Block, int]:
        """Parse consecutive table lines."""
        content_lines = []
        i = start
        
        while i < len(lines) and self._is_table_line(lines[i]):
            content_lines.append(lines[i])
            i += 1
        
        end = i - 1 if i > start else start
        
        return Block(
            type="table",
            content='\n'.join(content_lines),
            start_line=start,
            end_line=end,
        ), end
    
    def _parse_text_block(self, lines: List[str], start: int) -> tuple[Block, int]:
        """Parse consecutive text lines until we hit a heading, table, or significant break."""
        content_lines = []
        i = start
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Stop at tables
            if self._is_table_line(stripped):
                break
            
            # Stop at headings
            if stripped and self._is_heading_line(lines, i):
                break
            
            content_lines.append(line)
            i += 1
        
        end = i - 1 if i > start else start
        
        return Block(
            type="text",
            content='\n'.join(content_lines),
            start_line=start,
            end_line=end,
        ), end


def segment_text(text: str, source: str = "") -> BlockStream:
    """Segment text into typed blocks.
    
    This is Phase 1 of the chunking pipeline.
    
    Args:
        text: Raw text content
        source: Source identifier
        
    Returns:
        BlockStream with ordered, typed blocks
    """
    parser = BlockParser()
    return parser.parse(text, source)
