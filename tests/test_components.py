"""Tests for individual components."""
import numpy as np
import pytest

from storage.memory import ChunkStore, StoredChunk
from chunking.datatypes import Chunk, ChunkMetadata
from retrieval.retriever import Retriever, RetrievedChunk


class TestChunkStore:
    """Tests for ChunkStore."""
    
    def test_add_and_get(self):
        """Test adding and retrieving a chunk."""
        store = ChunkStore(vector_dim=4)
        
        chunk = Chunk(
            chunk_id="test_1",
            content="Test content",
            metadata=ChunkMetadata(
                source="test",
                section_path="item1a",
                content_type="prose",
                company="TEST",
            ),
        )
        embedding = np.array([1.0, 0.0, 0.0, 0.0])
        
        assert store.add(chunk, embedding) is True
        assert store.count() == 1
        
        retrieved = store.get("test_1")
        assert retrieved is not None
        assert retrieved.content == "Test content"
    
    def test_add_duplicate_returns_false(self):
        """Test that adding duplicate chunk returns False."""
        store = ChunkStore(vector_dim=4)
        
        chunk = Chunk(
            chunk_id="test_1",
            content="Test content",
            metadata=ChunkMetadata(
                source="test",
                section_path="item1a",
                content_type="prose",
                company="TEST",
            ),
        )
        embedding = np.array([1.0, 0.0, 0.0, 0.0])
        
        assert store.add(chunk, embedding) is True
        assert store.add(chunk, embedding) is False
        assert store.count() == 1
    
    def test_get_by_section(self):
        """Test filtering chunks by section."""
        store = ChunkStore(vector_dim=4)
        
        for i, section in enumerate(["item1a", "item1a", "item7"]):
            chunk = Chunk(
                chunk_id=f"test_{i}",
                content=f"Content {i}",
                metadata=ChunkMetadata(
                    source="test",
                    section_path=section,
                    content_type="prose",
                    company="TEST",
                ),
            )
            store.add(chunk, np.random.rand(4))
        
        item1a_chunks = store.get_by_section("item1a")
        assert len(item1a_chunks) == 2
        
        item7_chunks = store.get_by_section("item7")
        assert len(item7_chunks) == 1
    
    def test_get_vectors(self):
        """Test getting vector matrix."""
        store = ChunkStore(vector_dim=4)
        
        for i in range(3):
            chunk = Chunk(
                chunk_id=f"test_{i}",
                content=f"Content {i}",
                metadata=ChunkMetadata(
                    source="test",
                    section_path="item1a",
                    content_type="prose",
                    company="TEST",
                ),
            )
            store.add(chunk, np.random.rand(4))
        
        vectors = store.get_vectors()
        assert vectors.shape == (3, 4)
        
        chunk_ids = store.get_chunk_ids()
        assert len(chunk_ids) == 3
    
    def test_clear(self):
        """Test clearing store."""
        store = ChunkStore(vector_dim=4)
        
        chunk = Chunk(
            chunk_id="test_1",
            content="Test content",
            metadata=ChunkMetadata(
                source="test",
                section_path="item1a",
                content_type="prose",
                company="TEST",
            ),
        )
        store.add(chunk, np.random.rand(4))
        
        assert store.count() == 1
        store.clear()
        assert store.count() == 0


class TestRetriever:
    """Tests for Retriever."""
    
    def test_search_empty_store(self):
        """Test searching empty store returns empty list."""
        store = ChunkStore(vector_dim=384)
        retriever = Retriever(store)
        
        results = retriever.search("test query")
        assert results == []
    
    def test_get_by_section(self):
        """Test direct section lookup."""
        store = ChunkStore(vector_dim=4)
        
        chunk = Chunk(
            chunk_id="test_1",
            content="Test content",
            metadata=ChunkMetadata(
                source="test",
                section_path="item1a",
                content_type="prose",
                company="TEST",
            ),
        )
        store.add(chunk, np.array([1.0, 0.0, 0.0, 0.0]))
        
        retriever = Retriever(store)
        chunks = retriever.get_by_section("item1a")
        
        assert len(chunks) == 1
        assert chunks[0].content == "Test content"


class TestReranker:
    """Tests for Reranker."""
    
    def test_reranker_with_custom_fn(self):
        """Test reranker with custom function."""
        from retrieval.reranker import Reranker
        
        # Simple rerank function that reverses order
        def reverse_rerank(query: str, docs: list) -> list:
            return [(i, 1.0 - i * 0.1) for i in reversed(range(len(docs)))]
        
        reranker = Reranker(rerank_fn=reverse_rerank)
        
        # Create mock results
        results = []
        for i in range(3):
            chunk = Chunk(
                chunk_id=f"test_{i}",
                content=f"Content {i}",
                metadata=ChunkMetadata(
                    source="test",
                    section_path="item1a",
                    content_type="prose",
                    company="TEST",
                ),
            )
            results.append(RetrievedChunk(chunk=chunk, score=0.5))
        
        reranked = reranker.rerank("test query", results)
        
        # Should be reversed
        assert len(reranked) == 3
        assert reranked[0].chunk.chunk_id == "test_2"
    
    def test_reranker_invalid_fn_raises(self):
        """Test that non-callable raises ValueError."""
        from retrieval.reranker import Reranker
        
        with pytest.raises(ValueError):
            Reranker(rerank_fn="not a function")
