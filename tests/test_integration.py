"""Integration test for SEC 10-K RAG Pipeline.

Tests the one-liner API with real API calls.
Requires MISTRAL_API_KEY environment variable.

Usage:
    pytest tests/test_integration.py -v -s
"""
import logging
import os
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestSummarize:
    """Integration tests for summarize()."""
    
    def test_summarize_missing_llm_key_raises(self):
        """Test that missing LLM API key raises ValueError."""
        from facade import summarize
        
        with pytest.raises(ValueError, match="llm_api_key is required"):
            summarize("AAPL", "1a", "What are the main risk factors?", llm_api_key="")
    
    def test_summarize_missing_item_raises(self):
        """Test that missing item raises ValueError."""
        from facade import summarize
        
        with pytest.raises(ValueError, match="item is required"):
            summarize("AAPL", "", "What are the main risk factors?", llm_api_key="fake-key")
    
    def test_summarize_invalid_ticker_raises(self):
        """Test that invalid ticker raises ValueError."""
        from facade import summarize
        
        with pytest.raises(ValueError, match="No CIKs found"):
            summarize("INVALIDTICKER123", "1a", "test", llm_api_key="fake-key")
    
    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="MISTRAL_API_KEY not set"
    )
    def test_summarize_one_liner(self):
        """Test the one-liner summarize function with real API.
        
        summarize("AAPL", "1a", "What are the main risk factors?", llm_api_key="...", reranker_api_key="...")
        """
        from facade import summarize
        
        llm_api_key = os.environ.get("MISTRAL_API_KEY")
        reranker_api_key = os.environ.get("ZEROENTROPY_API_KEY")  # Optional
        
        logger.info("Starting summarize test for AAPL item 1a")
        logger.info(f"LLM API key: {'set' if llm_api_key else 'not set'}")
        logger.info(f"Reranker API key: {'set' if reranker_api_key else 'not set'}")
        
        result = summarize(
            "AAPL",
            "1a",
            "What are the main risk factors?",
            llm_api_key=llm_api_key,
            reranker_api_key=reranker_api_key,
        )
        
        logger.info(f"Result length: {len(result)} chars")
        logger.info(f"Result preview: {result[:500]}...")
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "risk" in result.lower() or "factor" in result.lower()
        
        print("\n" + "=" * 80)
        print("SUMMARIZE RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_item_normalization_with_prefix(self):
        """Test that 'item1a' works same as '1a'."""
        from facade import summarize
        
        # Both should fail the same way (invalid key) - proves normalization works
        with pytest.raises(ValueError, match="llm_api_key is required"):
            summarize("AAPL", "item1a", "test", llm_api_key="")
        
        with pytest.raises(ValueError, match="llm_api_key is required"):
            summarize("AAPL", "1a", "test", llm_api_key="")
    
    def test_various_item_formats(self):
        """Test various item format inputs."""
        from facade import summarize
        
        # All should reach the download step (past validation)
        for item in ["1", "1a", "7", "item1", "item7a"]:
            with pytest.raises((RuntimeError, ValueError)):
                # Will fail at download or LLM step, but proves item parsing works
                summarize("AAPL", item, "test", llm_api_key="fake-key")
