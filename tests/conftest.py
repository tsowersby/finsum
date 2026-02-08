"""Pytest configuration and fixtures."""
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def llm_api_key():
    """Get LLM API key from environment."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY not set")
    return key


@pytest.fixture
def reranker_api_key():
    """Get reranker API key from environment (optional)."""
    return os.environ.get("ZEROENTROPY_API_KEY")
