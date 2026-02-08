"""LLM Client for generating summaries.

Simple wrapper around Mistral Chat API.

Usage:
    llm = LLMClient(api_key="your-key")
    response = llm.generate("What are the risk factors?", context_text)
"""
from __future__ import annotations

import requests

from config import get_llm_config

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

SYSTEM_PROMPT = """You are a document summarization assistant for SEC 10-K filings.

Guidelines:
- Summarize using ONLY information from the provided context
- Write in clear, professional prose
- If information is not in the context, state: "This information was not found in the provided sections."
- Do not infer, calculate, or make predictions beyond the text
- Do not provide financial advice
"""


class LLMClient:
    """Simple Mistral LLM client.
    
    Example:
        llm = LLMClient(api_key="your-mistral-key")
        answer = llm.generate("What are the main risks?", context)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        
        cfg = get_llm_config()
        model = model or cfg.model
        temperature = temperature if temperature is not None else cfg.temperature
        max_tokens = max_tokens or cfg.max_tokens
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
    
    def generate(self, query: str, context: str) -> str:
        """Generate a response using the provided context.
        
        Args:
            query: User's question
            context: Retrieved document content
            
        Returns:
            Generated response text
            
        Raises:
            RuntimeError: If API call fails
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}",
            },
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        response = self._session.post(MISTRAL_ENDPOINT, json=payload)
        
        if not response.ok:
            raise RuntimeError(f"LLM API error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
