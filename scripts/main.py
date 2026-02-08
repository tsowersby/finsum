# scripts/main.py
"""Demo script for the finsum SEC 10-K RAG pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path so we can import finsum
sys.path.insert(0, str(Path(__file__).parent.parent))

from finsum.facade import summarize


def get_input(prompt: str, required: bool = True) -> Optional[str]:
    """Prompt user for input."""
    value = input(prompt).strip()
    if required and not value:
        print("This field is required.")
        sys.exit(1)
    return value or None


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SEC 10-K filings using the finsum RAG pipeline."
    )
    parser.add_argument("--ticker", help="Company ticker (e.g., AAPL)")
    parser.add_argument("--item", help="Item section (e.g., 1a, 7)")
    parser.add_argument("--query", help="Question to answer")
    parser.add_argument("--llm-api-key", help="Mistral API key")
    parser.add_argument("--reranker-api-key", help="Optional Zero Entropy API key")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of chunks for context (default: 10)"
    )

    args = parser.parse_args()

    # Use CLI args or prompt for input
    ticker = args.ticker or get_input("Ticker (e.g., AAPL): ")
    item = args.item or get_input("Item section (e.g., 1a, 7): ")
    query = args.query or get_input("Query: ")
    llm_api_key = args.llm_api_key or get_input("Mistral API key: ")
    reranker_api_key = args.reranker_api_key
    if reranker_api_key is None:
        reranker_api_key = get_input("Zero Entropy API key (optional, press Enter to skip): ", required=False)
    top_k = args.top_k

    print(f"\nSummarizing {ticker} 10-K Item {item}...")
    print(f"Query: {query}\n")

    summary = summarize(
        ticker=ticker,
        item=item,
        query=query,
        llm_api_key=llm_api_key,
        reranker_api_key=reranker_api_key,
        top_k=top_k,
    )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()