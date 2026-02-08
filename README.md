# SEC 10-K RAG Pipeline

Summarize SEC 10-K filings with one function call.

## Usage

```python
from facade import summarize

summary = summarize(
    ticker="AAPL",
    item="1a",
    query="What are the main risk factors?",
    llm_api_key="your-mistral-api-key",
)
print(summary)
```

### With Reranking

```python
summary = summarize(
    ticker="AAPL",
    item="7",
    query="What drove revenue growth?",
    llm_api_key="your-mistral-key",
    reranker_api_key="your-zeroentropy-key",
)
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `ticker` | Yes | Company ticker symbol (e.g., "AAPL") |
| `item` | Yes | Item section to analyze (e.g., "1a", "7") |
| `query` | Yes | Question or topic to summarize |
| `llm_api_key` | Yes | Mistral API key |
| `reranker_api_key` | No | Zero Entropy API key for reranking |
| `top_k` | No | Number of chunks for context (default: 10) |

## Item Sections

| Item | Section Name |
|------|--------------|
| `1` | Business |
| `1a` | Risk Factors |
| `1b` | Unresolved Staff Comments |
| `2` | Properties |
| `3` | Legal Proceedings |
| `4` | Mine Safety Disclosures |
| `5` | Market for Common Equity |
| `6` | Reserved |
| `7` | Management's Discussion and Analysis (MD&A) |
| `7a` | Quantitative and Qualitative Disclosures About Market Risk |
| `8` | Financial Statements |
| `9` | Changes in and Disagreements with Accountants |
| `9a` | Controls and Procedures |
| `9b` | Other Information |
| `10` | Directors and Executive Officers |
| `11` | Executive Compensation |
| `12` | Security Ownership |
| `13` | Certain Relationships |
| `14` | Principal Accountant Fees |
| `15` | Exhibits and Financial Statement Schedules |

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Settings in `config/settings.json`:

```json
{
  "chunking": {"chunk_size_tokens": 512, "chunk_overlap_tokens": 50},
  "llm": {"model": "mistral-small-latest", "temperature": 0.3, "max_tokens": 1024},
  "retrieval": {"embedding_model": "BAAI/bge-large-en-v1.5", "top_k": 10, "rerank_top_k": 20}
}
```

## Architecture

```
summarize(ticker, item, query, llm_api_key, reranker_api_key)
├── FilingDownloader   - Downloads 10-K via datamule
├── ChunkingPipeline   - Splits item section into chunks
├── ChunkStore         - In-memory storage
├── Retriever          - Vector search (sentence-transformers)
├── Reranker           - Optional reranking (Zero Entropy)
└── LLMClient          - Mistral API
```
