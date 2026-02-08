# RAG Project TODO

## Completed

- [x] Refactored to simpler OOP design with clean class structure
- [x] Simplified storage layer to in-memory dict + NumPy arrays for vectors
- [x] Created main RAGPipeline facade class for simple interface
- [x] Converted ingestion to OOP FilingDownloader class
- [x] Refactored retriever for in-memory vector search (no PostgreSQL)
- [x] Made reranker API-key agnostic (accepts key as parameter)
- [x] Made LLM client API-key agnostic (accepts key as parameter)
- [x] Broke up monolithic functions into smaller methods
- [x] Fixed AI bullshit code. Did not fully remove shit smell

## Future Improvements

- [ ] Fix AI Bullshit code more
- [ ] Diagram design (not in VSCode)
- [ ] Add local LLM support (ollama, etc)