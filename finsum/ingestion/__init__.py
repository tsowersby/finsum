"""SEC Filing Ingestion module.

Downloads and extracts SEC 10-K filings using datamule.
Returns in-memory Filing objects ready for chunking.

Usage:
    from ingestion import FilingDownloader, download_filing
    
    # Quick download
    filing = download_filing("AAPL")
    
    # With config
    downloader = FilingDownloader()
    filing = downloader.download("MSFT")
"""
from .downloader import (
    FilingDownloader,
    DownloaderConfig,
    Filing,
    SectionInfo,
    TableInfo,
    download_filing,
    TENK_ITEMS,
)

__all__ = [
    "FilingDownloader",
    "DownloaderConfig",
    "Filing",
    "SectionInfo",
    "TableInfo",
    "download_filing",
    "TENK_ITEMS",
]
