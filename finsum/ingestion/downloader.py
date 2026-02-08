"""SEC Filing Downloader - OOP wrapper for datamule 10-K extraction.

Downloads and extracts SEC 10-K filings for a single ticker using datamule.
Returns structured data ready for chunking pipeline.

Usage:
    downloader = FilingDownloader()
    filing = downloader.download("AAPL")
    
    # Access sections
    print(filing.sections["item1"])  # Business section text
    print(filing.metadata)  # Filing metadata
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# Standard 10-K item definitions
# [] TODO: Make 10-k items either dynamic or turn into json config file.
TENK_ITEMS: Dict[str, str] = {
    "item1": "Business",
    "item1a": "Risk Factors",
    "item1b": "Unresolved Staff Comments",
    "item1c": "Cybersecurity",
    "item2": "Properties",
    "item3": "Legal Proceedings",
    "item4": "Mine Safety Disclosures",
    "item5": "Market for Registrant's Common Equity",
    "item6": "Reserved",
    "item7": "Management's Discussion and Analysis",
    "item7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "item8": "Financial Statements and Supplementary Data",
    "item9": "Changes in and Disagreements with Accountants",
    "item9a": "Controls and Procedures",
    "item9b": "Other Information",
    "item9c": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "item10": "Directors, Executive Officers and Corporate Governance",
    "item11": "Executive Compensation",
    "item12": "Security Ownership of Certain Beneficial Owners",
    "item13": "Certain Relationships and Related Transactions",
    "item14": "Principal Accountant Fees and Services",
}


@dataclass
class SectionInfo:
    """Metadata for a single SEC item section."""
    item: str
    title: str
    content: str
    char_count: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.content)
        if self.word_count == 0:
            self.word_count = len(self.content.split())


@dataclass
class TableInfo:
    """Extracted table data from 10-K."""
    table_number: int
    name: str
    description: str
    data: List[List[str]]
    
    @property
    def row_count(self) -> int:
        return len(self.data)
    
    @property
    def column_count(self) -> int:
        return len(self.data[0]) if self.data else 0


@dataclass
class Filing:
    """Complete SEC 10-K filing with extracted content.
    
    In-memory representation of a downloaded 10-K filing.
    All content is stored as Python objects - no files.
    
    Attributes:
        ticker: Company ticker symbol
        accession: SEC accession number
        filing_date: Date of filing
        sec_url: URL to SEC viewer
        sections: Dict mapping item key to SectionInfo
        tables: List of extracted tables
    """
    ticker: str
    accession: str
    filing_date: str
    sec_url: str = ""
    sections: Dict[str, SectionInfo] = field(default_factory=dict)
    tables: List[TableInfo] = field(default_factory=list)
    
    @property
    def available_items(self) -> List[str]:
        """List of item keys with extracted content."""
        return list(self.sections.keys())
    
    def get_section_text(self, item: str) -> Optional[str]:
        """Get text content for a section by item key."""
        section = self.sections.get(item.lower())
        return section.content if section else None
    
    def get_all_text(self, items: Optional[List[str]] = None) -> str:
        """Get combined text from specified or all sections."""
        target_items = items or self.available_items
        parts = []
        for item in target_items:
            section = self.sections.get(item.lower())
            if section:
                parts.append(f"## {section.item.upper()}: {section.title}\n\n{section.content}")
        return "\n\n".join(parts)
    
    @property
    def total_chars(self) -> int:
        return sum(s.char_count for s in self.sections.values())
    
    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "accession": self.accession,
            "filing_date": self.filing_date,
            "sec_url": self.sec_url,
            "sections": {
                k: {
                    "item": v.item,
                    "title": v.title,
                    "char_count": v.char_count,
                    "word_count": v.word_count,
                }
                for k, v in self.sections.items()
            },
            "available_items": self.available_items,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
        }


@dataclass
class DownloaderConfig:
    """Configuration for filing downloader."""
    lookback_days: int = 455  # ~1.25 years to catch latest filing
    item_filter: Optional[List[str]] = None  # Specific items to extract, or None for all
    extract_tables: bool = False  # Whether to extract tables (slower)
    
    def __post_init__(self):
        if self.item_filter:
            self.item_filter = [i.lower() for i in self.item_filter]


class FilingDownloader:
    """Downloads and extracts SEC 10-K filings for a single ticker.
    
    Uses datamule for SEC data access. All data is returned in-memory
    as Filing objects - no files are persisted.
    
    Example:
        downloader = FilingDownloader()
        filing = downloader.download("AAPL")
        
        # Get specific section
        mda = filing.get_section_text("item7")
        
        # Get all available items
        print(filing.available_items)
    """
    
    def __init__(self, config: Optional[DownloaderConfig] = None):
        """Initialize downloader.
        
        Args:
            config: Optional configuration for download behavior
        """
        self.config = config or DownloaderConfig()
        self._items = TENK_ITEMS
    
    def download(self, ticker: str) -> Filing:
        """Download and extract the latest 10-K for a ticker.
        
        Args:
            ticker: Company ticker symbol (e.g., "AAPL")
            
        Returns:
            Filing object with extracted sections
            
        Raises:
            ValueError: If no 10-K found for ticker
            RuntimeError: If download or extraction fails
        """
        ticker = ticker.upper()
        logger.info(f"Downloading 10-K for {ticker}")
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        # Use temp directory for datamule cache
        cache_dir = Path(tempfile.mkdtemp(prefix=f"datamule_{ticker}_"))
        
        try:
            filing = self._download_and_extract(
                ticker, 
                cache_dir, 
                start_date, 
                end_date
            )
            return filing
        finally:
            # Clean up cache directory
            self._cleanup(cache_dir)
    
    def _download_and_extract(
        self,
        ticker: str,
        cache_dir: Path,
        start_date: date,
        end_date: date,
    ) -> Filing:
        """Core download and extraction logic.
        
        # TODO: This function is complex due to datamule API requirements.
        # Structural refactoring would require changes to the datamule
        # interaction pattern. Leaving as-is per constraints.
        """
        from datamule import Portfolio
        
        portfolio = Portfolio(str(cache_dir))
        
        # Download 10-K documents
        portfolio.download_submissions(
            submission_type='10-K',
            document_type='10-K',
            ticker=ticker,
            filing_date=(start_date.isoformat(), end_date.isoformat()),
            quiet=True,
        )
        
        # Get submissions and sort by date (most recent first)
        submissions = list(portfolio)
        if not submissions:
            raise ValueError(f"No 10-K found for {ticker} in the last {self.config.lookback_days} days")
        
        submissions.sort(key=lambda x: x.filing_date, reverse=True)
        submission = submissions[0]
        
        logger.info(f"{ticker}: Processing filing from {submission.filing_date}")
        
        # Initialize filing object
        filing = Filing(
            ticker=ticker,
            accession=submission.accession,
            filing_date=str(submission.filing_date),
            sec_url=f"https://www.sec.gov/cgi-bin/viewer?action=view&accession_number={submission.accession}",
        )
        
        # Extract sections from 10-K document
        for tenk in submission.document_type('10-K'):
            tenk.parse()
            self._extract_sections(tenk, filing)
            
            if self.config.extract_tables:
                self._extract_tables(tenk, filing)
            
            break  # Only process first 10-K document
        
        if not filing.sections:
            raise RuntimeError(f"No sections extracted from {ticker} 10-K")
        
        logger.info(f"{ticker}: Extracted {len(filing.sections)} sections, {filing.total_words} words")
        return filing
    
    def _extract_sections(self, tenk, filing: Filing) -> None:
        """Extract all item sections from parsed 10-K."""
        items_to_extract = self.config.item_filter or list(self._items.keys())
        
        for item_key in items_to_extract:
            item_key = item_key.lower()
            if item_key not in self._items:
                continue
            
            try:
                sections = tenk.get_section(item_key, format='text')
                
                if sections and len(sections) > 0 and sections[0].strip():
                    content = "\n\n".join(sections)
                    
                    filing.sections[item_key] = SectionInfo(
                        item=item_key,
                        title=self._items[item_key],
                        content=content,
                    )
                    
            except Exception as e:
                logger.debug(f"Could not extract {item_key}: {e}")
                continue
    
    def _extract_tables(self, tenk, filing: Filing) -> None:
        """Extract tables from parsed 10-K."""
        try:
            tables = tenk.tables
            if not tables:
                return
            
            for idx, table in enumerate(tables):
                name = str(table.name) if hasattr(table, 'name') and table.name else f"Table {idx+1}"
                desc = str(table.description) if hasattr(table, 'description') and table.description else ""
                
                raw_data = table.data if hasattr(table, 'data') and table.data else []
                rows = []
                for row in raw_data:
                    if isinstance(row, (list, tuple)):
                        rows.append([str(cell) if cell is not None else "" for cell in row])
                    else:
                        rows.append([str(row)])
                
                filing.tables.append(TableInfo(
                    table_number=idx + 1,
                    name=name,
                    description=desc,
                    data=rows,
                ))
                
        except Exception as e:
            logger.debug(f"Could not extract tables: {e}")
    
    def _cleanup(self, cache_dir: Path) -> None:
        """Clean up temporary cache directory."""
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass


def download_filing(ticker: str, **kwargs) -> Filing:
    """Quick function to download a 10-K filing.
    
    Args:
        ticker: Company ticker symbol
        **kwargs: Arguments passed to DownloaderConfig
        
    Returns:
        Filing object with extracted sections
    """
    config = DownloaderConfig(**kwargs) if kwargs else None
    downloader = FilingDownloader(config)
    return downloader.download(ticker)
