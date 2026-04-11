"""
Canonical data contracts for the Financial QA data pipeline.

FilingMetadata is the single source of truth for filing provenance.
It flows from sec_loader → extractor → chunker → vector_store / sql_store.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class FilingMetadata:
    """
    Provenance record for one SEC filing document.
    Inferred entirely from the local directory structure — no network calls.

    Directory layout expected:
        {filings_dir}/{TICKER}/{FORM}/{DATE}_{ACCESSION}/{primary_doc}
    """
    ticker:           str    # "AAPL"
    company:          str    # "Apple Inc." (from company_map.json or ticker fallback)
    filing_type:      str    # "10-K" | "10-Q" | "8-K"
    date:             str    # "YYYY-MM-DD" (filing date from directory name)
    accession_number: str    # "0000320193-23-000106"
    source_path:      str    # absolute path to the primary document file
    section:          str = ""  # filled in per-chunk: "ITEM 7", "ITEM 1A", etc.

    def flat_dict(self) -> dict:
        """Return as flat dict for ChromaDB metadata (no nested objects)."""
        return {
            "ticker":           self.ticker,
            "company":          self.company,
            "filing_type":      self.filing_type,
            "date":             self.date,
            "accession_number": self.accession_number,
            "source_path":      self.source_path,
            "section":          self.section,
        }


@dataclass
class TextChunk:
    """
    One chunk of text from a filing, ready for embedding and VectorDB storage.
    chunk_id is globally unique across all chunks in the store.
    """
    text:         str
    ticker:       str
    doc_type:     str    # "10-K" | "10-Q" | "8-K"
    filing_date:  str    # "YYYY-MM-DD"
    source_path:  str
    chunk_index:  int
    section:      str = ""
    metadata:     dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}__{self.doc_type}__{self.filing_date}"

    @property
    def chunk_id(self) -> str:
        url_hash = hashlib.md5(self.source_path.encode()).hexdigest()[:8]
        return f"{self.doc_id}__{url_hash}__{self.chunk_index:05d}"


@dataclass
class FinancialRow:
    """
    One normalized financial data point extracted from a filing table.
    Stored in SQLite financials table.
    """
    ticker:    str
    period:    str          # "2023-09" (monthly) | "FY2023" (annual)
    statement: str          # "income_statement" | "balance_sheet" | "cash_flow" | "unknown"
    metric:    str          # "Total Revenue", "Net Income", etc. (cleaned label)
    value:     float | None
    unit:      str = "USD"
    raw_value: str = ""
