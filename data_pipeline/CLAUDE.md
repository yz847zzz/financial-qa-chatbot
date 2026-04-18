# Data Pipeline — Part 1 CLAUDE.md

## Purpose

Load **locally downloaded** SEC EDGAR filings (10-K, 10-Q, 8-K), extract text
and financial tables, chunk text for RAG, and persist into:
- **ChromaDB** at `../data/vectordb/` (semantic search)
- **SQLite** at `../data/financials.db` (structured queries)

This part has **NO dependency on Parts 2 or 3**.
Do NOT implement any network download — all files are pre-downloaded to disk.

## Commands

```bash
pip install -r requirements.txt
pip install -e .

# Smoke test individual stores
python -m data_pipeline.storage.vector_store
python -m data_pipeline.storage.sql_store

# Run full ingest
python scripts/run_ingest.py \
  --ticker AAPL MSFT GOOGL \
  --filings-dir ../data/filings \
  --db-path ../data/financials.db \
  --vectordb-dir ../data/vectordb \
  --start 2020-01-01 \
  --end 2024-12-31

# Validate results
python scripts/validate_store.py

# Tests
pytest tests/ -v
```

## Input File Layout

```
data/filings/
└── {TICKER}/
    └── {FORM}/           # "10-K", "10-Q", "8-K"
        └── {DATE}_{ACCESSION}/
            └── primary_doc.htm   (or .pdf)
```

Example: `data/filings/AAPL/10-K/2023-11-03_0000320193-23-000106/primary_doc.htm`

The `sec_loader.py` walks this tree and yields `FilingMetadata` objects.
It must handle both `.htm` and `.pdf` files.

## Data Contracts — Critical

### FilingMetadata (`metadata/models.py`)

```python
@dataclass
class FilingMetadata:
    company: str           # "Apple Inc."
    ticker: str            # "AAPL"
    filing_type: str       # "10-K" | "10-Q" | "8-K"
    date: str              # "YYYY-MM-DD"
    section: str           # "ITEM 7" | "ITEM 1A" | "" for 8-K/unsectioned
    accession_number: str  # "0000320193-23-000106"
    source_path: str       # absolute path to original file
```

### TextChunk (`processing/chunker.py`)

```python
@dataclass
class TextChunk:
    text: str
    ticker: str
    doc_type: str          # same as filing_type
    filing_date: str
    source_path: str
    chunk_index: int
    section: str
    metadata: dict         # flat dict with ALL FilingMetadata fields

    @property
    def chunk_id(self) -> str:
        # "{ticker}__{doc_type}__{date}__{hash8}__{index:05d}"
        url_hash = hashlib.md5(self.source_path.encode()).hexdigest()[:8]
        return f"{self.ticker}__{self.doc_type}__{self.filing_date}__{url_hash}__{self.chunk_index:05d}"
```

The `metadata` dict stored in ChromaDB must include every FilingMetadata field
as a flat key (no nested dicts — ChromaDB limitation).

### SQLite Schema (`storage/schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS financials (
    ticker       TEXT NOT NULL,
    period       TEXT NOT NULL,   -- "YYYY-MM" or "FYYYY"
    statement    TEXT NOT NULL,   -- "income_statement" | "balance_sheet" | "cash_flow"
    metric       TEXT NOT NULL,   -- "Total Revenue", "Net Income", etc.
    value        REAL,
    unit         TEXT DEFAULT 'USD',
    raw_value    TEXT,
    UNIQUE(ticker, period, statement, metric) ON CONFLICT REPLACE
);

CREATE TABLE IF NOT EXISTS filing_metadata (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL,
    company          TEXT,
    filing_type      TEXT,         -- "10-K" | "10-Q" | "8-K"
    date             TEXT,         -- "YYYY-MM-DD"
    accession_number TEXT UNIQUE,
    section_count    INTEGER DEFAULT 0,
    chunk_count      INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_financials_ticker ON financials(ticker);
CREATE INDEX IF NOT EXISTS idx_financials_metric ON financials(metric);
CREATE INDEX IF NOT EXISTS idx_filing_metadata_ticker ON filing_metadata(ticker);
```

## Module Responsibilities

### `ingestion/sec_loader.py`

```python
def walk_filings(filings_dir: str, tickers: list[str] | None = None,
                 start: str | None = None, end: str | None = None
                 ) -> Iterator[tuple[FilingMetadata, Path]]:
    """Walk the filings directory tree, yield (metadata, file_path) pairs.
    Filter by tickers list if provided. Filter by date range if provided."""
```

- Infer `ticker`, `filing_type`, `date`, `accession_number` from directory structure
- Company name: read from a `company_map.json` sidecar if present, else use ticker
- Skip hidden files, non-htm/pdf files, and malformed paths with `logger.warning`

### `ingestion/filing_index.py` (optional)

```python
def load_index(index_path: str) -> list[FilingMetadata]:
    """Load a JSON manifest of pre-downloaded filings.
    JSON format: [{"ticker": ..., "filing_type": ..., "date": ..., "path": ...}]"""
```

### `processing/extractor.py`

```python
class HtmlExtractor:
    def extract(self, path: Path) -> str:
        """Extract plain text from .htm/.html SEC filing using BeautifulSoup.
        Remove <script>, <style>, boilerplate nav. Preserve table text inline."""

class PdfExtractor:
    def extract(self, path: Path) -> str:
        """Extract plain text from .pdf using pdfplumber.
        Concatenate pages with double newline."""

def get_extractor(path: Path) -> HtmlExtractor | PdfExtractor:
    """Return the appropriate extractor based on file suffix."""
```

### `processing/section_splitter.py`

```python
def split_into_sections(text: str, filing_type: str) -> dict[str, str]:
    """Split 10-K/10-Q text into named ITEM sections using regex.
    Returns {"ITEM 1": "...", "ITEM 1A": "...", "ITEM 7": "...", ...}
    For 8-K or unsectioned: returns {"": full_text}"""
```

Regex pattern for ITEM headings:
`r"(?i)(?:^|\n)\s*ITEM\s+(\d+[A-Z]?)\s*[.\-–]?\s*([^\n]{0,80})"`

### `processing/chunker.py`

```python
def chunk_text(text: str, metadata: FilingMetadata,
               chunk_size: int = 512, overlap: int = 64,
               section: str = "") -> list[TextChunk]:
    """Sliding-window chunker. Split on sentence boundaries when possible.
    Each chunk carries full metadata for ChromaDB storage."""

def chunk_filing(metadata: FilingMetadata, file_path: Path) -> list[TextChunk]:
    """End-to-end: extract → split sections → chunk each section.
    Returns flat list of TextChunk across all sections."""
```

Chunking parameters:
- `chunk_size = 512` characters
- `overlap = 64` characters
- Section-aware for 10-K/10-Q (preserves section metadata per chunk)
- Flat chunking for 8-K and unsectioned content

### `processing/table_parser.py`

```python
@dataclass
class FinancialRow:
    ticker: str
    period: str      # "YYYY-MM" or "FYYYY"
    statement: str
    metric: str
    value: float | None
    unit: str
    raw_value: str

def parse_financial_tables(text: str, ticker: str,
                            filing_date: str) -> list[FinancialRow]:
    """Extract financial statement tables from filing text.
    Recognize income statement, balance sheet, cash flow keywords.
    Returns FinancialRow objects ready for SQLite upsert."""
```

### `storage/vector_store.py`

ChromaDB-backed vector store with extended metadata fields for financial filing provenance.

```python
class VectorStore:
    def __init__(self, persist_dir: str,
                 collection_name: str = "financial_docs",
                 embedding_model: str = "all-MiniLM-L6-v2"): ...

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """Batch embed and add chunks. Skip duplicates by chunk_id.
        Returns count of newly added chunks."""

    def search(self, query: str, n_results: int = 10,
               filters: dict | None = None) -> list[dict]:
        """Semantic search. filters: {"ticker": "AAPL", "filing_type": "10-K"}
        Returns list of {"text": ..., "metadata": ..., "distance": ...}"""

    def search_multi_query(self, queries: list[str], n_results: int = 5,
                           filters: dict | None = None) -> list[dict]:
        """Run multiple queries, deduplicate by chunk_id, return merged results."""
```

### `storage/sql_store.py`

SQLite store implementing the two-table schema (financials + filing_metadata).

```python
class SQLStore:
    def __init__(self, db_path: str): ...

    def upsert_financials(self, rows: list[FinancialRow]) -> int: ...

    def upsert_filing_metadata(self, metadata: FilingMetadata,
                                section_count: int, chunk_count: int) -> None: ...

    def get_pivot(self, ticker: str, statement: str,
                  metrics: list[str] | None = None) -> pd.DataFrame: ...

    def execute_raw(self, sql: str) -> pd.DataFrame:
        """Execute a raw SELECT statement. Raises ValueError for non-SELECT.
        Used by the NL2SQL module in deployment."""
```

### `scripts/run_ingest.py`

CLI entry point. Example:

```bash
python scripts/run_ingest.py \
  --ticker AAPL MSFT \
  --filings-dir ../data/filings \
  --db-path ../data/financials.db \
  --vectordb-dir ../data/vectordb
```

Must print a summary on completion:

```
Ingest complete:
  Filings processed : 42
  Filings skipped   : 3
  SQL rows upserted : 8540
  Vector chunks added: 12300
```

### `scripts/validate_store.py`

Smoke test — run after ingest:
1. Count rows in each SQLite table
2. Sample `get_pivot("AAPL", "income_statement")`
3. Search VectorDB for "revenue growth" and print top 3 results
4. Print pass/fail for each check

## Dependencies (`requirements.txt`)

```
beautifulsoup4>=4.12
pdfplumber>=0.10
chromadb>=0.5
sentence-transformers>=3.0
loguru>=0.7
pandas>=2.0
numpy>=1.26
pytest>=8.0
```

## Testing Requirements

All tests in `tests/` must pass with `pytest tests/ -v` and require:
- **No network access**
- **No GPU**
- Fixture files in `tests/fixtures/`: a small `sample.htm` and `sample.pdf`

| Test file | What to verify |
|---|---|
| `test_extractor.py` | HtmlExtractor and PdfExtractor return non-empty strings from fixture files |
| `test_chunker.py` | chunk_id uniqueness across 100 chunks; overlap chars preserved; section in metadata |
| `test_vector_store.py` | Add 10 chunks → search → top result matches; metadata filter works |
| `test_sql_store.py` | Upsert rows → query → get_pivot returns DataFrame; execute_raw SELECT works; INSERT rejected |

## Error Handling

- Skip unreadable files with `logger.warning(f"Skipping {path}: {e}")`
- Log `(ticker, filing_type, date, reason)` for every skipped file
- `run_ingest.py` collects all skip reasons and prints at the end
- Never raise from inside the ingest loop — catch and continue
