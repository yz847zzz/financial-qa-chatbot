"""
Walk the local SEC filings directory and yield FilingMetadata objects.

Expected layout (produced by sec_downloader.py):
    {filings_dir}/{TICKER}/{FORM}/{DATE}_{ACCESSION}/{primary_doc.htm|.pdf}

No network calls — reads only from local disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from loguru import logger

from ..metadata.models import FilingMetadata

# Form types we process
_SUPPORTED_FORMS = {"10-K", "10-Q", "8-K"}
_SUPPORTED_EXTENSIONS = {".htm", ".html", ".pdf"}

# Optional sidecar file mapping ticker → company name
_COMPANY_MAP_FILE = "company_map.json"


def _load_company_map(filings_dir: Path) -> dict[str, str]:
    """Load company_map.json if present, else return empty dict."""
    cmap_path = filings_dir / _COMPANY_MAP_FILE
    if cmap_path.exists():
        try:
            return json.loads(cmap_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load company_map.json: {e}")
    return {}


def walk_filings(
    filings_dir: str | Path,
    tickers: list[str] | None = None,
    form_types: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[tuple[FilingMetadata, Path]]:
    """
    Walk the local filings directory tree and yield (FilingMetadata, file_path) pairs.

    Skips:
      - Tickers not in the tickers list (if provided)
      - Form types not in form_types list (if provided)
      - Filing dates outside [start_date, end_date] (if provided)
      - Files with unsupported extensions
      - Non-primary documents (picks the largest .htm or first .pdf per folder)

    Args:
        filings_dir: root directory of downloaded filings
        tickers:     filter to these tickers only (None = all)
        form_types:  filter to these form types only (None = all supported)
        start_date:  "YYYY-MM-DD" lower bound on filing date
        end_date:    "YYYY-MM-DD" upper bound on filing date

    Yields:
        (FilingMetadata, Path) — one per filing document
    """
    filings_dir = Path(filings_dir)
    if not filings_dir.is_dir():
        logger.error(f"Filings directory not found: {filings_dir}")
        return

    form_types = form_types or list(_SUPPORTED_FORMS)
    form_set = set(form_types)
    ticker_set = {t.upper() for t in tickers} if tickers else None
    company_map = _load_company_map(filings_dir)

    for ticker_dir in sorted(filings_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        if ticker in ("NEWS",):   # skip the news/ subdirectory
            continue
        if ticker_set and ticker not in ticker_set:
            continue

        for form_dir in sorted(ticker_dir.iterdir()):
            if not form_dir.is_dir():
                continue
            form = form_dir.name
            if form not in form_set:
                continue

            for filing_dir in sorted(form_dir.iterdir()):
                if not filing_dir.is_dir():
                    continue

                # Directory name: {DATE}_{ACCESSION} or just {ACCESSION}
                dir_name = filing_dir.name
                date, accession = _parse_dir_name(dir_name)
                if date is None:
                    logger.warning(f"Cannot parse date from directory: {filing_dir}")
                    continue

                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue

                primary = _pick_primary_file(filing_dir)
                if primary is None:
                    logger.warning(f"No usable file in {filing_dir}")
                    continue

                company = company_map.get(ticker, ticker)

                yield (
                    FilingMetadata(
                        ticker=ticker,
                        company=company,
                        filing_type=form,
                        date=date,
                        accession_number=accession,
                        source_path=str(primary.resolve()),
                    ),
                    primary,
                )


def _parse_dir_name(dir_name: str) -> tuple[str | None, str]:
    """
    Parse {DATE}_{ACCESSION} directory name.
    Returns (date_str, accession_str).
    date_str is None if parsing fails.

    Examples:
      "2023-11-03_000032019323000106" → ("2023-11-03", "000032019323000106")
      "000032019323000106"            → (None,          "000032019323000106")
    """
    if "_" in dir_name:
        parts = dir_name.split("_", 1)
        if len(parts[0]) == 10 and parts[0][4] == "-" and parts[0][7] == "-":
            return parts[0], parts[1]
    return None, dir_name


def _pick_primary_file(filing_dir: Path) -> Path | None:
    """
    Select the primary document from a filing directory.

    Strategy:
      1. Largest .htm / .html file (SEC primary docs are usually large HTML)
      2. First .pdf file
      3. None if nothing found
    """
    htm_files = sorted(
        [f for f in filing_dir.iterdir() if f.suffix.lower() in (".htm", ".html")],
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    if htm_files:
        return htm_files[0]

    pdf_files = [f for f in filing_dir.iterdir() if f.suffix.lower() == ".pdf"]
    if pdf_files:
        return pdf_files[0]

    return None


def count_filings(
    filings_dir: str | Path,
    tickers: list[str] | None = None,
    form_types: list[str] | None = None,
) -> dict[str, int]:
    """
    Quick count of available filings without loading content.
    Returns {form_type: count}.
    """
    counts: dict[str, int] = {}
    for meta, _ in walk_filings(filings_dir, tickers=tickers, form_types=form_types):
        counts[meta.filing_type] = counts.get(meta.filing_type, 0) + 1
    return counts


if __name__ == "__main__":
    import sys
    filings_dir = sys.argv[1] if len(sys.argv) > 1 else "data/filings"
    counts = count_filings(filings_dir)
    print(f"Available filings in {filings_dir}:")
    for form, n in sorted(counts.items()):
        print(f"  {form}: {n}")
