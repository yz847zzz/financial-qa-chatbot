"""
Download SEC EDGAR filings (10-K, 10-Q, 8-K) for a list of tickers.

*** DO NOT EXECUTE IN THE CHATBOT PIPELINE — for initial data acquisition only ***

Filings are already downloaded to:
    data/filings/

Run this script only if you need to add more tickers or refresh filings.

Uses the SEC EDGAR public REST API — no authentication required.
Rate-limited to 10 req/s per SEC policy (0.11s sleep between calls).

Usage:
    python -m data_pipeline.ingestion.sec_downloader \
        --ticker AAPL MSFT NVDA \
        --forms 10-K 10-Q \
        --start 2019-01-01 \
        --end 2024-12-31 \
        --out data/filings

"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from loguru import logger

_HEADERS = {"User-Agent": "finqa-chatbot contact@finqa-research.com"}
_SEC_BASE = "https://www.sec.gov"
_EDGAR_BASE = "https://data.sec.gov"
_TICKER_JSON = f"{_SEC_BASE}/files/company_tickers.json"

_ticker_to_cik: dict[str, str] = {}


def _load_ticker_map() -> None:
    global _ticker_to_cik
    if _ticker_to_cik:
        return
    resp = requests.get(_TICKER_JSON, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    _ticker_to_cik = {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in data.values()
    }
    logger.info(f"Loaded {len(_ticker_to_cik)} ticker→CIK mappings")


def get_cik(ticker: str) -> str:
    _load_ticker_map()
    ticker = ticker.upper()
    if ticker not in _ticker_to_cik:
        raise ValueError(f"CIK not found for ticker '{ticker}'")
    return _ticker_to_cik[ticker]


def list_filings(
    ticker: str,
    form_types: list[str],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Return filing metadata for a ticker filtered by form type and date range.
    Each dict: ticker, cik, form, date, accession_number, primary_document
    """
    cik = get_cik(ticker)
    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(0.11)

    meta = resp.json()
    recent = meta.get("filings", {}).get("recent", {})

    results = []
    for form, date, acc, primary in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if form in form_types and start_date <= date <= end_date:
            results.append({
                "ticker": ticker, "cik": cik,
                "form": form, "date": date,
                "accession_number": acc,
                "primary_document": primary,
            })

    # Also check older filings (>10 filings returned as separate pages)
    for older in meta.get("filings", {}).get("files", []):
        page_url = f"{_EDGAR_BASE}/submissions/{older['name']}"
        try:
            page_resp = requests.get(page_url, headers=_HEADERS, timeout=15)
            page_resp.raise_for_status()
            time.sleep(0.11)
            page_data = page_resp.json()
            for form, date, acc, primary in zip(
                page_data.get("form", []),
                page_data.get("filingDate", []),
                page_data.get("accessionNumber", []),
                page_data.get("primaryDocument", []),
            ):
                if form in form_types and start_date <= date <= end_date:
                    results.append({
                        "ticker": ticker, "cik": cik,
                        "form": form, "date": date,
                        "accession_number": acc,
                        "primary_document": primary,
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch older filings page {older['name']}: {e}")

    return results


def download_filing(filing: dict, save_dir: Path) -> Path | None:
    """Download the primary document of a filing. Returns local path or None."""
    cik = str(int(filing["cik"]))
    acc = filing["accession_number"].replace("-", "")
    primary = filing["primary_document"]
    url = f"{_SEC_BASE}/Archives/edgar/data/{cik}/{acc}/{primary}"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        time.sleep(0.11)
    except Exception as e:
        logger.warning(f"Download failed {url}: {e}")
        return None

    out_path = save_dir / primary
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    logger.info(f"Saved {filing['form']} ({filing['date']}) → {out_path}")
    return out_path


def download_ticker(
    ticker: str,
    form_types: list[str],
    start_date: str,
    end_date: str,
    base_dir: Path,
) -> list[dict]:
    """
    Download all matching filings for a ticker.

    Local layout:
        base_dir/{ticker}/{form}/{date}_{accession}/primary_doc

    Returns list of dicts with filing metadata + 'local_path' key.
    """
    base_dir = Path(base_dir)
    filings = list_filings(ticker, form_types, start_date, end_date)
    logger.info(f"{ticker}: {len(filings)} filings of {form_types}")

    records = []
    for filing in filings:
        acc_clean = filing["accession_number"].replace("-", "")
        save_dir = (
            base_dir / ticker / filing["form"]
            / f"{filing['date']}_{acc_clean}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        local_path = download_filing(filing, save_dir)
        records.append({**filing, "local_path": str(local_path) if local_path else None})

    return records


def download_batch(
    tickers: list[str],
    form_types: list[str],
    start_date: str,
    end_date: str,
    base_dir: str | Path,
    company_map: dict[str, str] | None = None,
) -> dict:
    """
    Download filings for a list of tickers and save a company_map.json sidecar.

    Args:
        tickers:     list of stock tickers
        form_types:  e.g. ["10-K", "10-Q", "8-K"]
        start_date:  "YYYY-MM-DD"
        end_date:    "YYYY-MM-DD"
        base_dir:    root directory to save files
        company_map: optional {ticker: company_name} dict to save as sidecar

    Returns:
        summary dict: {tickers, downloaded, skipped}
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load company names from EDGAR if not provided
    if company_map is None:
        try:
            _load_ticker_map()
            cmap_resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=_HEADERS, timeout=15,
            )
            raw = cmap_resp.json()
            company_map = {
                v["ticker"].upper(): v.get("title", v["ticker"])
                for v in raw.values()
            }
        except Exception:
            company_map = {}

    # Save company map sidecar
    if company_map:
        cmap_path = base_dir / "company_map.json"
        cmap_path.write_text(json.dumps(company_map, indent=2), encoding="utf-8")
        logger.info(f"Company map saved → {cmap_path}")

    downloaded = 0
    skipped = 0
    for ticker in tickers:
        try:
            records = download_ticker(ticker, form_types, start_date, end_date, base_dir)
            downloaded += sum(1 for r in records if r["local_path"])
            skipped += sum(1 for r in records if not r["local_path"])
        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")
            skipped += 1

    return {"tickers": tickers, "downloaded": downloaded, "skipped": skipped}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SEC EDGAR filings")
    parser.add_argument("--ticker", nargs="+", required=True)
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--out",   default="data/filings")
    args = parser.parse_args()

    summary = download_batch(
        tickers=args.ticker,
        form_types=args.forms,
        start_date=args.start,
        end_date=args.end,
        base_dir=args.out,
    )
    print(f"\nDownload complete: {summary}")
