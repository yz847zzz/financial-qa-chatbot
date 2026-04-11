"""
Parse financial tables from SEC EDGAR filings into normalized FinancialRow objects.

Flow (mirrors FINDMIND's financial_state.py):
  raw HTML/PDF table (2D list of strings)
    → detect statement type (balance sheet / income / cash flow)
    → detect reporting scale (millions / thousands / billions)
    → parse each row: metric label + first numeric value
    → emit FinancialRow(ticker, period, statement, metric, value)

Key difference from Chinese FINDMIND: all keywords are English US GAAP.
Parenthesis-negatives "(1,234)" are standard US financial notation.
"""

from __future__ import annotations

import re
from loguru import logger

from ..metadata.models import FinancialRow


# ── Statement keyword anchors (US GAAP English) ───────────────────────────────

_BS_ANCHORS = [
    "TOTAL ASSETS", "CURRENT ASSETS", "CASH AND CASH EQUIVALENTS",
    "ACCOUNTS RECEIVABLE", "INVENTORIES", "GOODWILL", "INTANGIBLE ASSETS",
    "PROPERTY PLANT AND EQUIPMENT", "PROPERTY AND EQUIPMENT",
    "TOTAL LIABILITIES", "CURRENT LIABILITIES", "ACCOUNTS PAYABLE",
    "LONG-TERM DEBT", "DEFERRED REVENUE", "DEFERRED TAX",
    "STOCKHOLDERS EQUITY", "SHAREHOLDERS EQUITY", "RETAINED EARNINGS",
    "COMMON STOCK", "ADDITIONAL PAID-IN CAPITAL", "TREASURY STOCK",
    "TOTAL EQUITY", "NONCONTROLLING INTEREST",
]

_IS_ANCHORS = [
    "TOTAL REVENUE", "NET REVENUE", "NET SALES", "REVENUES",
    "COST OF GOODS SOLD", "COST OF REVENUE", "GROSS PROFIT",
    "OPERATING INCOME", "OPERATING EXPENSES",
    "RESEARCH AND DEVELOPMENT", "SELLING GENERAL AND ADMINISTRATIVE",
    "DEPRECIATION AND AMORTIZATION",
    "INTEREST EXPENSE", "INTEREST INCOME",
    "INCOME BEFORE TAX", "PROVISION FOR INCOME TAXES",
    "NET INCOME", "EARNINGS PER SHARE", "DILUTED EPS", "BASIC EPS",
    "WEIGHTED AVERAGE SHARES",
]

_CF_ANCHORS = [
    "CASH FLOWS FROM OPERATING", "OPERATING ACTIVITIES",
    "CASH FLOWS FROM INVESTING", "INVESTING ACTIVITIES",
    "CASH FLOWS FROM FINANCING", "FINANCING ACTIVITIES",
    "CAPITAL EXPENDITURES", "PURCHASES OF PROPERTY",
    "DEPRECIATION", "STOCK-BASED COMPENSATION",
    "NET CASH PROVIDED", "NET CASH USED",
    "CASH AND CASH EQUIVALENTS AT END",
]

_STATEMENT_ANCHORS: dict[str, list[str]] = {
    "balance_sheet":    _BS_ANCHORS,
    "income_statement": _IS_ANCHORS,
    "cash_flow":        _CF_ANCHORS,
}

# Financial statement page keywords for targeted table extraction in PDFs
FS_PAGE_KEYWORDS = [
    "CONSOLIDATED BALANCE SHEET",
    "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "CONSOLIDATED STATEMENTS OF INCOME",
    "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "BALANCE SHEETS",
    "STATEMENTS OF OPERATIONS",
    "STATEMENTS OF CASH FLOWS",
    "STATEMENTS OF INCOME",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_metric(name: str) -> str:
    """
    Normalise a metric label from a table row header.
    Strips leading bullets/numbers, collapses whitespace.
    """
    name = re.sub(r"^\s*[\d\.\-\(\)ivxlIVXL]+[\.\)]\s*", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip(" :.,;")


def _parse_numeric(raw: str) -> float | None:
    """
    Parse US financial number strings:
      "(1,234.5)"  → -1234.5   (parenthesis = negative)
      "1,234.5"    → 1234.5
      "--" / "N/A" → None
    """
    if not raw:
        return None
    raw = raw.strip()
    if raw in ("", "-", "--", "—", "N/A", "n/a", "nm", "NM", "*", "—"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()")
    raw = re.sub(r"[$,%\s]", "", raw)
    try:
        v = float(raw)
        return -v if negative else v
    except ValueError:
        return None


def detect_scale(text: str) -> float:
    """
    Detect reporting scale from surrounding document text.
    Returns multiplier: 1e9 (billions) | 1e6 (millions) | 1e3 (thousands) | 1.0
    """
    t = text.upper()
    if "IN BILLIONS" in t or "BILLIONS OF DOLLARS" in t:
        return 1_000_000_000.0
    if "IN MILLIONS" in t or "MILLIONS OF DOLLARS" in t or "(IN MILLIONS)" in t:
        return 1_000_000.0
    if "IN THOUSANDS" in t or "THOUSANDS OF DOLLARS" in t or "(IN THOUSANDS)" in t:
        return 1_000.0
    return 1.0


def infer_statement_type(table_text: str) -> str:
    """
    Score a table's combined text against keyword anchors.
    Returns statement type with most hits (min 2 required), else "unknown".
    """
    text_upper = table_text.upper()
    scores = {
        stmt: sum(1 for kw in anchors if kw in text_upper)
        for stmt, anchors in _STATEMENT_ANCHORS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "unknown"


# ── Main parsers ───────────────────────────────────────────────────────────────

def parse_table_rows(
    ticker: str,
    period: str,
    raw_table: list[list[str]],
    statement_type: str | None = None,
    scale: float = 1.0,
) -> list[FinancialRow]:
    """
    Convert a raw 2D table into FinancialRow objects.

    Layout assumed: col[0] = metric label, col[1+] = period values.
    Takes the first parseable numeric column as the primary value.

    Args:
        ticker:         stock ticker
        period:         "YYYY-MM" or "FYYYY"
        raw_table:      2D list (rows × cells) from extractor
        statement_type: override auto-detection
        scale:          multiplier from detect_scale()
    """
    if statement_type is None:
        flat = " ".join(cell for row in raw_table for cell in row)
        statement_type = infer_statement_type(flat)

    rows: list[FinancialRow] = []
    for row in raw_table:
        if not row or not row[0].strip():
            continue
        metric = _clean_metric(row[0])
        if not metric or len(metric) < 3:
            continue

        value: float | None = None
        raw_val: str = ""
        for cell in row[1:]:
            candidate = _parse_numeric(cell)
            if candidate is not None:
                value = candidate * scale
                raw_val = cell.strip()
                break

        rows.append(FinancialRow(
            ticker=ticker,
            period=period,
            statement=statement_type,
            metric=metric,
            value=value,
            raw_value=raw_val,
        ))

    return rows


def parse_text_financials(ticker: str, period: str, text: str) -> list[FinancialRow]:
    """
    Regex fallback parser for when table extraction yields nothing.
    Scans plain text for "Label .... $12,345.6" patterns.
    """
    scale = detect_scale(text)
    rows: list[FinancialRow] = []
    pattern = re.compile(
        r"([A-Z][A-Za-z &,\-/]{5,60}?)"
        r"[\s\.]{2,}"
        r"[\$\(]?\s*([\d,]+(?:\.\d+)?)\)?"
    )
    for match in pattern.finditer(text):
        metric = _clean_metric(match.group(1))
        raw_val = match.group(2)
        value = _parse_numeric(raw_val)
        if value is not None and metric and len(metric) >= 5:
            stmt = infer_statement_type(metric)
            rows.append(FinancialRow(
                ticker=ticker,
                period=period,
                statement=stmt,
                metric=metric,
                value=value * scale,
                raw_value=raw_val,
            ))
    logger.debug(f"Text fallback: {len(rows)} rows for {ticker} {period}")
    return rows


def extract_all_rows_from_filing(
    ticker: str,
    period: str,
    extractor,  # HtmlExtractor | PdfExtractor
) -> list[FinancialRow]:
    """
    High-level helper: extract ALL tables from a filing and parse to FinancialRow.
    Used by both the metric heatmap pass and the final ingest pass.
    """
    from .extractor import HtmlExtractor, PdfExtractor

    all_rows: list[FinancialRow] = []

    if isinstance(extractor, HtmlExtractor):
        full_text = extractor.extract_all_text()
        scale = detect_scale(full_text)
        raw_tables = extractor.extract_tables_as_rows()
        for raw_table in raw_tables:
            all_rows.extend(parse_table_rows(ticker, period, raw_table, scale=scale))
        if not all_rows:
            all_rows = parse_text_financials(ticker, period, full_text)

    elif isinstance(extractor, PdfExtractor):
        pages = extractor.extract_all_text()
        full_text = "\n\n".join(p.text for p in pages)
        scale = detect_scale(full_text)
        target_pages = extractor.find_pages_with_keywords(FS_PAGE_KEYWORDS)
        for pnum in target_pages:
            for raw_table in extractor.extract_tables_from_page(pnum):
                all_rows.extend(parse_table_rows(ticker, period, raw_table, scale=scale))
        if not all_rows:
            all_rows = parse_text_financials(ticker, period, full_text)

    return all_rows


if __name__ == "__main__":
    sample = [
        ["", "Sep 2023", "Sep 2022"],
        ["Total net sales", "383,285", "394,328"],
        ["Cost of sales", "214,137", "223,546"],
        ["Gross margin", "169,148", "170,782"],
        ["Net income", "96,995", "99,803"],
        ["Earnings per share — diluted", "6.13", "6.11"],
    ]
    rows = parse_table_rows("AAPL", "FY2023", sample, scale=1_000_000)
    for r in rows:
        print(r)
