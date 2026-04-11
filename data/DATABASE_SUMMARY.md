# SQLite Database Summary — `data/financials.db`

Generated: 2026-04-08

---

## Overview

| Table | Rows | Purpose |
|---|---|---|
| `filing_metadata` | 2,624 | Filing provenance — one row per filing document |
| `financials` | 62,276 | Extracted financial metrics per ticker/period |
| `canonical_metrics` | 137 | Frequency-filtered metric vocabulary (≥30% of tickers) |

---

## `filing_metadata`

Tracks every processed filing with its date, form type, and ingest stats.

| Field | Value |
|---|---|
| Tickers | 94 |
| Date range | 2019-01-02 → 2024-12-20 |
| Form types | 10-K (503 filings), 8-K (2,121 filings) |

**Coverage notes:**
- Most tickers have 5–6 annual 10-K filings (FY2019–FY2024)
- A few tickers have partial coverage due to filing availability:
  - `GOOG`, `NFLX`, `WMT` — 2 filings only
  - `VZ` — 1 filing only
  - `CRM`, `ACN` — recent years only (2023–2024)
- High 8-K counts on some tickers (e.g. `AMT` 162, `AXP` 148) reflect frequent material event disclosures

**Tickers (94 total):**
```
AAPL  ABT   ACN   ADBE  ADP   AIG   ALL   AMD   AMGN  AMT
AMZN  AON   APD   AVGO  AXP   BA    BIIB  BMY   BRK-B BSX
CAT   CHTR  CI    CL    CME   COST  CRM   CSCO  CVS   CVX
DE    DHR   DIS   DUK   ECL   ELV   EMR   ETN   FDX   GD
GE    GILD  GOOG  HD    HON   IBM   INTC  ISRG  JNJ   KO
LMT   LOW   MA    MCD   MDT   MMM   MO    MRK   MSFT  NEE
NFLX  NKE   NOC   NSC   NVDA  ORCL  PEP   PFE   PG    PM
PNC   QCOM  REGN  RTX   SBUX  SLB   SO    SPG   SYK   T
TGT   TMO   TSLA  TXN   UNH   UPS   USB   V     VRTX  VZ
WM    WMT   XOM   ZTS
```

---

## `financials`

Extracted financial metrics from HTML tables in 10-K and 8-K filings.

### Row counts by statement type

| Statement | Rows | Notes |
|---|---|---|
| `balance_sheet` | 18,037 | Best-classified statement |
| `income_statement` | 6,977 | |
| `cash_flow` | 4,497 | |
| `unknown` | 32,765 | Parser could not assign to a specific statement; includes footnotes, schedules, pension tables |

### Row counts by fiscal year (10-K)

| Period | Rows |
|---|---|
| FY2019 | 8,094 |
| FY2020 | 9,723 |
| FY2021 | 10,772 |
| FY2022 | 10,727 |
| FY2023 | 11,204 |
| FY2024 | 11,621 |

### Top 15 most common metrics

| Metric | Occurrences |
|---|---|
| Other | 1,171 |
| Net income | 909 |
| Cash and cash equivalents | 638 |
| Total assets | 583 |
| Assets | 569 |
| Accounts payable | 546 |
| Depreciation and amortization | 509 |
| Long-term debt | 508 |
| Deferred income taxes | 498 |
| Inventories | 482 |
| Other assets | 481 |
| Goodwill | 454 |
| Other, net | 430 |
| Interest expense | 423 |
| Liabilities | 423 |

Total distinct raw metric labels: **2,454** (many are near-duplicates; deduplicated via `canonical_metrics`)

### Period format

- Annual (10-K): `FY{YYYY}` — e.g. `FY2023`
- Quarterly/event (8-K): `YYYY-MM` — e.g. `2023-09`
- Values in USD; unit stored in `unit` column (default: `USD`)

---

## `canonical_metrics`

Built by the metric frequency heatmap analysis (`run_metric_analysis.py`).
A metric is canonical if it appears in ≥ 30% of all tickers.
Used to filter `financials` writes during ingest — prevents polluting SQL with one-off footnote labels.

| Field | Value |
|---|---|
| Total canonical metrics | 137 |
| Threshold | 30% of tickers |
| Coverage range | 30.9% → 97.9% of tickers |

### Breakdown by statement

| Statement | Count |
|---|---|
| `balance_sheet` | 39 |
| `income_statement` | 17 |
| `cash_flow` | 9 |
| `unknown` | 72 |

> **Note:** 72 of 137 canonical metrics are classified as `unknown` — these are real financial terms
> (e.g. pension items, lease schedules, tax rollforwards) that appear across many filings but
> whose table headers don't match standard income/balance/cashflow keywords.

### Top canonical metrics by coverage

| Canonical Name | Statement | % of Tickers |
|---|---|---|
| Assets | balance_sheet | 97.9% |
| Other | unknown | 97.9% |
| Goodwill | balance_sheet | 91.5% |
| Cash Equivalents | balance_sheet | 90.4% |
| Assets Current | balance_sheet | 89.4% |
| Current Liabilities | balance_sheet | 89.4% |
| Debt Long Term | unknown | 81.9% |
| Expense Interest | income_statement | 81.9% |
| Earnings Retained | balance_sheet | 80.9% |
| Liabilities | balance_sheet | 80.9% |
| Deferred Income Taxes | balance_sheet | 78.7% |
| Accounts Payable | balance_sheet | 75.5% |
| Depreciation Amortization | income_statement | 73.4% |
| Cash Operating Activities | cash_flow | 72.3% |
| Income | income_statement | 72.3% |

---

## ChromaDB (Vector Store)

Stored at `data/vectordb/` — populated in parallel with SQLite.

| Field | Value |
|---|---|
| Collection | `financial_docs` |
| Embedding model | `all-MiniLM-L6-v2` |
| Total chunks | **516,955** |
| Source | 10-K full text, split by ITEM section, 512-char windows with 64-char overlap |

Each chunk carries metadata: `ticker`, `company`, `filing_type`, `date`, `section`, `accession_number`, `source_path`, `chunk_index`.

---

## Known Data Quality Issues

1. **52% of `financials` rows are `unknown` statement** — the table parser cannot always determine whether a table is income/balance/cashflow from the surrounding HTML context. These rows are still stored and searchable but will not appear in typed pivots.

2. **Sparse tickers** — `VZ` (1 filing), `GOOG`/`NFLX`/`WMT` (2 filings) have limited coverage due to fewer files in the source filings directory.

3. **Metric normalisation is bag-of-words** — canonical metric keys like `"Cash Equivalents"` collapse both `"Cash and cash equivalents"` and `"Cash equivalents and short-term investments"`. Good for coverage, but the display name loses some specificity.

4. **Values not inflation-adjusted** — all values are nominal USD as reported.
