"""
Extract text and tables from SEC EDGAR HTML and PDF filings.

Includes section-splitting logic baked in (HtmlExtractor.extract_sections).

Factory: make_extractor(path) → HtmlExtractor | PdfExtractor
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import warnings

import pdfplumber
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from loguru import logger

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class PageText:
    page_num: int
    text: str


# ── ITEM heading regex for 10-K / 10-Q section splitting ──────────────────────

_ITEM_ANCHORS = [
    "ITEM 1.", "ITEM 1A.", "ITEM 1B.", "ITEM 1C.",
    "ITEM 2.", "ITEM 3.", "ITEM 4.",
    "ITEM 5.", "ITEM 6.", "ITEM 7.", "ITEM 7A.",
    "ITEM 8.", "ITEM 9.", "ITEM 9A.", "ITEM 9B.",
    "ITEM 10.", "ITEM 11.", "ITEM 12.", "ITEM 13.", "ITEM 14.", "ITEM 15.",
    "PART I", "PART II", "PART III", "PART IV",
]


class HtmlExtractor:
    """
    Extract text and tables from SEC EDGAR .htm/.html filings.
    Most 10-K, 10-Q, 8-K filings are HTML.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._soup: BeautifulSoup | None = None

    def _get_soup(self) -> BeautifulSoup:
        if self._soup is None:
            content = self.path.read_bytes()
            self._soup = BeautifulSoup(content, "lxml")
        return self._soup

    def extract_all_text(self) -> str:
        """
        Full visible text with scripts/styles and iXBRL tags stripped.
        Modern SEC EDGAR filings are iXBRL-formatted: they embed XML namespaced
        tags (ix:nonfraction, xbrli:context, etc.) in the HTML. We remove those
        tags but keep their text content so financial numbers are preserved.
        Collapses excessive blank lines.
        """
        soup = self._get_soup()

        # Remove non-visible / boilerplate tags entirely (no text kept)
        for tag in soup(["script", "style", "meta", "head"]):
            tag.decompose()

        # Strip iXBRL/XBRL structural tags but preserve their text content.
        # These tags look like <ix:nonfraction ...>89,500</ix:nonfraction>
        # or <xbrli:context ...>...</xbrli:context>.
        # We unwrap data-bearing tags (keep text) and decompose context/schema tags.
        _XBRL_DECOMPOSE = re.compile(
            r"^(xbrli|xbrldi|xlink|link|dei|us-gaap|iso4217|"
            r"ix:header|ix:references|ix:resources)$",
            re.IGNORECASE,
        )
        _XBRL_UNWRAP = re.compile(r"^ix:", re.IGNORECASE)

        for tag in soup.find_all(True):
            name = tag.name or ""
            if _XBRL_DECOMPOSE.match(name) or "context" in name.lower():
                tag.decompose()
            elif _XBRL_UNWRAP.match(name):
                tag.unwrap()  # keep text, remove the tag wrapper

        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_sections(self, anchors: list[str] | None = None) -> dict[str, str]:
        """
        Split full text into named SEC Item sections.
        Returns {section_header: section_text}.
        Sections with no content are dropped.
        """
        anchors = anchors or _ITEM_ANCHORS
        full_text = self.extract_all_text()
        lines = full_text.split("\n")

        sections: dict[str, list[str]] = {"preamble": []}
        current = "preamble"

        # Track how many times each anchor key has been seen.
        # The first occurrence is usually the Table of Contents entry (short).
        # The second occurrence is the actual body section (long).
        # We reset the accumulator on the second+ occurrence so the body wins.
        seen_count: dict[str, int] = {}

        for line in lines:
            upper = line.strip().upper()
            matched = False
            for anchor in anchors:
                if upper.startswith(anchor) and len(line.strip()) < 120:
                    current = line.strip()
                    seen_count[current] = seen_count.get(current, 0) + 1
                    if seen_count[current] == 1:
                        sections[current] = []   # first occurrence — start fresh
                    else:
                        sections[current] = []   # body occurrence — reset, discard TOC stub
                    matched = True
                    break
            if not matched:
                sections[current].append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}

    def extract_tables_as_rows(self) -> list[list[list[str]]]:
        """
        Extract all HTML <table> elements as 2D lists.
        Skips single-row tables and tables with no data cells.
        """
        soup = self._get_soup()
        tables: list[list[list[str]]] = []
        for table_tag in soup.find_all("table"):
            rows: list[list[str]] = []
            for tr in table_tag.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if any(c for c in cells):
                    rows.append(cells)
            if len(rows) >= 2:
                tables.append(rows)
        return tables

    def find_sections_with_keywords(self, keywords: list[str]) -> dict[str, str]:
        sections = self.extract_sections()
        return {
            name: text
            for name, text in sections.items()
            if any(kw.upper() in text.upper() for kw in keywords)
        }


class PdfExtractor:
    """Extract text and tables from PDF financial filings (pdfplumber)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_pages(self) -> Iterator[PageText]:
        with pdfplumber.open(self.path) as pdf:
            for page in pdf.pages:
                yield PageText(page_num=page.page_number, text=page.extract_text() or "")

    def extract_all_text(self) -> list[PageText]:
        return list(self.iter_pages())

    def extract_tables_from_page(self, page_num: int) -> list[list[list[str]]]:
        with pdfplumber.open(self.path) as pdf:
            page = pdf.pages[page_num - 1]
            raw_tables = page.extract_tables() or []
        cleaned = []
        for table in raw_tables:
            rows = [[cell.strip() if cell else "" for cell in row] for row in table]
            if sum(1 for row in rows for c in row if c) >= 3:
                cleaned.append(rows)
        return cleaned

    def find_pages_with_keywords(self, keywords: list[str]) -> list[int]:
        matches = []
        for page in self.iter_pages():
            if any(kw.upper() in page.text.upper() for kw in keywords):
                matches.append(page.page_num)
        return matches

    def extract_tables_near_keywords(
        self, keywords: list[str], window: int = 2
    ) -> list[list[list[str]]]:
        match_pages = self.find_pages_with_keywords(keywords)
        if not match_pages:
            return []
        with pdfplumber.open(self.path) as pdf:
            max_page = len(pdf.pages)
        page_set: set[int] = set()
        for p in match_pages:
            for offset in range(-window, window + 1):
                c = p + offset
                if 1 <= c <= max_page:
                    page_set.add(c)
        all_tables: list[list[list[str]]] = []
        for pnum in sorted(page_set):
            all_tables.extend(self.extract_tables_from_page(pnum))
        return all_tables


def make_extractor(path: str | Path) -> HtmlExtractor | PdfExtractor:
    """Factory: return the right extractor based on file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PdfExtractor(path)
    elif suffix in (".html", ".htm"):
        return HtmlExtractor(path)
    else:
        logger.warning(f"Unknown suffix '{suffix}' for {path.name}, defaulting to HTML")
        return HtmlExtractor(path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m data_pipeline.processing.extractor <file>")
        sys.exit(1)
    ext = make_extractor(sys.argv[1])
    if isinstance(ext, HtmlExtractor):
        sections = ext.extract_sections()
        print(f"Sections found: {list(sections.keys())}")
        tables = ext.extract_tables_as_rows()
        print(f"Tables found: {len(tables)}")
    else:
        pages = ext.extract_all_text()
        print(f"Pages: {len(pages)}, chars: {sum(len(p.text) for p in pages)}")
