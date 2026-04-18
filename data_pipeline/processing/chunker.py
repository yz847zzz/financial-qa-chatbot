"""
Split document text into overlapping chunks for embedding.

TextChunk is defined in metadata/models.py — this module just provides
the splitting logic.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..metadata.models import FilingMetadata, TextChunk


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    ticker: str,
    doc_type: str,
    filing_date: str,
    source_path: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
    section: str = "",
) -> list[TextChunk]:
    """
    Sliding-window chunker with sentence-boundary awareness.

    Args:
        text:        full section or document text
        chunk_size:  target character count per chunk (soft limit)
        overlap:     approx chars to carry over between adjacent chunks
        metadata:    extra flat dict stored on each chunk
        section:     SEC Item section name (e.g. "ITEM 7")
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[TextChunk] = []
    current: list[str] = []
    current_len = 0
    chunk_idx = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_len + sent_len > chunk_size and current:
            chunks.append(TextChunk(
                text=" ".join(current),
                ticker=ticker,
                doc_type=doc_type,
                filing_date=filing_date,
                source_path=source_path,
                chunk_index=chunk_idx,
                section=section,
                metadata={**metadata, "section": section},
            ))
            chunk_idx += 1

            # Carry-over overlap
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev)
            current = carried
            current_len = carried_len

        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(TextChunk(
            text=" ".join(current),
            ticker=ticker,
            doc_type=doc_type,
            filing_date=filing_date,
            source_path=source_path,
            chunk_index=chunk_idx,
            section=section,
            metadata={**metadata, "section": section},
        ))

    return chunks


def chunk_by_section(
    sections: dict[str, str],
    ticker: str,
    doc_type: str,
    filing_date: str,
    source_path: str,
    filing_meta: FilingMetadata | None = None,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[TextChunk]:
    """
    Chunk a {section_name: text} dict.
    Each chunk's metadata includes the section name.
    Global chunk_index is re-assigned across all sections.
    """
    all_chunks: list[TextChunk] = []
    base_meta = filing_meta.flat_dict() if filing_meta else {}

    for section_name, section_text in sections.items():
        section_chunks = chunk_text(
            text=section_text,
            ticker=ticker,
            doc_type=doc_type,
            filing_date=filing_date,
            source_path=source_path,
            chunk_size=chunk_size,
            overlap=overlap,
            section=section_name,
            metadata={**base_meta, "section": section_name},
        )
        for chunk in section_chunks:
            chunk.chunk_index = len(all_chunks)
            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    sample = (
        "Apple Inc. reported record revenue of $89.5 billion in Q1 FY2024. "
        "Services revenue grew 11% year-over-year to $23.1 billion. "
        "The company repurchased $20.7 billion of its own shares. "
        "Tim Cook said: 'We are thrilled with the results.' "
        "iPhone revenue came in at $69.7 billion, beating estimates."
    )
    chunks = chunk_text(sample, "AAPL", "10-K", "2024-02-01", "/tmp/test.htm",
                        chunk_size=120, overlap=30)
    for c in chunks:
        print(f"[{c.chunk_id}] ({len(c.text)}c) {c.text[:80]}")
