"""
ChromaDB-backed vector store for semantic search over filing text chunks.
Extended metadata fields added for financial filing provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

from ..metadata.models import TextChunk

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_BATCH_SIZE = 64


class VectorStore:
    """
    ChromaDB persistent store for TextChunk embeddings.

    Usage:
        vs = VectorStore("data/vectordb")
        vs.add(chunks)
        results = vs.search("Apple revenue growth", ticker="AAPL", n=5)
    """

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "financial_docs",
        embedding_model: str = _DEFAULT_MODEL,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model '{embedding_model}'...")
        self._embedder = SentenceTransformer(embedding_model)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready: {self.persist_dir} ({self._col.count()} chunks stored)"
        )

    def add(self, chunks: Sequence[TextChunk], batch_size: int = _BATCH_SIZE) -> int:
        """
        Embed and store chunks. Deduplicates by chunk_id.
        Returns number of newly added chunks.
        """
        if not chunks:
            return 0

        existing = set(self._col.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0

        total_added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ).tolist()

            self._col.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "ticker":       c.ticker,
                        "doc_type":     c.doc_type,
                        "filing_date":  c.filing_date,
                        "source_path":  c.source_path,
                        "chunk_index":  c.chunk_index,
                        "doc_id":       c.doc_id,
                        "section":      c.section,
                        **{k: v for k, v in c.metadata.items()
                           if isinstance(v, (str, int, float, bool))},
                    }
                    for c in batch
                ],
            )
            total_added += len(batch)

        logger.info(f"Added {total_added} chunks (store total: {self._col.count()})")
        return total_added

    def search(
        self,
        query: str,
        ticker: str | None = None,
        doc_type: str | None = None,
        filing_date_gte: str | None = None,
        section: str | None = None,
        n: int = 5,
    ) -> list[dict]:
        """
        Semantic search with optional metadata filters.

        Returns list of dicts:
            {text, ticker, doc_type, filing_date, section, distance, ...metadata}
        """
        if self._col.count() == 0:
            return []

        embedding = self._embedder.encode([query], normalize_embeddings=True).tolist()

        filters: list[dict] = []
        if ticker:
            filters.append({"ticker": {"$eq": ticker}})
        if doc_type:
            filters.append({"doc_type": {"$eq": doc_type}})
        if filing_date_gte:
            filters.append({"filing_date": {"$gte": filing_date_gte}})
        if section:
            filters.append({"section": {"$eq": section}})

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        kwargs: dict = {
            "query_embeddings": embedding,
            "n_results": min(n, self._col.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "distance": float(dist), **meta})
        return output

    def search_multi_query(
        self,
        queries: list[str],
        ticker: str | None = None,
        n_per_query: int = 3,
    ) -> list[dict]:
        """
        Multi-query retrieval: run each query, deduplicate, sort by distance.
        """
        seen: dict[str, dict] = {}
        for q in queries:
            for result in self.search(q, ticker=ticker, n=n_per_query):
                cid = f"{result.get('doc_id', '')}_{result.get('chunk_index', 0)}"
                if cid not in seen or result["distance"] < seen[cid]["distance"]:
                    seen[cid] = result
        return sorted(seen.values(), key=lambda x: x["distance"])

    def delete_by_ticker(self, ticker: str) -> int:
        ids = self._col.get(where={"ticker": {"$eq": ticker}})["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    @property
    def count(self) -> int:
        return self._col.count()

    def __repr__(self) -> str:
        return f"VectorStore({self.persist_dir}, {self.count} chunks)"


if __name__ == "__main__":
    from ..processing.chunker import chunk_text
    vs = VectorStore("data/test_vectordb")
    chunks = chunk_text(
        "Apple revenue was $89.5B in Q1 FY2024. Services grew 11% YoY.",
        "AAPL", "10-K", "2024-02-01", "/tmp/test.htm", chunk_size=80, overlap=20,
    )
    vs.add(chunks)
    print(f"Store count: {vs.count}")
    for r in vs.search("revenue growth", ticker="AAPL", n=2):
        print(f"  [{r['distance']:.3f}] {r['text'][:80]}")
