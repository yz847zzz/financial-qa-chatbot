"""
Hybrid retrieval pipeline for financial RAG.

Two-stage architecture (industry standard):
  Stage 1 — Recall: BM25 (sparse) + ChromaDB (dense) → merged candidate pool
  Stage 2 — Rerank: cross-encoder scores each (query, passage) → top-N returned

Recall merging uses Reciprocal Rank Fusion (RRF), which is parameter-free and
consistently outperforms score-based fusion when the two retrievers use
incompatible score scales (cosine distance vs BM25 TF-IDF score).

Metadata pre-filtering (applied before scoring, not after):
  Filters extracted automatically from the question:
    - ticker   : "AAPL", "Apple" → {"ticker": "AAPL"}
    - filing_type: "10-K", "10-Q", "8-K" mentioned explicitly
    - date     : "FY2023", "fiscal 2022", "2021" → year-range filter on metadata.date
  Filtering happens at the recall stage for both ChromaDB (where= clause) and BM25
  (document index mask), so the reranker only sees relevant documents.
  Pre-filtering is faster and more precise than post-filtering, especially when
  one company dominates the corpus (e.g. AAPL has ~500 chunks, all others irrelevant
  if the question is about Apple).

Cross-encoder model options (all downloadable from HuggingFace):
  - "cross-encoder/ms-marco-MiniLM-L-6-v2"   fast, 22M params, good general quality
  - "BAAI/bge-reranker-base"                  better multilingual + finance quality
  - "cross-encoder/ms-marco-electra-base"     higher quality, slower
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Known ticker universe (used for filter extraction) ─────────────────────────
# Superset of what's in the DB — false positives are harmless (just no results).
_KNOWN_TICKERS: set[str] = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "DIS", "BAC", "XOM",
    "PFE", "CVX", "KO", "ABBV", "AVGO", "COST", "MRK", "TMO", "CSCO",
    "NKE", "ORCL", "CRM", "LLY", "AMD", "INTC", "QCOM", "NFLX", "MCD",
    "LOW", "CAT", "MMM", "GILD", "REGN", "BIIB", "TXN", "ACN", "ABT",
    "PEP", "IBM", "BA", "HON", "FDX", "UPS", "GE", "T", "DE",
    "ADBE", "CRM", "SBUX", "TGT", "AMGN", "VRTX", "MA", "BRK-B",
}

# Company name → ticker (lowercase keys)
_COMPANY_TO_TICKER: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META",
    "nvidia": "NVDA", "tesla": "TSLA", "jpmorgan": "JPM",
    "visa": "V", "johnson": "JNJ", "walmart": "WMT",
    "procter": "PG", "unitedhealth": "UNH", "home depot": "HD",
    "disney": "DIS", "bank of america": "BAC", "exxon": "XOM",
    "pfizer": "PFE", "chevron": "CVX", "coca-cola": "KO",
    "abbvie": "ABBV", "broadcom": "AVGO", "costco": "COST",
    "merck": "MRK", "thermo fisher": "TMO", "cisco": "CSCO",
    "nike": "NKE", "oracle": "ORCL", "salesforce": "CRM",
    "eli lilly": "LLY", "intel": "INTC", "qualcomm": "QCOM",
    "netflix": "NFLX", "mcdonald": "MCD", "lowe": "LOW",
    "caterpillar": "CAT", "3m": "MMM", "gilead": "GILD",
    "regeneron": "REGN", "biogen": "BIIB", "texas instruments": "TXN",
    "accenture": "ACN", "abbott": "ABT", "pepsico": "PEP",
    "ibm": "IBM", "boeing": "BA", "honeywell": "HON",
    "fedex": "FDX", "ups": "UPS", "general electric": "GE",
    "starbucks": "SBUX", "target": "TGT", "amgen": "AMGN",
    "mastercard": "MA", "adobe": "ADBE",
}


# ── Data contract ──────────────────────────────────────────────────────────────

@dataclass
class MetadataFilters:
    """
    Filters extracted from the user question.
    None means "no constraint" (match all documents).
    """
    ticker: str | None = None         # e.g. "AAPL"
    filing_type: str | None = None    # "10-K" | "10-Q" | "8-K"
    year: str | None = None           # e.g. "2023" — matched against metadata.date prefix


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_rank: int | None = None      # rank in ChromaDB results (1-based; None if not retrieved)
    sparse_rank: int | None = None     # rank in BM25 results (1-based; None if not retrieved)
    rrf_score: float = 0.0             # Reciprocal Rank Fusion score (higher = better)
    rerank_score: float | None = None  # cross-encoder score (higher = more relevant)
    mmr_score: float | None = None     # MMR score at selection time (higher = selected earlier)


# ── Filter extraction ──────────────────────────────────────────────────────────

def parse_filters(question: str) -> MetadataFilters:
    """
    Extract metadata constraints from the natural language question.

    Examples:
        "What was Apple's revenue in FY2023?"
            → MetadataFilters(ticker="AAPL", year="2023")
        "Show MSFT 10-K filings from 2022"
            → MetadataFilters(ticker="MSFT", filing_type="10-K", year="2022")
        "How do tech companies handle AI risk?"
            → MetadataFilters()  (no constraints — broad search)
    """
    filters = MetadataFilters()
    q_lower = question.lower()

    # ── Ticker: uppercase word matching known tickers ──────────────────────────
    words = set(re.findall(r"\b[A-Z]{1,5}\b", question))
    matched = words & _KNOWN_TICKERS
    if len(matched) == 1:
        filters.ticker = matched.pop()
    elif len(matched) > 1:
        # Multiple tickers → don't constrain (comparison query)
        filters.ticker = None

    # ── Company name → ticker (if no direct ticker match yet) ─────────────────
    if filters.ticker is None:
        for company, ticker in _COMPANY_TO_TICKER.items():
            if company in q_lower:
                filters.ticker = ticker
                break

    # ── Filing type ───────────────────────────────────────────────────────────
    if re.search(r"\b10-?K\b", question, re.I):
        filters.filing_type = "10-K"
    elif re.search(r"\b10-?Q\b", question, re.I):
        filters.filing_type = "10-Q"
    elif re.search(r"\b8-?K\b", question, re.I):
        filters.filing_type = "8-K"

    # ── Year: FY2023, fiscal 2022, "in 2021", "for 2020" ─────────────────────
    year_match = re.search(r"\b(?:FY|fiscal\s+)?20(\d{2})\b", question, re.I)
    if year_match:
        filters.year = "20" + year_match.group(1)  # e.g. "2023"

    return filters


def _build_chroma_where(filters: MetadataFilters) -> dict | None:
    """
    Convert MetadataFilters to a ChromaDB `where` clause.
    ChromaDB supports: {"field": "value"} and {"$and": [...]} operators.
    Year filter uses "$contains" on the date string (e.g. date="2023-11-03" contains "2023").
    Returns None if no filters are active (fetch all).
    """
    clauses = []

    if filters.ticker:
        clauses.append({"ticker": {"$eq": filters.ticker}})
    if filters.filing_type:
        clauses.append({"filing_type": {"$eq": filters.filing_type}})
    if filters.year:
        clauses.append({"date": {"$contains": filters.year}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _matches_filters(meta: dict, filters: MetadataFilters) -> bool:
    """Check whether a document's metadata satisfies the filters (for BM25 masking)."""
    if filters.ticker and meta.get("ticker") != filters.ticker:
        return False
    if filters.filing_type and meta.get("filing_type") != filters.filing_type:
        return False
    if filters.year and filters.year not in (meta.get("date") or ""):
        return False
    return True


# ── BM25 index (built once at startup) ────────────────────────────────────────

class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer for manual query embedding.
    ChromaDB 1.x enforces embedding-function name matching against whatever was
    stored when the collection was first created.  Opening the collection without
    an ef (client.get_collection(name=...)) and embedding queries manually avoids
    that version-mismatch conflict entirely.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()


class BM25Index:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Loads all documents from ChromaDB once and builds the index in memory.
    Stores document metadata alongside text so filter masking works at query time.
    """

    def __init__(self, collection):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        print("Building BM25 index from ChromaDB documents ...", flush=True)

        PAGE = 5000
        all_ids, all_docs, all_metas = [], [], []
        offset = 0
        while True:
            batch = collection.get(
                limit=PAGE,
                offset=offset,
                include=["documents", "metadatas"],
            )
            if not batch["ids"]:
                break
            all_ids.extend(batch["ids"])
            all_docs.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])
            offset += PAGE
            if len(batch["ids"]) < PAGE:
                break

        self._ids = all_ids
        self._docs = all_docs
        self._metas = all_metas  # stored for filter masking
        tokenized = [self._tokenize(d) for d in all_docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built: {len(all_ids):,} documents.", flush=True)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9$%]+", text.lower())

    def search(
        self,
        query: str,
        k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[str, float]]:
        """
        Returns up to k (chunk_id, bm25_score) pairs, sorted by score descending.

        If filters are provided, only documents whose metadata satisfies the
        filters are eligible — the BM25 scores of masked-out documents are set
        to -inf before sorting (pre-filter, not post-filter).
        """
        scores = self._bm25.get_scores(self._tokenize(query))

        # Apply metadata mask: zero out ineligible documents
        if filters and any([filters.ticker, filters.filing_type, filters.year]):
            mask = np.array([
                _matches_filters(m, filters) for m in self._metas
            ], dtype=bool)
            scores = np.where(mask, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:k]
        # Exclude masked-out (score == -inf) results
        return [
            (self._ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > -np.inf
        ]


# ── Stage 1: Recall ────────────────────────────────────────────────────────────

def _dense_recall(
    question: str,
    collection,
    embed_fn,
    k: int,
    filters: MetadataFilters | None = None,
) -> list[RetrievedChunk]:
    """ChromaDB semantic search with optional metadata pre-filter.
    Uses pre-computed query embedding (query_embeddings=) to avoid
    ChromaDB 1.x embedding-function name-conflict errors."""
    query_vec = embed_fn.embed([question])
    where = _build_chroma_where(filters) if filters else None
    # ChromaDB 1.x: "ids" is always returned, not a valid include field
    kwargs = dict(
        query_embeddings=query_vec,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        # ChromaDB raises if where= matches 0 docs and n_results > 0
        print(f"[WARN] Dense recall with filter failed ({e}), retrying without filter.")
        results = collection.query(
            query_embeddings=query_vec,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    chunks = []
    for rank, (cid, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), start=1):
        chunks.append(RetrievedChunk(
            chunk_id=cid, text=doc, metadata=meta, dense_rank=rank,
        ))
    return chunks


def _sparse_recall(
    question: str,
    bm25: BM25Index,
    k: int,
    collection,
    filters: MetadataFilters | None = None,
) -> list[RetrievedChunk]:
    """BM25 keyword search with metadata mask pre-applied."""
    hits = bm25.search(question, k, filters=filters)
    if not hits:
        return []

    hit_ids = [h[0] for h in hits]
    fetched = collection.get(
        ids=hit_ids,
        include=["documents", "metadatas"],
    )
    id_to_data = {
        cid: (doc, meta)
        for cid, doc, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        )
    }

    chunks = []
    for rank, cid in enumerate(hit_ids, start=1):
        if cid not in id_to_data:
            continue
        doc, meta = id_to_data[cid]
        chunks.append(RetrievedChunk(
            chunk_id=cid, text=doc, metadata=meta, sparse_rank=rank,
        ))
    return chunks


# ── Stage 1b: Reciprocal Rank Fusion ──────────────────────────────────────────
# RRF(d) = Σ_r  1 / (k + rank_r(d))   k=60 (Cormack et al. 2009)

RRF_K = 60


def _reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge two ranked lists via RRF. Returns deduped list sorted by rrf_score desc."""
    merged: dict[str, RetrievedChunk] = {}

    for chunk in dense:
        if chunk.chunk_id not in merged:
            merged[chunk.chunk_id] = chunk
        else:
            merged[chunk.chunk_id].dense_rank = chunk.dense_rank
        merged[chunk.chunk_id].rrf_score += 1.0 / (RRF_K + chunk.dense_rank)

    for chunk in sparse:
        if chunk.chunk_id not in merged:
            merged[chunk.chunk_id] = chunk
        else:
            merged[chunk.chunk_id].sparse_rank = chunk.sparse_rank
        merged[chunk.chunk_id].rrf_score += 1.0 / (RRF_K + chunk.sparse_rank)

    return sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)


# ── Stage 2: Cross-encoder Reranking ──────────────────────────────────────────

class CrossEncoderReranker:
    """
    Scores (query, passage) pairs using a cross-encoder.
    Full cross-attention between query and passage — more accurate than
    bi-encoder cosine similarity for final relevance judgement.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        print(f"Loading cross-encoder: {model_name} ...", flush=True)
        self._model = CrossEncoder(model_name, max_length=512)
        print("Cross-encoder ready.", flush=True)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score each (query, chunk.text) pair. Returns chunks sorted by score desc."""
        if not chunks:
            return chunks
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)


# ── Stage 3: MMR Deduplication ────────────────────────────────────────────────
# Maximal Marginal Relevance (Carbonell & Goldstein, 1998).
#
# Problem: top-k by rerank_score often returns adjacent chunks from the same
# filing section — 60-80% token overlap, wasting the LLM context window on
# repetition while crowding out chunks from other filings.
#
# MMR fixes this with a greedy selection loop:
#   score(c) = λ · rel(c) − (1−λ) · max_{s ∈ selected} sim(c, s)
#
# At each step, pick the candidate that is most relevant *and* most different
# from what's already been selected. λ=0.7 keeps relevance dominant.
#
# Similarity metric: character trigram Jaccard.
#   - No model call (O(n·top_n) string ops, fast on 512-token chunks)
#   - Captures literal near-duplicates better than embedding cosine at this scale
#   - A Jaccard score >0.5 on 3-grams reliably identifies same-section overlap
#
# Alternative if you want semantic diversity instead of lexical:
#   use bi-encoder embeddings (already available in ChromaDB) as the similarity.
#   That catches paraphrased duplicates but is slower.


def _trigram_jaccard(a: str, b: str) -> float:
    """
    Character trigram Jaccard similarity ∈ [0, 1].
    1.0 = identical text, 0.0 = no shared trigrams.
    Fast: ~0.1ms for two 512-char strings.
    """
    if not a or not b:
        return 0.0
    n = 3
    set_a = {a[i: i + n] for i in range(len(a) - n + 1)}
    set_b = {b[i: i + n] for i in range(len(b) - n + 1)}
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _normalize_rerank_scores(chunks: list[RetrievedChunk]) -> dict[str, float]:
    """
    Min-max normalize rerank_scores to [0, 1] so λ blending is meaningful.
    Cross-encoder raw scores have arbitrary scale (can be negative, >1, etc.).
    """
    scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if not scores:
        return {c.chunk_id: 0.0 for c in chunks}
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {c.chunk_id: 1.0 for c in chunks}
    return {c.chunk_id: (c.rerank_score - lo) / (hi - lo) for c in chunks}


def mmr_select(
    chunks: list[RetrievedChunk],
    top_n: int,
    lambda_param: float = 0.7,
    overlap_threshold: float = 0.85,
) -> list[RetrievedChunk]:
    """
    Greedy MMR selection from a reranked candidate pool.

    Args:
        chunks:           candidates sorted by rerank_score descending
        top_n:            number of chunks to return
        lambda_param:     λ — relevance weight (0=pure diversity, 1=pure relevance)
                          0.7 is a good default: relevance-first but dedup aggressive
        overlap_threshold: hard dedup — if Jaccard(c, any_selected) > this, skip c
                          entirely (saves MMR score budget for clearly distinct chunks).
                          Set to 1.0 to rely purely on MMR soft penalty.

    Returns:
        Diverse top-n chunks with mmr_score set on each.

    Complexity: O(top_n · |candidates|) Jaccard comparisons.
    For 40 candidates, top_n=5: 200 comparisons → ~20ms, negligible.
    """
    if not chunks or top_n <= 0:
        return []

    norm_scores = _normalize_rerank_scores(chunks)
    remaining = list(chunks)  # candidates not yet selected
    selected: list[RetrievedChunk] = []

    while len(selected) < top_n and remaining:
        if not selected:
            # First pick: simply the highest rerank score (no diversity penalty yet)
            best = remaining.pop(0)
            best.mmr_score = norm_scores[best.chunk_id]
            selected.append(best)
            continue

        best_score = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            # Max similarity to any already-selected chunk
            max_sim = max(
                _trigram_jaccard(candidate.text, s.text) for s in selected
            )

            # Hard dedup: skip near-duplicates entirely
            if max_sim >= overlap_threshold:
                continue

            # MMR score
            rel = norm_scores[candidate.chunk_id]
            mmr = lambda_param * rel - (1.0 - lambda_param) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx = i

        if best_score == -float("inf"):
            # All remaining candidates were hard-deduped — stop early
            break

        chosen = remaining.pop(best_idx)
        chosen.mmr_score = best_score
        selected.append(chosen)

    return selected


# ── Public API ─────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Full three-stage retrieval pipeline with metadata pre-filtering and MMR dedup.

    Stage 1 — Recall:   BM25 (sparse) + ChromaDB dense, both pre-filtered by
                        ticker / filing_type / year extracted from the question.
    Stage 1b — RRF:     Reciprocal Rank Fusion merges both ranked lists.
    Stage 2 — Rerank:   Cross-encoder scores each (query, passage) pair.
    Stage 3 — MMR:      Greedy selection maximising relevance − overlap penalty,
                        so top_n chunks come from different sections/filings.

    Startup (once):
        retriever = HybridRetriever(collection)

    Per-query:
        chunks = retriever.retrieve("What did Apple say about supply chain?", top_n=3)
        # filters: ticker=AAPL (auto-extracted)
        # recall → RRF → rerank → MMR → 3 diverse, relevant chunks returned
    """

    def __init__(
        self,
        collection,
        recall_k: int = 20,
        embed_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        mmr_lambda: float = 0.7,
        mmr_overlap_threshold: float = 0.85,
    ):
        """
        Args:
            collection:            ChromaDB collection (opened without embedding_function)
            recall_k:              candidates per retriever (BM25 + dense = up to 2×recall_k)
            embed_model:           SentenceTransformer model name for dense query embedding
            reranker_model:        HuggingFace cross-encoder model ID
            mmr_lambda:            λ for MMR — 0.7 means 70% relevance, 30% diversity
            mmr_overlap_threshold: hard dedup cutoff — chunks with Jaccard > this are
                                   skipped entirely (catches adjacent-chunk copies)
        """
        self._collection = collection
        self._recall_k = recall_k
        self._mmr_lambda = mmr_lambda
        self._mmr_overlap_threshold = mmr_overlap_threshold
        self._embed = EmbeddingModel(embed_model)
        self._bm25 = BM25Index(collection)
        self._reranker = CrossEncoderReranker(reranker_model)

    def retrieve(
        self,
        query: str,
        top_n: int = 3,
        filters: MetadataFilters | None = None,
    ) -> list[RetrievedChunk]:
        """
        Full pipeline: filters → recall → RRF → rerank → MMR → top_n.

        Args:
            query:    user question; metadata filters extracted automatically
            top_n:    final number of diverse, relevant chunks to return
            filters:  override auto-extracted filters (pass MetadataFilters() to disable)

        Returns:
            List of RetrievedChunk, MMR-selected and sorted by mmr_score descending.
            Each chunk has .rerank_score (relevance), .mmr_score (relevance−overlap),
            .rrf_score, .dense_rank, .sparse_rank for debugging.
        """
        if filters is None:
            filters = parse_filters(query)
            _log_filters(query, filters)

        dense_hits  = _dense_recall(query, self._collection, self._embed, self._recall_k, filters)
        sparse_hits = _sparse_recall(query, self._bm25, self._recall_k, self._collection, filters)
        candidates  = _reciprocal_rank_fusion(dense_hits, sparse_hits)
        reranked    = self._reranker.rerank(query, candidates)
        selected    = mmr_select(reranked, top_n,
                                 lambda_param=self._mmr_lambda,
                                 overlap_threshold=self._mmr_overlap_threshold)
        return selected

    def retrieve_with_debug(
        self,
        query: str,
        top_n: int = 3,
        filters: MetadataFilters | None = None,
    ) -> dict:
        """Same as retrieve() but returns per-stage diagnostics."""
        if filters is None:
            filters = parse_filters(query)

        dense_hits  = _dense_recall(query, self._collection, self._embed, self._recall_k, filters)
        sparse_hits = _sparse_recall(query, self._bm25, self._recall_k, self._collection, filters)
        candidates  = _reciprocal_rank_fusion(dense_hits, sparse_hits)
        reranked    = self._reranker.rerank(query, candidates)
        selected    = mmr_select(reranked, top_n,
                                 lambda_param=self._mmr_lambda,
                                 overlap_threshold=self._mmr_overlap_threshold)

        return {
            "query":                query,
            "filters":              filters,
            "dense_recall_count":   len(dense_hits),
            "sparse_recall_count":  len(sparse_hits),
            "candidate_pool_size":  len(candidates),
            "after_rerank":         reranked,
            "top_n":                selected,   # MMR-selected final output
        }


def _log_filters(query: str, filters: MetadataFilters) -> None:
    active = {k: v for k, v in vars(filters).items() if v is not None}
    if active:
        print(f"[Retriever] filters from query: {active}", flush=True)
    else:
        print("[Retriever] no metadata filters extracted — broad search", flush=True)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    VECTORDB = ROOT / "data" / "vectordb"

    if not VECTORDB.exists():
        print(f"VectorDB not found at {VECTORDB}. Run the data pipeline first.")
        sys.exit(1)

    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(VECTORDB))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name="financial_docs", embedding_function=ef)

    retriever = HybridRetriever(collection, recall_k=20)

    test_cases = [
        # (question, expected_filter_summary)
        ("What was Apple's revenue in fiscal year 2023?",                "ticker=AAPL, year=2023"),
        ("How did Microsoft describe its AI strategy in the 10-K?",      "ticker=MSFT, filing_type=10-K"),
        ("NVIDIA supply chain risks",                                     "ticker=NVDA"),
        ("Compare Apple and Microsoft revenue in 2022",                   "no ticker (multi), year=2022"),
        ("What are the main risks disclosed in recent 10-K filings?",    "filing_type=10-K"),
    ]

    for q, expected in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"Expected filters: {expected}")
        debug = retriever.retrieve_with_debug(q, top_n=3)
        f = debug["filters"]
        print(f"Parsed filters:   ticker={f.ticker}, filing_type={f.filing_type}, year={f.year}")
        print(f"Dense recall: {debug['dense_recall_count']}  "
              f"Sparse recall: {debug['sparse_recall_count']}  "
              f"Pool: {debug['candidate_pool_size']}")
        print("Top results after rerank:")
        for i, c in enumerate(debug["top_n"], 1):
            meta = c.metadata
            print(f"  [{i}] rerank={c.rerank_score:.3f}  rrf={c.rrf_score:.4f}  "
                  f"dense={c.dense_rank}  sparse={c.sparse_rank}")
            print(f"       {meta.get('ticker','?')} {meta.get('filing_type','?')} "
                  f"{meta.get('date','?')}  section={meta.get('section','?')[:30]}")
            print(f"       {c.text[:100]}...")

    # Test filter parsing in isolation
    print(f"\n{'='*60}")
    print("Filter parsing unit tests:")
    parse_tests = [
        ("What was AAPL revenue in FY2023?",          MetadataFilters("AAPL", None, "2023")),
        ("Show MSFT 10-K from 2022",                  MetadataFilters("MSFT", "10-K", "2022")),
        ("Apple and Google revenue comparison",        MetadataFilters(None, None, None)),  # multi-ticker
        ("How do companies handle AI?",                MetadataFilters(None, None, None)),
        ("Tesla 8-K filing in 2023",                  MetadataFilters("TSLA", "8-K", "2023")),
    ]
    all_pass = True
    for q, expected_f in parse_tests:
        got = parse_filters(q)
        ok = (got.ticker == expected_f.ticker and
              got.filing_type == expected_f.filing_type and
              got.year == expected_f.year)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] '{q[:50]}' → ticker={got.ticker}, "
              f"filing_type={got.filing_type}, year={got.year}")
    print(f"\nFilter parsing: {'ALL PASS' if all_pass else 'SOME FAILED'}")
