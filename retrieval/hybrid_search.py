# retrieval/hybrid_search.py
# ============================================================
# Hybrid search: BM25 (keyword) + FAISS (semantic) combined
# via Reciprocal Rank Fusion (RRF).
#
# Why RRF?
#   - No need to tune alpha weights between systems
#   - Robust to score magnitude differences (BM25 vs cosine similarity)
#   - Works well empirically across many retrieval tasks
#
# Dependencies: rank_bm25  (pip install rank-bm25)
# ============================================================

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import faiss


class HybridSearchEngine:
    """
    Combines BM25 sparse retrieval with FAISS dense retrieval
    using Reciprocal Rank Fusion (RRF).

    Fusion formula:
        RRF_score(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the rank of document d in system i,
    and k=60 is a stabilizing constant (standard default).

    Usage:
        hybrid = HybridSearchEngine(chunks)
        hybrid.build()
        results = hybrid.search(query, faiss_index, embedding_model, top_k=5)
    """

    RRF_K = 60   # Standard RRF constant

    def __init__(self, chunks: List[Dict[str, Any]], language: str = "spanish"):
        self.chunks = chunks
        self.language = language
        self._bm25 = None

    def build(self) -> None:
        """Build the BM25 index from chunk texts."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank-bm25: pip install rank-bm25")

        tokenized = [self._tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)
        print(f"[HybridSearch] BM25 index built over {len(self.chunks)} chunks.")

    def search(
        self,
        query: str,
        faiss_index: faiss.Index,
        embedding_model,
        top_k: int = 5,
        bm25_candidates: int = 50,
        faiss_candidates: int = 50,
        allowed_indices: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and return fused, ranked results.

        Args:
            query:             Natural language query.
            faiss_index:       Built FAISS index.
            embedding_model:   SentenceTransformer instance.
            top_k:             Number of results to return.
            bm25_candidates:   How many BM25 results to consider for fusion.
            faiss_candidates:  How many FAISS results to consider for fusion.
            allowed_indices:   Optional set of chunk indices to restrict search to
                               (from skills metadata filtering).

        Returns:
            List of chunk dicts with "hybrid_score" added.
        """
        if self._bm25 is None:
            raise RuntimeError("Call build() before search().")

        # --- BM25 retrieval ---
        bm25_scores = self._bm25_search(query, bm25_candidates)

        # --- FAISS dense retrieval ---
        faiss_scores = self._faiss_search(
            query, faiss_index, embedding_model, faiss_candidates
        )

        # --- Apply metadata filter if provided ---
        if allowed_indices is not None:
            bm25_scores  = [(i, s) for i, s in bm25_scores  if i in allowed_indices]
            faiss_scores = [(i, s) for i, s in faiss_scores if i in allowed_indices]

        # --- Reciprocal Rank Fusion ---
        fused = self._rrf_fusion(bm25_scores, faiss_scores)

        # --- Build result list ---
        results = []
        for idx, score in fused[:top_k]:
            chunk = dict(self.chunks[idx])
            chunk["hybrid_score"]  = round(score, 6)
            chunk["relevance_score"] = round(score, 6)
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, n: int) -> List[tuple]:
        """Return list of (chunk_idx, bm25_score) sorted by score desc."""
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Get top-n indices by score
        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def _faiss_search(self, query: str, index, model, n: int) -> List[tuple]:
        """Return list of (chunk_idx, cosine_similarity) sorted desc."""
        query_vec = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        distances, indices = index.search(query_vec, n)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                # Convert L2 distance to similarity score
                results.append((int(idx), float(1 / (1 + dist))))
        return results

    def _rrf_fusion(
        self,
        list1: List[tuple],
        list2: List[tuple],
    ) -> List[tuple]:
        """
        Apply Reciprocal Rank Fusion to two ranked lists.
        Returns merged list sorted by RRF score descending.
        """
        scores: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(list1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        for rank, (idx, _) in enumerate(list2):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer with stopword removal."""
        STOPWORDS = {
            "de", "la", "el", "en", "y", "a", "que", "los", "las", "del",
            "un", "una", "con", "por", "para", "se", "al", "es", "su",
            "the", "of", "and", "to", "in", "a", "is", "that", "for",
        }
        tokens = re.findall(r'\b[a-záéíóúüñ]{2,}\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS]
