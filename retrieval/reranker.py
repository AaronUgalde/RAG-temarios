# retrieval/reranker.py
# ============================================================
# Re-ranking layer: improves top-K precision by scoring each
# (query, chunk) pair more accurately than the initial retrieval.
#
# Two implementations provided:
#   1. CrossEncoderReranker — local model, fast, no API cost
#   2. LLMReranker          — uses OpenRouter, slower but flexible
#
# When to use which:
#   - CrossEncoder: production default (ms latency, free)
#   - LLMReranker:  when you need nuanced academic understanding
# ============================================================

import time
import httpx
from typing import Any, Dict, List, Optional

from openai import OpenAI


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model.

    A cross-encoder takes BOTH the query and the passage as a single
    input and produces a single relevance score — far more accurate than
    bi-encoder (embedding) similarity, at the cost of speed.

    Recommended model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    (fast, ~80MB, excellent multilingual zero-shot performance)

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, chunks, top_n=5)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                print(f"[CrossEncoderReranker] Loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "pip install sentence-transformers"
                )

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks by cross-encoder relevance score.

        Args:
            query:  The user query string.
            chunks: List of retrieved chunk dicts (must have "text").
            top_n:  How many to return (None = return all, re-ranked).

        Returns:
            Chunks sorted by cross-encoder score, with "rerank_score" key added.
        """
        if not chunks:
            return chunks

        self._load()

        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored:
            enriched = dict(chunk)
            enriched["rerank_score"] = float(score)
            enriched["relevance_score"] = float(score)  # overwrite for downstream use
            results.append(enriched)

        return results[:top_n] if top_n else results


class LLMReranker:
    """
    Re-ranks chunks using an LLM to score each (query, passage) pair.

    Slower than CrossEncoder but useful when:
    - You need domain-specific understanding
    - You want explanations for why a chunk is relevant
    - You're already paying for LLM calls

    Scoring prompt asks the model to return a relevance score 1-10.
    """

    SYSTEM_PROMPT = """\
You are a precise relevance judge for an academic retrieval system.
Given a query and a document passage, score how relevant the passage is
for answering the query. Respond ONLY with JSON: {"score": int, "reason": "brief"}
Score 1-10: 1=completely irrelevant, 10=perfectly answers the query.
"""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             http_client=httpx.Client())
        self.model = model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        delay: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Score and re-rank all chunks. Makes one LLM call per chunk.

        Args:
            query:  The user query.
            chunks: Retrieved chunks to re-rank.
            top_n:  Return only top N after re-ranking.
            delay:  Seconds to wait between API calls (rate limiting).

        Returns:
            Chunks sorted by LLM relevance score, with "rerank_score" added.
        """
        import json

        scored_chunks = []
        for chunk in chunks:
            prompt = (
                f"QUERY: {query}\n\n"
                f"PASSAGE: {chunk['text'][:800]}\n\n"
                "How relevant is this passage for answering the query?"
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=100,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                parsed = json.loads(raw)
                score = float(parsed.get("score", 5)) / 10.0
                reason = parsed.get("reason", "")
            except Exception:
                score = 0.5
                reason = "scoring failed"

            enriched = dict(chunk)
            enriched["rerank_score"]   = score
            enriched["rerank_reason"]  = reason
            enriched["relevance_score"] = score
            scored_chunks.append(enriched)

            time.sleep(delay)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_chunks[:top_n] if top_n else scored_chunks
