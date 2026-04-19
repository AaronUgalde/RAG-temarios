# retrieval/compressor.py
# ============================================================
# Context compression: strips irrelevant sentences from chunks
# before they are sent to the LLM.
#
# Why this matters:
#   - LLM context windows are finite and expensive
#   - Irrelevant sentences add noise and hurt answer quality
#   - Studies show compressed context → better faithfulness
#
# Two strategies:
#   1. SentenceScorer    — local, embedding-based sentence filtering
#   2. LLMCompressor     — LLM extracts only the relevant sentences
# ============================================================

import re
from typing import Any, Dict, List, Optional


class SentenceScorer:
    """
    Compresses chunks by keeping only sentences that are semantically
    similar to the query, using embedding cosine similarity.

    This is the recommended default: fast, free, works offline.

    Usage:
        compressor = SentenceScorer(embedding_model)
        compressed = compressor.compress(query, chunks, threshold=0.3)
    """

    def __init__(self, embedding_model, min_sentences: int = 2):
        """
        Args:
            embedding_model: A SentenceTransformer instance (already loaded).
            min_sentences:   Always keep at least this many sentences per chunk.
        """
        self.model = embedding_model
        self.min_sentences = min_sentences

    def compress(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        threshold: float = 0.3,
        max_sentences_per_chunk: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Compress each chunk to its most query-relevant sentences.

        Args:
            query:                   The user query.
            chunks:                  List of chunk dicts.
            threshold:               Minimum cosine similarity to keep a sentence.
            max_sentences_per_chunk: Hard cap on sentences kept per chunk.

        Returns:
            Chunks with "text" replaced by compressed text.
            Original text preserved in "original_text".
        """
        import numpy as np

        query_vec = self.model.encode([query], convert_to_numpy=True)

        compressed_chunks = []
        for chunk in chunks:
            sentences = self._split_sentences(chunk["text"])
            if len(sentences) <= self.min_sentences:
                # Chunk is already short — keep as-is
                compressed_chunks.append(dict(chunk))
                continue

            sent_vecs = self.model.encode(sentences, convert_to_numpy=True)

            # Cosine similarity: dot product of normalized vectors
            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            s_norm = sent_vecs / (np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-9)
            sims = (s_norm @ q_norm.T).flatten()

            # Keep sentences above threshold, enforce min/max
            keep_idx = [i for i, s in enumerate(sims) if s >= threshold]
            if len(keep_idx) < self.min_sentences:
                # Fall back to top-N by score
                keep_idx = np.argsort(sims)[::-1][:self.min_sentences].tolist()
            keep_idx = sorted(keep_idx[:max_sentences_per_chunk])

            kept_text = " ".join(sentences[i] for i in keep_idx)

            enriched = dict(chunk)
            enriched["original_text"]   = chunk["text"]
            enriched["text"]            = kept_text
            enriched["compression_ratio"] = round(len(kept_text) / max(len(chunk["text"]), 1), 2)
            compressed_chunks.append(enriched)

        return compressed_chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Split on . ! ? followed by whitespace or end of string
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter empty and very short fragments
        return [s.strip() for s in parts if len(s.strip()) > 20]


class LLMCompressor:
    """
    Uses an LLM to extract only the sentences from each chunk that
    are directly relevant to the query.

    More accurate than embedding-based compression but slower and costs tokens.
    Recommended for high-stakes queries where quality matters most.

    Usage:
        compressor = LLMCompressor(api_key=..., model="openai/gpt-4o-mini")
        compressed = compressor.compress(query, chunks)
    """

    SYSTEM_PROMPT = """\
You are a context compression assistant for a university RAG system.
Given a query and a document passage, extract ONLY the sentences that are
directly relevant to answering the query. Do not add, invent, or summarize.
Copy exact sentences from the passage. If nothing is relevant, return an empty string.
Respond with ONLY the extracted text, no explanations.
"""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        import httpx
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             http_client=httpx.Client())
        self.model = model

    def compress(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        min_chars: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Compress each chunk by extracting query-relevant sentences via LLM.

        Args:
            query:     The user query.
            chunks:    Chunks to compress.
            min_chars: If extracted text is shorter than this, keep original.

        Returns:
            Chunks with compressed "text" field.
        """
        import time

        compressed = []
        for chunk in chunks:
            prompt = (
                f"QUERY: {query}\n\n"
                f"PASSAGE:\n{chunk['text'][:1500]}\n\n"
                "Extract only the sentences relevant to the query."
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=400,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                extracted = response.choices[0].message.content.strip()
            except Exception:
                extracted = ""

            enriched = dict(chunk)
            enriched["original_text"] = chunk["text"]
            if len(extracted) >= min_chars:
                enriched["text"] = extracted
                enriched["compression_ratio"] = round(
                    len(extracted) / max(len(chunk["text"]), 1), 2
                )
            else:
                enriched["compression_ratio"] = 1.0

            compressed.append(enriched)
            time.sleep(0.1)

        return compressed


class ContextCompressor:
    """
    High-level façade that selects the appropriate compression strategy.

    Usage:
        compressor = ContextCompressor(embedding_model=model, strategy="sentence")
        compressed_chunks = compressor.compress(query, chunks)
    """

    def __init__(
        self,
        embedding_model=None,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        strategy: str = "sentence",  # "sentence" | "llm" | "none"
    ):
        """
        Args:
            embedding_model: SentenceTransformer (required for "sentence" strategy).
            api_key:         OpenRouter key (required for "llm" strategy).
            model:           LLM model name (for "llm" strategy).
            strategy:        Which compressor to use.
        """
        self.strategy = strategy

        if strategy == "sentence":
            if embedding_model is None:
                raise ValueError("embedding_model required for 'sentence' strategy")
            self._compressor = SentenceScorer(embedding_model)
        elif strategy == "llm":
            if api_key is None:
                raise ValueError("api_key required for 'llm' strategy")
            self._compressor = LLMCompressor(api_key=api_key, model=model)
        elif strategy == "none":
            self._compressor = None
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'sentence', 'llm', or 'none'.")

    def compress(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Compress chunks using the configured strategy."""
        if self._compressor is None:
            return chunks
        return self._compressor.compress(query, chunks, **kwargs)
