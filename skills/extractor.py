# skills/extractor.py
# ============================================================
# LLM-based skill extraction from syllabus chunks.
#
# Design decisions:
#   - Batched processing with configurable concurrency
#   - Graceful fallback on parse errors (returns empty skills)
#   - Caches results to avoid re-calling the LLM on unchanged chunks
#   - Uses the same OpenRouter client as the rest of the system
# ============================================================

import json
import time
import hashlib
import logging
import httpx
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .prompts import (
    SKILL_EXTRACTION_SYSTEM,
    SKILL_EXTRACTION_PROMPT,
    DOCUMENT_SKILL_AGGREGATION_PROMPT,
)

logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Extracts structured skills from syllabus chunks using an LLM.

    Usage:
        extractor = SkillExtractor(api_key="sk-or-...", model="openai/gpt-4o-mini")
        chunks_with_skills = extractor.extract_batch(chunks)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 10,
        retry_delay: float = 1.0,
        max_retries: int = 3,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             http_client=httpx.Client())
        self.model = model
        self.batch_size = batch_size
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self._cache: Dict[str, Dict] = {}   # chunk_hash → skills dict

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def extract_batch(
        self, chunks: List[Dict[str, Any]], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all chunks and attach extracted skills as metadata.

        Args:
            chunks: List of chunk dicts (must have "text" key).
            show_progress: Print progress to stdout.

        Returns:
            Same list with a "skills_metadata" key added to each dict.
        """
        total = len(chunks)
        enriched = []

        for i, chunk in enumerate(chunks):
            if show_progress and i % 10 == 0:
                print(f"  [SkillExtractor] {i}/{total} chunks processed...")

            skills_meta = self._extract_single(chunk["text"])
            enriched_chunk = {**chunk, "skills_metadata": skills_meta}
            enriched.append(enriched_chunk)

            # Small delay to respect rate limits
            if i > 0 and i % self.batch_size == 0:
                time.sleep(self.retry_delay)

        print(f"  [SkillExtractor] Done. {total} chunks enriched with skills.")
        return enriched

    def aggregate_document_skills(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate per-chunk skills into a single document-level profile.
        Groups chunks by source document and calls the LLM once per doc.

        Returns:
            Dict mapping source filename → aggregated skills profile.
        """
        from collections import defaultdict
        import json as _json

        docs: Dict[str, List] = defaultdict(list)
        for c in chunks:
            source = c.get("source", "unknown")
            meta = c.get("skills_metadata", {})
            if meta.get("skills"):
                docs[source].append(meta)

        doc_profiles: Dict[str, Dict] = {}
        for source, chunk_skills_list in docs.items():
            chunk_skills_json = _json.dumps(chunk_skills_list, ensure_ascii=False)
            prompt = DOCUMENT_SKILL_AGGREGATION_PROMPT.format(
                chunk_skills_json=chunk_skills_json
            )
            profile = self._call_llm(prompt)
            doc_profiles[source] = profile
            print(f"  [SkillExtractor] Document profile built: {source}")

        return doc_profiles

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _extract_single(self, text: str) -> Dict[str, Any]:
        """Extract skills from one chunk text, using cache if available."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = SKILL_EXTRACTION_PROMPT.format(chunk_text=text[:2000])
        result = self._call_llm(prompt)
        self._cache[cache_key] = result
        return result

    def _call_llm(self, user_prompt: str) -> Dict[str, Any]:
        """
        Call the LLM with retry logic. Returns parsed JSON or fallback dict.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0.0,   # Deterministic for structured extraction
                    messages=[
                        {"role": "system", "content": SKILL_EXTRACTION_SYSTEM},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                # Strip markdown fences if model added them
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return json.loads(raw)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))

        # Fallback: return empty skills metadata
        return {
            "skills": [],
            "difficulty": "unknown",
            "domain": "unknown",
            "topics": [],
            "prerequisites": [],
        }
