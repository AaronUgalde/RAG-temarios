# evaluation/dataset.py
# ============================================================
# Evaluation dataset management.
#
# Two modes:
#   1. MANUAL: load a hand-crafted JSONL file
#   2. AUTO-GENERATE: use an LLM to create queries from chunks
#
# Dataset format (JSONL, one entry per line):
#   {
#     "query": "¿Qué materias de álgebra lineal hay en la carrera de sistemas?",
#     "relevant_sources": ["temario_ingenieria_sistemas_computacionales.pdf"],
#     "relevant_chunk_ids": [12, 13, 45],  // optional — for chunk-level eval
#     "difficulty": "intermediate",         // optional metadata
#     "domain": "mathematics"              // optional metadata
#   }
# ============================================================

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


DATASET_GEN_SYSTEM = """\
You are an expert at generating realistic evaluation queries for university course RAG systems.
You will receive a syllabus text fragment and must generate diverse, natural-language questions
that a student or professor might ask. Questions should vary in specificity and difficulty.
Always respond with VALID JSON only.
"""

DATASET_GEN_PROMPT = """\
Given this university syllabus fragment:
\"\"\"
{chunk_text}
\"\"\"

Source document: {source}

Generate {n_queries} diverse questions that this fragment could answer.
Respond ONLY with a JSON array:
[
  {{
    "query": "question in natural language",
    "expected_answer_hint": "brief hint about what the answer should contain"
  }},
  ...
]
"""


class EvaluationDataset:
    """
    Manages evaluation queries and their ground-truth relevance labels.

    Usage:
        # Load from file
        ds = EvaluationDataset.from_file("evaluation/eval_queries.jsonl")

        # Auto-generate with LLM
        ds = EvaluationDataset.auto_generate(chunks, api_key=..., n_per_chunk=2)

        # Iterate
        for item in ds:
            query = item["query"]
            relevant = set(item["relevant_sources"])
    """

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    # ------------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "EvaluationDataset":
        """Load dataset from a JSONL file (one JSON object per line)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        print(f"[EvaluationDataset] Loaded {len(items)} queries from {path}")
        return cls(items)

    @classmethod
    def auto_generate(
        cls,
        chunks: List[Dict[str, Any]],
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        n_per_chunk: int = 2,
        max_chunks: int = 50,
        output_path: Optional[str] = "evaluation/eval_queries.jsonl",
    ) -> "EvaluationDataset":
        """
        Auto-generate evaluation queries from chunks using an LLM.

        Samples `max_chunks` chunks (spread across documents), generates
        `n_per_chunk` queries per chunk, and tags each with the source doc.

        Args:
            chunks:      The indexed chunks (must have "text" and "source").
            api_key:     OpenRouter API key.
            model:       Model to use for generation.
            n_per_chunk: Queries to generate per chunk.
            max_chunks:  Max number of chunks to sample (cost control).
            output_path: Save generated dataset here (JSONL).

        Returns:
            EvaluationDataset instance ready for evaluation.
        """
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Sample chunks spread across all documents
        sampled = cls._sample_chunks(chunks, max_chunks)
        all_items = []

        for i, chunk in enumerate(sampled):
            print(f"  [DatasetGen] Generating queries {i+1}/{len(sampled)}: {chunk['source']}")
            prompt = DATASET_GEN_PROMPT.format(
                chunk_text=chunk["text"][:1500],
                source=chunk["source"],
                n_queries=n_per_chunk,
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=600,
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": DATASET_GEN_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                generated = json.loads(raw)
                for g in generated:
                    all_items.append({
                        "query": g["query"],
                        "expected_answer_hint": g.get("expected_answer_hint", ""),
                        "relevant_sources": [chunk["source"]],
                        "relevant_chunk_ids": [chunk["chunk_id"]],
                        "carrera": chunk.get("carrera", ""),
                    })
            except Exception as e:
                print(f"    [WARN] Failed for chunk {chunk['chunk_id']}: {e}")

            time.sleep(0.5)

        dataset = cls(all_items)
        if output_path:
            dataset.save(output_path)
        return dataset

    @staticmethod
    def _sample_chunks(
        chunks: List[Dict], max_chunks: int
    ) -> List[Dict]:
        """Sample chunks evenly from all source documents."""
        from collections import defaultdict
        import random

        by_source: Dict[str, List] = defaultdict(list)
        for c in chunks:
            by_source[c["source"]].append(c)

        sampled = []
        per_doc = max(1, max_chunks // len(by_source))
        for source_chunks in by_source.values():
            take = min(per_doc, len(source_chunks))
            sampled.extend(random.sample(source_chunks, take))

        return sampled[:max_chunks]

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the dataset as JSONL."""
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[EvaluationDataset] Saved {len(self.items)} queries to {path}")
