# skills/metadata_store.py
# ============================================================
# Stores, persists, and queries skills metadata alongside FAISS.
#
# Architecture decision:
#   FAISS handles vector similarity. This store handles structured
#   metadata filtering. Both are indexed by the same integer chunk ID
#   so they compose naturally.
# ============================================================

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class SkillsMetadataStore:
    """
    Persists and queries skills metadata for chunks and documents.

    Storage layout (inside vectorstore/):
        skills_chunk_meta.pkl    — per-chunk skills metadata list
        skills_doc_profiles.json — per-document aggregated profiles
        skills_index.json        — inverted index: skill → [chunk_ids]

    All metadata is keyed by chunk list position (same index as FAISS).
    """

    CHUNK_META_FILE  = "vectorstore/skills_chunk_meta.pkl"
    DOC_PROFILE_FILE = "vectorstore/skills_doc_profiles.json"
    SKILLS_INDEX_FILE = "vectorstore/skills_index.json"

    def __init__(self):
        self.chunk_meta: List[Dict[str, Any]] = []
        self.doc_profiles: Dict[str, Dict[str, Any]] = {}
        # Inverted index: lowercase skill → set of chunk indices
        self._skill_index: Dict[str, Set[int]] = {}

    # ------------------------------------------------------------------
    # BUILD from enriched chunks
    # ------------------------------------------------------------------

    def build_from_chunks(
        self,
        enriched_chunks: List[Dict[str, Any]],
        doc_profiles: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """
        Populate the store from a list of skill-enriched chunks.

        Args:
            enriched_chunks: Chunks with "skills_metadata" key attached.
            doc_profiles:    Optional document-level profiles dict.
        """
        self.chunk_meta = []
        self._skill_index = {}

        for idx, chunk in enumerate(enriched_chunks):
            meta = chunk.get("skills_metadata", {})
            self.chunk_meta.append(meta)

            # Build inverted index
            for skill in meta.get("skills", []):
                key = skill.lower().strip()
                if key not in self._skill_index:
                    self._skill_index[key] = set()
                self._skill_index[key].add(idx)

            for topic in meta.get("topics", []):
                key = topic.lower().strip()
                if key not in self._skill_index:
                    self._skill_index[key] = set()
                self._skill_index[key].add(idx)

        if doc_profiles:
            self.doc_profiles = doc_profiles

        print(f"[SkillsMetadataStore] Built. {len(self.chunk_meta)} chunks, "
              f"{len(self._skill_index)} unique skills/topics indexed.")
        self.save()

    # ------------------------------------------------------------------
    # FILTERING (used during retrieval)
    # ------------------------------------------------------------------

    def filter_by_skill(self, skill: str) -> Set[int]:
        """Return chunk indices that contain the given skill (partial match)."""
        skill_lower = skill.lower().strip()
        matching = set()
        for indexed_skill, ids in self._skill_index.items():
            if skill_lower in indexed_skill or indexed_skill in skill_lower:
                matching.update(ids)
        return matching

    def filter_by_domain(self, domain: str) -> Set[int]:
        """Return chunk indices whose domain matches (partial match)."""
        domain_lower = domain.lower().strip()
        return {
            i for i, m in enumerate(self.chunk_meta)
            if domain_lower in m.get("domain", "").lower()
        }

    def filter_by_difficulty(self, difficulty: str) -> Set[int]:
        """Return chunk indices matching a difficulty level exactly."""
        difficulty_lower = difficulty.lower().strip()
        return {
            i for i, m in enumerate(self.chunk_meta)
            if m.get("difficulty", "").lower() == difficulty_lower
        }

    def apply_filters(
        self,
        skill: Optional[str] = None,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Optional[Set[int]]:
        """
        Combine multiple filters with AND logic.
        Returns None if no filters are applied (meaning: use all chunks).
        """
        result: Optional[Set[int]] = None

        def intersect(current, new_set):
            return new_set if current is None else current & new_set

        if skill:
            result = intersect(result, self.filter_by_skill(skill))
        if domain:
            result = intersect(result, self.filter_by_domain(domain))
        if difficulty:
            result = intersect(result, self.filter_by_difficulty(difficulty))

        return result

    # ------------------------------------------------------------------
    # ACCESSORS
    # ------------------------------------------------------------------

    def get_chunk_skills(self, chunk_idx: int) -> Dict[str, Any]:
        """Return the skills metadata for a specific chunk index."""
        if 0 <= chunk_idx < len(self.chunk_meta):
            return self.chunk_meta[chunk_idx]
        return {}

    def get_document_profile(self, source: str) -> Dict[str, Any]:
        """Return the aggregated document-level skills profile."""
        return self.doc_profiles.get(source, {})

    def get_all_skills(self) -> List[str]:
        """Return a sorted list of all unique skills across all chunks."""
        return sorted(self._skill_index.keys())

    def get_all_domains(self) -> List[str]:
        """Return a sorted list of all unique domains."""
        domains = {m.get("domain", "") for m in self.chunk_meta if m.get("domain")}
        return sorted(domains)

    def get_all_difficulties(self) -> List[str]:
        """Return the unique difficulty levels present in the store."""
        difficulties = {m.get("difficulty", "") for m in self.chunk_meta if m.get("difficulty")}
        return sorted(difficulties)

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self) -> None:
        os.makedirs("vectorstore", exist_ok=True)
        with open(self.CHUNK_META_FILE, "wb") as f:
            pickle.dump(self.chunk_meta, f)

        with open(self.DOC_PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.doc_profiles, f, ensure_ascii=False, indent=2)

        # Serialize inverted index (sets → lists for JSON)
        serializable_index = {k: list(v) for k, v in self._skill_index.items()}
        with open(self.SKILLS_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_index, f, ensure_ascii=False, indent=2)

        print(f"[SkillsMetadataStore] Saved to vectorstore/")

    def load(self) -> bool:
        """Load persisted metadata. Returns True if successful."""
        try:
            with open(self.CHUNK_META_FILE, "rb") as f:
                self.chunk_meta = pickle.load(f)

            with open(self.DOC_PROFILE_FILE, "r", encoding="utf-8") as f:
                self.doc_profiles = json.load(f)

            with open(self.SKILLS_INDEX_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self._skill_index = {k: set(v) for k, v in raw.items()}

            print(f"[SkillsMetadataStore] Loaded. {len(self.chunk_meta)} chunks, "
                  f"{len(self._skill_index)} skills indexed.")
            return True
        except FileNotFoundError:
            return False

    def is_loaded(self) -> bool:
        return len(self.chunk_meta) > 0
