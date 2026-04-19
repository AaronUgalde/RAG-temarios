"""
===============================================================================
rag_engine_v2.py  —  Skills Intelligence RAG Engine
===============================================================================
Upgrades over v1:
  ✓ Skill extraction + metadata filtering
  ✓ Hybrid search (BM25 + FAISS via RRF)
  ✓ Cross-encoder re-ranking
  ✓ Context compression (embedding-based)
  ✓ Fully backward-compatible with existing app.py

QUICK-START (drop-in replacement):
    from rag_engine_v2 import RAGEngineV2
    engine = RAGEngineV2(api_key=..., model=..., top_k=5)
    # All existing methods (build_index, load_index, query) still work.
===============================================================================
"""

import os
import pickle
import numpy as np
import faiss
import httpx

from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from document_loader import DocumentLoader
from skills.extractor import SkillExtractor
from skills.metadata_store import SkillsMetadataStore
from retrieval.hybrid_search import HybridSearchEngine
from retrieval.reranker import CrossEncoderReranker
from retrieval.compressor import ContextCompressor


class RAGEngineV2:
    """
    Skills-aware RAG engine with hybrid search, re-ranking, and compression.

    Pipeline (per query):
      1. Metadata filter  → restrict FAISS search space by skill/domain/difficulty
      2. Hybrid retrieval → BM25 + FAISS fused with RRF
      3. Re-ranking       → cross-encoder scores (query, passage) pairs
      4. Compression      → keep only query-relevant sentences per chunk
      5. Generation       → LLM call with compressed, enriched context
    """

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    INDEX_FILE   = "vectorstore/faiss_index.bin"
    CHUNKS_FILE  = "vectorstore/chunks_metadata.pkl"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        use_compression: bool = True,
        rerank_top_n: Optional[int] = None,
        compression_strategy: str = "sentence",  # "sentence" | "llm" | "none"
    ):
        print("[RAGEngineV2] Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)

        # Explicitly pass our own httpx.Client so the OpenAI SDK never tries
        # to set the `proxies` kwarg that was removed in httpx >= 0.28.
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            http_client=httpx.Client(),
        )
        self.api_key = api_key
        self.model   = model
        self.top_k   = top_k

        # Feature flags
        self.use_hybrid      = use_hybrid
        self.use_reranker    = use_reranker
        self.use_compression = use_compression
        self.rerank_top_n    = rerank_top_n or top_k

        # Core components
        self.index:  Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]]  = []

        self.skills_store = SkillsMetadataStore()
        self.hybrid_engine: Optional[HybridSearchEngine] = None

        # Lazy-init reranker (downloads ~80MB model on first use)
        self._reranker: Optional[CrossEncoderReranker] = None

        self.compressor = ContextCompressor(
            embedding_model=self.embedding_model,
            strategy=compression_strategy,
        )

        print(f"[RAGEngineV2] Ready. Model={model} | hybrid={use_hybrid} | "
              f"rerank={use_reranker} | compress={use_compression}")

    # =========================================================================
    # INDEXING
    # =========================================================================

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        extract_skills: bool = False,
        skill_model: str = "openai/gpt-4o-mini",
    ) -> None:
        """
        Build FAISS index + optional skill extraction.

        Args:
            chunks:         Chunks from DocumentLoader (must have "text").
            extract_skills: If True, calls LLM to extract skills per chunk.
                            Set False for fast re-indexing without skill refresh.
            skill_model:    Model to use for skill extraction.
        """
        if not chunks:
            raise ValueError("Empty chunk list.")

        # --- Optional skill extraction ---
        if extract_skills:
            print("[RAGEngineV2] Extracting skills from chunks...")
            extractor = SkillExtractor(
                api_key=self.api_key,
                model=skill_model,
            )
            chunks = extractor.extract_batch(chunks)
            doc_profiles = extractor.aggregate_document_skills(chunks)
            self.skills_store.build_from_chunks(chunks, doc_profiles)
        else:
            # Try to load existing skills store
            if not self.skills_store.load():
                print("[RAGEngineV2] No skills metadata found. "
                      "Run with extract_skills=True to enable skill filtering.")

        self.chunks = chunks

        # --- Build FAISS index ---
        print(f"[RAGEngineV2] Encoding {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        # --- Build BM25 index ---
        if self.use_hybrid:
            self.hybrid_engine = HybridSearchEngine(self.chunks)
            self.hybrid_engine.build()

        self._save_index()
        print(f"[RAGEngineV2] Index built: {self.index.ntotal} vectors.")

    def _save_index(self) -> None:
        os.makedirs("vectorstore", exist_ok=True)
        faiss.write_index(self.index, self.INDEX_FILE)
        with open(self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[RAGEngineV2] Index saved.")

    def load_index(self) -> bool:
        if not os.path.exists(self.INDEX_FILE):
            return False
        self.index = faiss.read_index(self.INDEX_FILE)
        with open(self.CHUNKS_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        self.skills_store.load()

        if self.use_hybrid:
            self.hybrid_engine = HybridSearchEngine(self.chunks)
            self.hybrid_engine.build()

        print(f"[RAGEngineV2] Loaded: {self.index.ntotal} vectors.")
        return True

    def is_index_built(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    # =========================================================================
    # RETRIEVAL  (the upgraded core)
    # =========================================================================

    def retrieve(
        self,
        query: str,
        carrera_filter: Optional[str] = None,
        skill_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using the full upgraded pipeline.

        Step 1 — Metadata filtering:
            Compute the set of allowed chunk indices from skills metadata.
        Step 2 — Hybrid or FAISS search:
            BM25 + FAISS fused via RRF (or plain FAISS if hybrid disabled).
        Step 3 — Re-ranking:
            Cross-encoder scores each (query, chunk) pair.
        Step 4 — Context compression:
            Strip irrelevant sentences from each chunk.
        """
        if not self.is_index_built():
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        # ------ Step 1: build allowed index set from all active filters ------
        allowed = None

        if self.skills_store.is_loaded():
            allowed = self.skills_store.apply_filters(
                skill=skill_filter,
                domain=domain_filter,
                difficulty=difficulty_filter,
            )

        # carrera_filter is a legacy filter on chunk metadata, not skills store
        if carrera_filter:
            carrera_set = {
                i for i, c in enumerate(self.chunks)
                if carrera_filter.lower() in c.get("carrera", "").lower()
            }
            allowed = carrera_set if allowed is None else allowed & carrera_set

        # ------ Step 2: retrieval ------
        fetch_k = self.top_k * 4  # fetch more candidates for reranking

        if self.use_hybrid and self.hybrid_engine:
            results = self.hybrid_engine.search(
                query=query,
                faiss_index=self.index,
                embedding_model=self.embedding_model,
                top_k=fetch_k,
                allowed_indices=allowed,
            )
        else:
            results = self._faiss_search(query, fetch_k, allowed)

        if not results:
            return []

        # ------ Step 3: re-ranking ------
        if self.use_reranker:
            if self._reranker is None:
                self._reranker = CrossEncoderReranker()
            results = self._reranker.rerank(query, results, top_n=self.top_k)
        else:
            results = results[:self.top_k]

        # ------ Step 4: context compression ------
        if self.use_compression:
            results = self.compressor.compress(query, results)

        return results

    def _faiss_search(
        self,
        query: str,
        k: int,
        allowed: Optional[set],
    ) -> List[Dict[str, Any]]:
        """Plain FAISS search with optional index restriction."""
        q_vec = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)

        search_k = k * 3 if allowed else k
        distances, indices = self.index.search(q_vec, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if allowed is not None and idx not in allowed:
                continue
            chunk = dict(self.chunks[idx])
            chunk["relevance_score"] = float(1 / (1 + dist))
            results.append(chunk)
            if len(results) >= k:
                break
        return results

    # =========================================================================
    # GENERATION
    # =========================================================================

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate an answer from the LLM given retrieved context."""

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            skills_meta = chunk.get("skills_metadata", {})
            skills_str  = ", ".join(skills_meta.get("skills", [])[:5])
            domain_str  = skills_meta.get("domain", "")
            score_str   = f"{chunk.get('relevance_score', 0):.2f}"

            header = (
                f"[Fuente {i}] {chunk['carrera']} | {chunk['source']} | "
                f"Pág. {chunk['page']} | Score: {score_str}"
            )
            if domain_str:
                header += f" | Dominio: {domain_str}"
            if skills_str:
                header += f" | Skills: {skills_str}"

            context_parts.append(f"{header}\n{chunk['text']}")

        context_block = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "Eres un asistente académico especializado en planes de estudio y "
            "temarios universitarios. Tu función es ayudar a estudiantes, docentes "
            "y coordinadores a consultar información sobre los contenidos de diversas "
            "carreras universitarias.\n\n"
            "INSTRUCCIONES:\n"
            "1. Responde ÚNICAMENTE con la información del contexto proporcionado.\n"
            "2. Si la respuesta no está en el contexto, indícalo explícitamente.\n"
            "3. Cita siempre la carrera y fuente de donde proviene la información.\n"
            "4. Usa lenguaje académico claro y estructurado.\n"
            "5. Al comparar múltiples carreras, organiza la información en tablas o listas."
        )

        user_message = (
            f"Fragmentos relevantes de los temarios universitarios:\n\n"
            f"{context_block}\n\n---\n\n"
            f"Consulta: {query}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1500,
            messages=messages,
        )
        return response.choices[0].message.content, retrieved_chunks

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def query(
        self,
        user_question: str,
        carrera_filter: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        skill_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end RAG: retrieve + generate.
        All parameters except user_question are optional filters.
        """
        retrieved = self.retrieve(
            query=user_question,
            carrera_filter=carrera_filter,
            skill_filter=skill_filter,
            domain_filter=domain_filter,
            difficulty_filter=difficulty_filter,
        )

        if not retrieved:
            return {
                "answer": (
                    "No encontré información relevante en los temarios disponibles. "
                    "Reformule su consulta o verifique que los documentos estén cargados."
                ),
                "sources": [],
            }

        answer, sources = self.generate_response(
            query=user_question,
            retrieved_chunks=retrieved,
            conversation_history=conversation_history,
        )
        return {"answer": answer, "sources": sources}

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_index_stats(self) -> Dict[str, Any]:
        if not self.is_index_built():
            return {"status": "No index built."}

        carreras = sorted(set(c["carrera"] for c in self.chunks))
        fuentes  = sorted(set(c["source"]  for c in self.chunks))
        stats = {
            "total_fragmentos":     len(self.chunks),
            "total_vectores_faiss": self.index.ntotal,
            "carreras":             carreras,
            "archivos":             fuentes,
        }
        if self.skills_store.is_loaded():
            stats["skills_indexed"] = len(self.skills_store.get_all_skills())
            stats["domains"]        = self.skills_store.get_all_domains()
            stats["difficulties"]   = self.skills_store.get_all_difficulties()
        return stats
