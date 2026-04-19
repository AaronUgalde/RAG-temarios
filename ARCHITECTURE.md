# Skills Intelligence RAG — Architecture & Upgrade Guide

## New Project Structure

```
RAG temarios/
│
├── app.py                          ← original app (unchanged, still works)
├── app_v2.py                       ← NEW: upgraded UI with all features
├── rag_engine.py                   ← original engine (unchanged)
├── rag_engine_v2.py                ← NEW: upgraded engine (drop-in replacement)
├── document_loader.py              ← unchanged
│
├── extract_skills.py               ← NEW: one-shot skill extraction script
├── run_evaluation.py               ← NEW: evaluation CLI
├── requirements_v2.txt             ← NEW: updated dependencies
│
├── skills/                         ← NEW: Component 1 — Skill Layer
│   ├── __init__.py
│   ├── prompts.py                  LLM prompt templates for extraction
│   ├── extractor.py                Batched LLM-based skill extraction
│   └── metadata_store.py           Inverted index for skills/domain/difficulty
│
├── evaluation/                     ← NEW: Component 2 — Evaluation Pipeline
│   ├── __init__.py
│   ├── metrics.py                  Recall@K, MRR, nDCG, MAP (pure Python)
│   ├── dataset.py                  Manual + auto-generated eval datasets
│   ├── runner.py                   Experiment runner with comparison
│   ├── report.py                   Console tables + CSV + LLM judge
│   └── eval_queries.jsonl          Seed dataset (20 hand-crafted queries)
│
├── retrieval/                      ← NEW: Component 3 — Context Engineering
│   ├── __init__.py
│   ├── hybrid_search.py            BM25 + FAISS via Reciprocal Rank Fusion
│   ├── reranker.py                 CrossEncoder + LLM reranker
│   └── compressor.py              Sentence-level + LLM context compression
│
├── temarios/                       ← your PDFs (unchanged)
└── vectorstore/                    ← FAISS index + new skills metadata files
    ├── faiss_index.bin
    ├── chunks_metadata.pkl
    ├── skills_chunk_meta.pkl       ← NEW
    ├── skills_doc_profiles.json    ← NEW
    └── skills_index.json           ← NEW
```

---

## Installation

```bash
pip install -r requirements_v2.txt
```

---

## Step-by-Step Activation

### Step 1 — Switch to the new app (zero config needed)
```bash
streamlit run app_v2.py
```
It loads your existing FAISS index automatically.
Hybrid search (BM25+FAISS) and compression are active immediately.
The cross-encoder reranker downloads ~80MB on first query.

### Step 2 — Extract skills (optional, enables filtering)
```bash
python extract_skills.py
```
Calls OpenRouter once per chunk (~$0.01-0.05 total with gpt-4o-mini).
After this, the sidebar in app_v2.py shows Domain / Difficulty / Skill filters.

### Step 3 — Run the evaluation pipeline
```bash
# Use the included seed dataset (20 hand-crafted queries):
python run_evaluation.py

# Auto-generate a richer dataset from your actual chunks:
python run_evaluation.py --generate

# Compare different chunk sizes:
python run_evaluation.py --compare-chunk-sizes
```

---

## Component Reference

### 1. Skill Extraction

```python
from skills.extractor import SkillExtractor
from skills.metadata_store import SkillsMetadataStore

extractor = SkillExtractor(api_key="sk-or-...", model="openai/gpt-4o-mini")
enriched_chunks = extractor.extract_batch(chunks)
doc_profiles    = extractor.aggregate_document_skills(enriched_chunks)

store = SkillsMetadataStore()
store.build_from_chunks(enriched_chunks, doc_profiles)

# Query the store
matching_ids = store.filter_by_skill("linear regression")
matching_ids = store.apply_filters(domain="machine learning", difficulty="beginner")
```

### 2. Evaluation

```python
from evaluation.dataset import EvaluationDataset
from evaluation.runner import EvaluationRunner
from evaluation.report import print_report

dataset = EvaluationDataset.from_file("evaluation/eval_queries.jsonl")
runner  = EvaluationRunner(dataset)

runner.add_experiment(
    name="my_pipeline",
    retrieve_fn=lambda q: engine.retrieve(q),
    k_values=[1, 3, 5],
)
runner.run()
print_report(runner.get_results())
```

### 3. Hybrid Search (standalone)

```python
from retrieval.hybrid_search import HybridSearchEngine

hybrid = HybridSearchEngine(chunks)
hybrid.build()
results = hybrid.search(query, faiss_index, embedding_model, top_k=5)
```

### 4. Re-ranking (standalone)

```python
from retrieval.reranker import CrossEncoderReranker

reranker  = CrossEncoderReranker()          # downloads model on first call
reranked  = reranker.rerank(query, chunks, top_n=5)
```

### 5. Context Compression (standalone)

```python
from retrieval.compressor import ContextCompressor

compressor = ContextCompressor(embedding_model=model, strategy="sentence")
compressed = compressor.compress(query, chunks, threshold=0.3)
```

---

## Retrieval Pipeline (full flow)

```
User Query
    │
    ▼
[1] Skills Metadata Filter ──── allowed_chunk_indices (Set[int])
    │
    ▼
[2] Hybrid Search ─────────────  BM25 top-50 ──┐
                                                 ├── RRF Fusion → top-20
                                  FAISS top-50 ──┘
    │
    ▼
[3] Cross-Encoder Reranker ───── score each (query, chunk) → top-5
    │
    ▼
[4] Context Compressor ────────  keep query-relevant sentences only
    │
    ▼
[5] LLM Generation ────────────  enriched prompt with skills metadata in headers
    │
    ▼
Answer + Sources
```

---

## Evaluation Metrics Reference

| Metric        | What it measures                               | Good value |
|---------------|------------------------------------------------|------------|
| Recall@K      | Fraction of relevant docs found in top-K       | > 0.7      |
| Precision@K   | Fraction of top-K that are relevant            | > 0.5      |
| MRR           | Reciprocal rank of first relevant result       | > 0.6      |
| nDCG@K        | Rank-weighted recall (position matters)        | > 0.6      |
| MAP           | Mean precision across all relevant positions   | > 0.5      |

---

## Cost Estimates (gpt-4o-mini via OpenRouter)

| Operation                    | Approx. cost       |
|------------------------------|--------------------|
| Skill extraction (all chunks)| ~$0.02–0.05        |
| Dataset auto-generation (40q)| ~$0.01             |
| LLM reranker (per query)     | ~$0.001            |
| LLM compressor (per query)   | ~$0.001            |

Cross-encoder reranker and sentence compressor are **free** (local models).
