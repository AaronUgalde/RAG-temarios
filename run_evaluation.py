"""
run_evaluation.py
=================
CLI script to run the offline evaluation pipeline.

Usage examples:
    # Generate eval dataset + run all experiments
    python run_evaluation.py --generate --api-key sk-or-...

    # Run experiments with existing dataset
    python run_evaluation.py --dataset evaluation/eval_queries.jsonl --api-key sk-or-...

    # Compare chunk sizes (triggers re-indexing with different sizes)
    python run_evaluation.py --compare-chunk-sizes --api-key sk-or-...
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def get_engine_v2(api_key, model, top_k=5, use_hybrid=True,
                  use_reranker=False, use_compression=False):
    """Build a fresh engine (no reranker/compression for speed in eval)."""
    from rag_engine_v2 import RAGEngineV2
    engine = RAGEngineV2(
        api_key=api_key,
        model=model,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        use_compression=use_compression,
    )
    if not engine.load_index():
        print("[ERROR] No index found. Run app.py and index your documents first.")
        sys.exit(1)
    return engine


def run_default_experiments(api_key, model, dataset_path, top_k_values=(3, 5, 10)):
    """
    Run the standard experiment grid:
        - Baseline (FAISS only, no reranker)
        - Hybrid (BM25+FAISS)
        - Hybrid + Reranker
        - Hybrid + Reranker + Compression
    """
    from evaluation.dataset import EvaluationDataset
    from evaluation.runner import EvaluationRunner
    from evaluation.report import print_report, save_report_csv
    from rag_engine_v2 import RAGEngineV2

    print(f"\n{'='*60}")
    print("SKILLS INTELLIGENCE RAG — EVALUATION")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Top-K values: {top_k_values}\n")

    dataset = EvaluationDataset.from_file(dataset_path)

    runner = EvaluationRunner(dataset)

    # --- Experiment 1: Baseline FAISS ---
    eng_baseline = RAGEngineV2(
        api_key=api_key, model=model, top_k=max(top_k_values),
        use_hybrid=False, use_reranker=False, use_compression=False
    )
    eng_baseline.load_index()
    runner.add_experiment(
        name="1_baseline_faiss",
        retrieve_fn=lambda q: eng_baseline.retrieve(q),
        k_values=list(top_k_values),
    )

    # --- Experiment 2: Hybrid (BM25 + FAISS) ---
    eng_hybrid = RAGEngineV2(
        api_key=api_key, model=model, top_k=max(top_k_values),
        use_hybrid=True, use_reranker=False, use_compression=False
    )
    eng_hybrid.load_index()
    runner.add_experiment(
        name="2_hybrid_bm25_faiss",
        retrieve_fn=lambda q: eng_hybrid.retrieve(q),
        k_values=list(top_k_values),
    )

    # --- Experiment 3: Hybrid + Reranker ---
    eng_rerank = RAGEngineV2(
        api_key=api_key, model=model, top_k=max(top_k_values),
        use_hybrid=True, use_reranker=True, use_compression=False
    )
    eng_rerank.load_index()
    runner.add_experiment(
        name="3_hybrid_reranker",
        retrieve_fn=lambda q: eng_rerank.retrieve(q),
        k_values=list(top_k_values),
    )

    # --- Experiment 4: Full Pipeline ---
    eng_full = RAGEngineV2(
        api_key=api_key, model=model, top_k=max(top_k_values),
        use_hybrid=True, use_reranker=True, use_compression=True
    )
    eng_full.load_index()
    runner.add_experiment(
        name="4_full_pipeline",
        retrieve_fn=lambda q: eng_full.retrieve(q),
        k_values=list(top_k_values),
    )

    runner.run()
    results = runner.get_results()

    print_report(results, k_values=list(top_k_values))
    save_report_csv(results, path="evaluation/results.csv", k_values=list(top_k_values))

    print("\nBest by recall@5:")
    for row in runner.compare("recall@5"):
        print(f"  {row['name']}: {row['recall@5']:.4f}")


def compare_chunk_sizes(api_key, model, dataset_path):
    """
    Re-index with different chunk sizes and compare retrieval quality.
    WARNING: This re-indexes documents multiple times — can take minutes.
    """
    from document_loader import DocumentLoader
    from evaluation.dataset import EvaluationDataset
    from evaluation.runner import EvaluationRunner
    from evaluation.report import print_report, save_report_csv
    from rag_engine_v2 import RAGEngineV2

    CONFIGS = [
        {"chunk_size": 500,  "overlap": 100},
        {"chunk_size": 1000, "overlap": 200},
        {"chunk_size": 1500, "overlap": 300},
    ]

    dataset = EvaluationDataset.from_file(dataset_path)
    runner = EvaluationRunner(dataset)

    for cfg in CONFIGS:
        cs = cfg["chunk_size"]
        co = cfg["overlap"]
        name = f"chunk_{cs}_overlap_{co}"
        print(f"\n[ChunkEval] Indexing with chunk_size={cs}, overlap={co}")

        loader = DocumentLoader("temarios", chunk_size=cs, chunk_overlap=co)
        chunks = loader.load_all_documents()

        engine = RAGEngineV2(
            api_key=api_key, model=model, top_k=5,
            use_hybrid=True, use_reranker=False, use_compression=False
        )
        engine.build_index(chunks, extract_skills=False)

        runner.add_experiment(
            name=name,
            retrieve_fn=lambda q, e=engine: e.retrieve(q),
            k_values=[1, 3, 5],
        )

    runner.run()
    results = runner.get_results()
    print_report(results, k_values=[1, 3, 5])
    save_report_csv(results, "evaluation/chunk_size_comparison.csv", k_values=[1, 3, 5])


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--model",   default=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--dataset", default="evaluation/eval_queries.jsonl")
    parser.add_argument("--generate", action="store_true",
                        help="Auto-generate evaluation dataset from indexed chunks")
    parser.add_argument("--compare-chunk-sizes", action="store_true",
                        help="Re-index with multiple chunk sizes and compare")
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5, 10])
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Set OPENROUTER_API_KEY in .env or pass --api-key")
        sys.exit(1)

    if args.generate:
        print("[DatasetGen] Generating evaluation dataset from indexed chunks...")
        import pickle
        with open("vectorstore/chunks_metadata.pkl", "rb") as f:
            chunks = pickle.load(f)
        from evaluation.dataset import EvaluationDataset
        EvaluationDataset.auto_generate(
            chunks=chunks,
            api_key=args.api_key,
            model=args.model,
            n_per_chunk=2,
            max_chunks=40,
            output_path=args.dataset,
        )

    if args.compare_chunk_sizes:
        compare_chunk_sizes(args.api_key, args.model, args.dataset)
    else:
        if not os.path.exists(args.dataset):
            print(f"[ERROR] Dataset not found: {args.dataset}")
            print("Run with --generate to create one first.")
            sys.exit(1)
        run_default_experiments(args.api_key, args.model, args.dataset, args.top_k)


if __name__ == "__main__":
    main()
