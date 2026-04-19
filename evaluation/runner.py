# evaluation/runner.py
# ============================================================
# Orchestrates evaluation experiments.
# Supports comparing: chunk_sizes, top_k values, retrieval modes.
# ============================================================

import time
from typing import Any, Callable, Dict, List, Optional

from .dataset import EvaluationDataset
from .metrics import compute_all_metrics


class EvaluationRunner:
    """
    Runs retrieval experiments and collects metrics.

    Usage:
        runner = EvaluationRunner(dataset)

        # Register a retrieval function
        runner.add_experiment(
            name="baseline_k5",
            retrieve_fn=lambda q: engine.retrieve(q),
            k_values=[1, 3, 5],
        )
        runner.run()
        results = runner.get_results()
    """

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset
        self._experiments: List[Dict[str, Any]] = []
        self._results: Dict[str, Dict] = {}

    def add_experiment(
        self,
        name: str,
        retrieve_fn: Callable[[str], List[Dict]],
        k_values: List[int] = (1, 3, 5, 10),
        id_field: str = "source",
    ) -> None:
        """
        Register an experiment.

        Args:
            name:        Unique name for this experiment config.
            retrieve_fn: Function (query: str) → List[chunk_dict].
                         Must return dicts with at least the `id_field` key.
            k_values:    Cutoff positions to evaluate at.
            id_field:    Chunk field used as the relevance ID.
                         Use "source" for document-level eval,
                         "chunk_id" for chunk-level eval.
        """
        self._experiments.append({
            "name": name,
            "retrieve_fn": retrieve_fn,
            "k_values": list(k_values),
            "id_field": id_field,
        })

    def run(self, verbose: bool = True) -> None:
        """
        Execute all registered experiments against the evaluation dataset.
        Results are stored internally; use get_results() to access them.
        """
        for exp in self._experiments:
            name       = exp["name"]
            retrieve   = exp["retrieve_fn"]
            k_values   = exp["k_values"]
            id_field   = exp["id_field"]

            if verbose:
                print(f"\n[EvalRunner] Running experiment: '{name}'")

            per_query_metrics = []
            latencies = []

            for item in self.dataset:
                query    = item["query"]
                relevant = set(item.get("relevant_sources", []))
                if id_field == "chunk_id":
                    relevant = set(item.get("relevant_chunk_ids", []))

                t0 = time.perf_counter()
                retrieved = retrieve(query)
                latency = time.perf_counter() - t0

                retrieved_ids = [r.get(id_field, "") for r in retrieved]
                q_metrics = compute_all_metrics(retrieved_ids, relevant, k_values)
                per_query_metrics.append(q_metrics)
                latencies.append(latency)

            # Aggregate: mean over all queries
            aggregated = self._aggregate(per_query_metrics)
            aggregated["avg_latency_ms"] = round(sum(latencies) / len(latencies) * 1000, 2)
            aggregated["n_queries"] = len(self.dataset)
            self._results[name] = aggregated

            if verbose:
                self._print_experiment_summary(name, aggregated, k_values)

    def get_results(self) -> Dict[str, Dict]:
        """Return all experiment results keyed by experiment name."""
        return dict(self._results)

    def compare(self, metric: str = "recall@5") -> List[Dict[str, Any]]:
        """
        Return experiments sorted by the given metric (descending).

        Args:
            metric: Metric key, e.g. "recall@5", "mrr", "ndcg@5".

        Returns:
            List of {"name": ..., metric: ...} sorted best-first.
        """
        rows = []
        for name, result in self._results.items():
            rows.append({"name": name, metric: result.get(metric, 0.0)})
        return sorted(rows, key=lambda x: x[metric], reverse=True)

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
        if not per_query:
            return {}
        keys = per_query[0].keys()
        return {
            k: round(sum(q[k] for q in per_query) / len(per_query), 4)
            for k in keys
        }

    @staticmethod
    def _print_experiment_summary(name: str, result: Dict, k_values: List[int]):
        print(f"  ┌─ {name}")
        for k in k_values:
            r = result.get(f"recall@{k}", "—")
            p = result.get(f"precision@{k}", "—")
            n = result.get(f"ndcg@{k}", "—")
            print(f"  │  recall@{k}={r:.4f}  precision@{k}={p:.4f}  ndcg@{k}={n:.4f}")
        print(f"  │  mrr={result.get('mrr', 0):.4f}  "
              f"map={result.get('ap', 0):.4f}  "
              f"avg_latency={result.get('avg_latency_ms', 0):.1f}ms")
        print(f"  └─ n_queries={result.get('n_queries', 0)}")
