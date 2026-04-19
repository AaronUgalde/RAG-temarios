# evaluation/metrics.py
# ============================================================
# Pure, dependency-free retrieval evaluation metrics.
# All functions operate on lists/sets — no external libraries.
# ============================================================

from typing import List, Set, Any


def recall_at_k(
    retrieved_ids: List[Any],
    relevant_ids: Set[Any],
    k: int,
) -> float:
    """
    Recall@K: fraction of relevant items found in top-K results.

    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_ids: Ranked list of retrieved item IDs.
        relevant_ids:  Set of ground-truth relevant item IDs.
        k:             Cutoff position.

    Returns:
        Float in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[Any],
    relevant_ids: Set[Any],
    k: int,
) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.

    Precision@K = |retrieved[:k] ∩ relevant| / k
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / k


def mean_reciprocal_rank(
    retrieved_ids: List[Any],
    relevant_ids: Set[Any],
) -> float:
    """
    MRR (Mean Reciprocal Rank) for a single query.

    MRR = 1 / rank_of_first_relevant_result
    Returns 0 if no relevant item appears in the list.

    To compute corpus-level MRR, average this over all queries.
    """
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[Any],
    relevant_ids: Set[Any],
    k: int,
) -> float:
    """
    nDCG@K (Normalized Discounted Cumulative Gain).

    Uses binary relevance (1 if relevant, 0 if not).
    Ideal DCG assumes all relevant docs appear at top positions.

    Args:
        retrieved_ids: Ranked list of retrieved item IDs.
        relevant_ids:  Set of ground-truth relevant item IDs.
        k:             Cutoff position.

    Returns:
        Float in [0, 1].
    """
    import math

    def dcg(ids, k):
        return sum(
            (1.0 / math.log2(i + 2))
            for i, item in enumerate(ids[:k])
            if item in relevant_ids
        )

    actual_dcg = dcg(retrieved_ids, k)
    # Ideal DCG: imagine all relevant items at top positions
    ideal_list = list(relevant_ids)[:k]
    ideal_dcg = dcg(ideal_list, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(
    retrieved_ids: List[Any],
    relevant_ids: Set[Any],
) -> float:
    """
    Average Precision (AP) for a single query.
    Combines precision at each relevant position.
    Mean AP over all queries = MAP.
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    total_precision = 0.0
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            hits += 1
            total_precision += hits / rank

    return total_precision / len(relevant_ids)


def compute_all_metrics(
    retrieved_ids: List[Any],
    relevant_ids: Set[Any],
    k_values: List[int] = (1, 3, 5, 10),
) -> dict:
    """
    Convenience wrapper: compute all metrics for a single query.

    Returns:
        Dict with keys like "recall@1", "precision@5", "mrr", "ndcg@5", "ap"
    """
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"]    = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"]      = ndcg_at_k(retrieved_ids, relevant_ids, k)
    metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    metrics["ap"]  = average_precision(retrieved_ids, relevant_ids)
    return metrics
