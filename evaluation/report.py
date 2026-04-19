# evaluation/report.py
# ============================================================
# Report generation: console tables + CSV export.
# ============================================================

import csv
import os
from typing import Any, Dict, List


def print_report(results: Dict[str, Dict], k_values: List[int] = (1, 3, 5, 10)) -> None:
    """
    Print a formatted comparison table to stdout.

    Args:
        results:  Dict from EvaluationRunner.get_results().
        k_values: Which k values to show in the table.
    """
    if not results:
        print("No results to display.")
        return

    # Build header
    metrics_cols = []
    for k in k_values:
        metrics_cols += [f"R@{k}", f"P@{k}", f"nDCG@{k}"]
    metrics_cols += ["MRR", "MAP", "Lat(ms)"]

    col_w = 9
    name_w = 30
    header = f"{'Experiment':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in metrics_cols)
    separator = "─" * len(header)

    print("\n" + separator)
    print("EVALUATION RESULTS")
    print(separator)
    print(header)
    print(separator)

    for name, res in results.items():
        row = f"{name:<{name_w}}"
        for k in k_values:
            row += f"{res.get(f'recall@{k}', 0):>{col_w}.4f}"
            row += f"{res.get(f'precision@{k}', 0):>{col_w}.4f}"
            row += f"{res.get(f'ndcg@{k}', 0):>{col_w}.4f}"
        row += f"{res.get('mrr', 0):>{col_w}.4f}"
        row += f"{res.get('ap', 0):>{col_w}.4f}"
        row += f"{res.get('avg_latency_ms', 0):>{col_w}.1f}"
        print(row)

    print(separator + "\n")


def save_report_csv(
    results: Dict[str, Dict],
    path: str = "evaluation/results.csv",
    k_values: List[int] = (1, 3, 5, 10),
) -> None:
    """
    Save evaluation results to a CSV file for further analysis.

    Args:
        results:  Dict from EvaluationRunner.get_results().
        path:     Output CSV path.
        k_values: K values to include.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metric_keys = []
    for k in k_values:
        metric_keys += [f"recall@{k}", f"precision@{k}", f"ndcg@{k}"]
    metric_keys += ["mrr", "ap", "avg_latency_ms", "n_queries"]

    fieldnames = ["experiment"] + metric_keys

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, res in results.items():
            row = {"experiment": name}
            for k in metric_keys:
                row[k] = res.get(k, "")
            writer.writerow(row)

    print(f"[Report] Results saved to {path}")


def llm_judge_score(
    query: str,
    answer: str,
    context: str,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    base_url: str = "https://openrouter.ai/api/v1",
) -> Dict[str, Any]:
    """
    Use an LLM as a judge to score answer quality on a 1-5 scale.
    Returns {"relevance": int, "faithfulness": int, "reasoning": str}.

    This is optional — use it for qualitative sampling, not full-scale eval.
    """
    import httpx
    from openai import OpenAI

    system = (
        "You are an evaluation judge for a university RAG system. "
        "Score the answer on two dimensions (1-5 each). "
        "Respond ONLY with JSON: "
        '{"relevance": int, "faithfulness": int, "reasoning": "brief explanation"}'
    )
    prompt = (
        f"QUERY: {query}\n\n"
        f"CONTEXT (retrieved): {context[:1000]}\n\n"
        f"ANSWER: {answer}\n\n"
        "Score:\n"
        "- relevance (1-5): Does the answer address the query?\n"
        "- faithfulness (1-5): Is the answer grounded in the context?\n"
    )

    client = OpenAI(api_key=api_key, base_url=base_url,
                    http_client=httpx.Client())
    try:
        import json
        response = client.chat.completions.create(
            model=model,
            max_tokens=200,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"relevance": 0, "faithfulness": 0, "reasoning": str(e)}
