
import argparse
import csv
import json
from pathlib import Path

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.faiss_service import FaissVectorStore


ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = ROOT / "tests" / "retrieval_eval_queries.json"
INDEX_DIR = (
    ROOT / "data" / "processed" /
    "library_a56403bd" / "faiss"
)
RESULTS_DIR = ROOT / "data" / "processed" / "evaluation"
TOP_K = 10


def load_queries():
    with QUERY_FILE.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not queries:
        raise ValueError("The evaluation query file is empty.")

    ids = [item["query_id"] for item in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query_id values found.")

    for item in queries:
        if not item.get("query", "").strip():
            raise ValueError(
                f"Empty query for {item['query_id']}"
            )
        if not isinstance(item.get("relevant_chunk_ids"), list):
            raise ValueError(
                f"relevant_chunk_ids must be a list: "
                f"{item['query_id']}"
            )

    return queries


def load_store():
    if not (INDEX_DIR / "faiss.index").is_file():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_DIR}"
        )

    embedding_service = EmbeddingService()
    store = FaissVectorStore(embedding_service)
    store.load(INDEX_DIR)
    return store


def collect_candidates(store, queries):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "retrieval_candidates.csv"

    rows = []

    for number, item in enumerate(queries, start=1):
        results = store.search(item["query"], top_k=TOP_K)

        print(
            f"[{number}/{len(queries)}] "
            f"{item['query_id']}: {item['query']}"
        )

        for result in results:
            rows.append({
                "query_id": item["query_id"],
                "query": item["query"],
                "rank": result["rank"],
                "score": result["score"],
                "chunk_id": result["chunk_id"],
                "paper_title": result["paper_title"],
                "section": result["section"],
                "text": result["text"],
                "relevant": ""
            })

    columns = [
        "query_id", "query", "rank", "score",
        "chunk_id", "paper_title", "section",
        "text", "relevant"
    ]

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCandidate results saved to: {output_file}")
    print(
        "Inspect the retrieved passages and copy relevant "
        "chunk IDs into the JSON query file."
    )


def precision_at_k(relevant_ids, retrieved_ids, k):
    if k <= 0:
        return 0.0

    hits = sum(
        1 for chunk_id in retrieved_ids[:k]
        if chunk_id in relevant_ids
    )
    return hits / k


def recall_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return None

    hits = sum(
        1 for chunk_id in retrieved_ids[:k]
        if chunk_id in relevant_ids
    )
    return hits / len(relevant_ids)


def reciprocal_rank(relevant_ids, retrieved_ids):
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return None

    dcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank, chunk_id in enumerate(
            retrieved_ids[:k], start=1
        )
        if chunk_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def evaluate(store, queries):
    evaluated = []
    skipped = []

    for item in queries:
        relevant_ids = set(item["relevant_chunk_ids"])

        if not relevant_ids:
            skipped.append(item["query_id"])
            continue

        results = store.search(item["query"], top_k=TOP_K)
        retrieved_ids = [
            result["chunk_id"] for result in results
        ]

        evaluated.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "precision@5": precision_at_k(
                relevant_ids, retrieved_ids, 5
            ),
            "recall@5": recall_at_k(
                relevant_ids, retrieved_ids, 5
            ),
            "precision@10": precision_at_k(
                relevant_ids, retrieved_ids, 10
            ),
            "recall@10": recall_at_k(
                relevant_ids, retrieved_ids, 10
            ),
            "mrr": reciprocal_rank(
                relevant_ids, retrieved_ids
            ),
            "ndcg@10": ndcg_at_k(
                relevant_ids, retrieved_ids, 10
            )
        })

    if not evaluated:
        print(
            "\nNo labeled queries yet. Run candidate collection "
            "and add relevant chunk IDs to the JSON file."
        )
        return

    metric_names = [
        "precision@5", "recall@5",
        "precision@10", "recall@10",
        "mrr", "ndcg@10"
    ]

    summary = {
        metric: sum(row[metric] for row in evaluated)
        / len(evaluated)
        for metric in metric_names
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "per_query_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(evaluated[0].keys())
        )
        writer.writeheader()
        writer.writerows(evaluated)

    with (RESULTS_DIR / "summary_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({
            "evaluated_queries": len(evaluated),
            "skipped_queries": skipped,
            "metrics": summary
        }, file, indent=2)

    print(f"\nEvaluated queries: {len(evaluated)}")
    print(f"Skipped unlabeled queries: {len(skipped)}")
    print("\nMean retrieval metrics:")

    for metric, value in summary.items():
        print(f"{metric:>12}: {value:.4f}")

    print(f"\nReports saved in: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Calculate metrics for labeled queries."
    )
    args = parser.parse_args()

    queries = load_queries()
    store = load_store()

    if args.evaluate:
        evaluate(store, queries)
    else:
        collect_candidates(store, queries)


if __name__ == "__main__":
    main()
