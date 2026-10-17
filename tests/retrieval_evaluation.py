
import argparse
import csv
import json
import math
from pathlib import Path

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.faiss_service import FaissVectorStore


ROOT = Path(__file__).resolve().parents[1]

QUERY_FILE = ROOT / "tests" / "retrieval_eval_queries.json"

INDEX_DIR = (
    ROOT
    / "data"
    / "processed"
    / "library_a56403bd"
    / "faiss"
)

RESULTS_DIR = ROOT / "data" / "processed" / "evaluation"

# Candidate pool size and evaluation cutoffs are separate.
CANDIDATE_K = 50
EVALUATION_CUTOFFS = (5, 10)


def load_queries():
    with QUERY_FILE.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list) or not queries:
        raise ValueError("Query file must contain a non-empty JSON list.")

    query_ids = [item.get("query_id") for item in queries]

    if any(not query_id for query_id in query_ids):
        raise ValueError("Every query must have a query_id.")

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id values found.")

    for item in queries:
        if not item.get("query", "").strip():
            raise ValueError(
                f"Empty query for {item['query_id']}."
            )

        relevant_ids = item.get("relevant_chunk_ids")

        if not isinstance(relevant_ids, list):
            raise ValueError(
                f"relevant_chunk_ids must be a list for "
                f"{item['query_id']}."
            )

    return queries


def load_store():
    index_file = INDEX_DIR / "faiss.index"

    if not index_file.is_file():
        raise FileNotFoundError(
            f"FAISS index not found: {index_file}"
        )

    chunks_file = INDEX_DIR / "chunks.json"

    if not chunks_file.is_file():
        raise FileNotFoundError(
            f"Chunk metadata not found: {chunks_file}"
        )

    embedding_service = EmbeddingService()
    store = FaissVectorStore(embedding_service)
    store.load(INDEX_DIR)

    print(f"Loaded {store.index.ntotal} indexed chunks.")
    return store


def collect_candidates(store, queries):
    """Export the top 50 retrieved candidates for manual review."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "retrieval_candidates_top50.csv"
    rows = []

    for number, item in enumerate(queries, start=1):
        results = store.search(
            item["query"],
            top_k=CANDIDATE_K
        )

        print(
            f"[{number}/{len(queries)}] "
            f"{item['query_id']}: retrieved {len(results)} candidates"
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
        "query_id",
        "query",
        "rank",
        "score",
        "chunk_id",
        "paper_title",
        "section",
        "text",
        "relevant"
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCandidates saved to: {output_file}")
    print(f"Total candidate rows: {len(rows)}")


def precision_at_k(relevant_ids, retrieved_ids, k):
    if k <= 0:
        return 0.0

    hits = sum(
        1 for chunk_id in retrieved_ids[:k]
        if chunk_id in relevant_ids
    )

    # Standard Precision@K uses K as the denominator.
    return hits / k


def recall_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return None

    hits = sum(
        1 for chunk_id in retrieved_ids[:k]
        if chunk_id in relevant_ids
    )

    return hits / len(relevant_ids)


def reciprocal_rank_at_k(relevant_ids, retrieved_ids, k):
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return None

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(
            retrieved_ids[:k], start=1
        )
        if chunk_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def evaluate(store, queries):
    evaluated = []
    skipped = []

    # Query once per item, then calculate every metric from
    # the same ranked candidate list.
    for number, item in enumerate(queries, start=1):
        relevant_ids = {
            chunk_id
            for chunk_id in item["relevant_chunk_ids"]
            if isinstance(chunk_id, str) and chunk_id.strip()
        }

        if not relevant_ids:
            skipped.append({
                "query_id": item["query_id"],
                "reason": "No relevant chunk IDs labeled."
            })
            continue

        results = store.search(
            item["query"],
            top_k=CANDIDATE_K
        )

        retrieved_ids = [
            result["chunk_id"] for result in results
        ]

        row = {
            "query_id": item["query_id"],
            "query": item["query"],
            "relevant_chunk_count": len(relevant_ids)
        }

        for k in EVALUATION_CUTOFFS:
            row[f"precision@{k}"] = precision_at_k(
                relevant_ids, retrieved_ids, k
            )
            row[f"recall@{k}"] = recall_at_k(
                relevant_ids, retrieved_ids, k
            )
            row[f"ndcg@{k}"] = ndcg_at_k(
                relevant_ids, retrieved_ids, k
            )

        row["mrr@10"] = reciprocal_rank_at_k(
            relevant_ids, retrieved_ids, 10
        )

        evaluated.append(row)

        print(
            f"[{number}/{len(queries)}] "
            f"{item['query_id']}: evaluated"
        )

    if not evaluated:
        print(
            "\nNo queries have relevance labels. "
            "Add relevant chunk IDs before evaluating."
        )
        return

    metric_names = [
        "precision@5",
        "recall@5",
        "ndcg@5",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "mrr@10"
    ]

    # Macro-average over labeled queries. Recall/NDCG are
    # omitted for queries without labels, which were skipped.
    summary = {
        metric: sum(row[metric] for row in evaluated)
        / len(evaluated)
        for metric in metric_names
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_query_file = RESULTS_DIR / "per_query_metrics.csv"

    with per_query_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(evaluated[0].keys())
        )
        writer.writeheader()
        writer.writerows(evaluated)

    summary_file = RESULTS_DIR / "summary_metrics.json"

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "candidate_k": CANDIDATE_K,
                "evaluation_cutoffs": list(EVALUATION_CUTOFFS),
                "evaluated_queries": len(evaluated),
                "skipped_queries": skipped,
                "metrics": summary
            },
            file,
            indent=2
        )

    print("\n" + "=" * 45)
    print("PAPERMIND RETRIEVAL EVALUATION")
    print("=" * 45)
    print(f"Candidate pool: top {CANDIDATE_K}")
    print(f"Evaluated queries: {len(evaluated)}")
    print(f"Skipped queries: {len(skipped)}")

    print("\nMean retrieval metrics:")

    for metric, value in summary.items():
        print(f"{metric:>14}: {value:.4f}")

    if skipped:
        print("\nSkipped queries:")
        for item in skipped:
            print(
                f"  {item['query_id']}: {item['reason']}"
            )

    print(f"\nPer-query report: {per_query_file}")
    print(f"Summary report:   {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PaperMind semantic retrieval."
    )

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
