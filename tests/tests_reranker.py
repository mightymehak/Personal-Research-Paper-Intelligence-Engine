
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.faiss_service import FaissVectorStore
from backend.app.services.reranker import CrossEncoderReranker


INDEX_DIR = "data/processed/library_a56403bd/faiss"

TEST_QUERIES = [
    "What are the advantages of decoupling vector indexes in PostgreSQL?",
    "How do approximate nearest neighbor indexes work?",
    "What metrics are used to evaluate vector search performance?",
]


def main():
    # Load the existing vector index.
    embedding_service = EmbeddingService()
    store = FaissVectorStore(embedding_service)
    store.load(INDEX_DIR)

    # Load the reranker once, not once per query.
    reranker = CrossEncoderReranker()

    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        # Stage 1: retrieve 50 candidates with FAISS.
        candidates = store.search(query, top_k=50)

        # Show the original FAISS ranking.
        print("\nTOP 3 BEFORE RERANKING")

        for result in candidates[:3]:
            print(
                f"\nRank: {result.get('rank')}"
                f"\nFAISS score: {result.get('score', 0):.4f}"
                f"\nPaper: {result.get('paper_title', '')}"
                f"\nSection: {result.get('section', '')}"
                f"\nText: {result.get('text', '')[:350]}"
            )

        # Stage 2: rerank those same candidates.
        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=10,
        )

        print("\nTOP 3 AFTER RERANKING")

        for result in reranked[:3]:
            print(
                f"\nNew rank: {result['rank']}"
                f"\nPrevious rank: {result.get('original_rank')}"
                f"\nReranker score: {result['rerank_score']:.4f}"
                f"\nPaper: {result.get('paper_title', '')}"
                f"\nSection: {result.get('section', '')}"
                f"\nText: {result.get('text', '')[:350]}"
            )


if __name__ == "__main__":
    main()
