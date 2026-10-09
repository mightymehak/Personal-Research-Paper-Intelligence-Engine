
from pathlib import Path

from backend.app.services.pdf_parser import parse_pdf
from backend.app.services.chunker import chunk_paper
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.faiss_service import FaissVectorStore


PDF_DIRECTORY = Path("data/raw")
PROCESSED_DIRECTORY = Path("data/processed")

TEST_QUERIES = [
    "How do vector databases perform similarity search?",
    "What are the advantages of decoupling vector indexes in PostgreSQL?",
    "How do approximate nearest neighbor indexes work?",
]


def main():
    if not PDF_DIRECTORY.exists():
        print("data/raw directory does not exist.")
        return

    libraries = [
        directory
        for directory in PDF_DIRECTORY.iterdir()
        if directory.is_dir()
    ]

    if not libraries:
        print("No research paper libraries found.")
        return

    embedding_service = EmbeddingService()

    for library_directory in libraries:
        print("\n" + "=" * 80)
        print(f"LIBRARY: {library_directory.name}")
        print("=" * 80)

        pdf_files = sorted(
            pdf
            for pdf in library_directory.rglob("*.pdf")
            if "__MACOSX" not in pdf.parts
            and not pdf.name.startswith("._")
        )

        if not pdf_files:
            print("No PDF files found.")
            continue

        all_chunks = []

        for pdf_file in pdf_files:
            try:
                paper = parse_pdf(pdf_file)
                chunks = chunk_paper(paper)

                all_chunks.extend(chunks)

                print(
                    f"{pdf_file.name}: "
                    f"{len(chunks)} chunks"
                )

            except Exception as exc:
                print(f"Failed to process {pdf_file.name}: {exc}")

        if not all_chunks:
            print("No chunks available to index.")
            continue

        store = FaissVectorStore(embedding_service)
        store.build_index(all_chunks)

        assert store.index.ntotal == len(all_chunks)
        assert store.index.d == 384

        print(f"\nTotal indexed chunks: {store.index.ntotal}")

        output_directory = (
            PROCESSED_DIRECTORY
            / library_directory.name
            / "faiss"
        )

        store.save(output_directory)

        # Test semantic retrieval.
        for query in TEST_QUERIES:
            print("\n" + "-" * 80)
            print(f"QUERY: {query}")
            print("-" * 80)

            results = store.search(query, top_k=3)

            for result in results:
                print(
                    f"\nRank: {result['rank']} | "
                    f"Similarity: {result['score']:.4f}"
                )
                print(f"Paper: {result['paper_title']}")
                print(f"Section: {result['section']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Text: {result['text'][:500]}...")

            assert len(results) <= 3
            assert all(
                result["paper_title"] for result in results
            )

        # Test loading the saved index.
        loaded_store = FaissVectorStore(embedding_service)
        loaded_store.load(output_directory)

        assert loaded_store.index.ntotal == len(all_chunks)

        original_results = store.search(
            TEST_QUERIES[0], top_k=3
        )
        loaded_results = loaded_store.search(
            TEST_QUERIES[0], top_k=3
        )

        assert [
            result["chunk_id"] for result in original_results
        ] == [
            result["chunk_id"] for result in loaded_results
        ]

        print("\nIndex persistence test: PASS")
        print("Semantic search test: PASS")

    print("\nFAISS TEST COMPLETE")


if __name__ == "__main__":
    main()
