from pathlib import Path

from backend.app.services.chunker import chunk_paper
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.pdf_parser import parse_pdf


PDF_DIRECTORY = Path("data/raw")


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

    # ---------------------------------------------------------
    # Load embedding model ONCE
    # ---------------------------------------------------------

    embedding_service = EmbeddingService()

    print(
        "\nModel:",
        embedding_service.model_name,
    )

    print(
        "Embedding dimension:",
        embedding_service.embedding_dimension,
    )

    for library_directory in libraries:

        print("\n" + "=" * 80)
        print(
            f"LIBRARY: {library_directory.name}"
        )
        print("=" * 80)

        pdf_files = [
            pdf
            for pdf in library_directory.rglob("*.pdf")
            if "__MACOSX" not in pdf.parts
            and not pdf.name.startswith("._")
        ]

        for pdf_file in pdf_files:

            print("\n" + "-" * 80)
            print(
                f"FILE: {pdf_file.name}"
            )
            print("-" * 80)

            paper = parse_pdf(pdf_file)

            chunks = chunk_paper(paper)

            print(
                f"\nTitle: {paper.title}"
            )

            print(
                f"Chunks: {len(chunks)}"
            )

            if not chunks:
                print(
                    "No chunks generated."
                )
                continue

            # -------------------------------------------------
            # Generate embeddings
            # -------------------------------------------------

            embeddings = (
                embedding_service.encode_chunks(
                    chunks
                )
            )

            print(
                "\nEmbedding shape:"
            )

            print(
                embeddings.shape
            )

            # -------------------------------------------------
            # Verify expected shape
            # -------------------------------------------------

            expected_shape = (
                len(chunks),
                embedding_service.embedding_dimension,
            )

            if embeddings.shape == expected_shape:

                print(
                    "Embedding shape validation: PASS"
                )

            else:

                print(
                    "Embedding shape validation: FAIL"
                )

                print(
                    f"Expected: {expected_shape}"
                )

                print(
                    f"Actual: {embeddings.shape}"
                )

            # -------------------------------------------------
            # Verify dtype
            # -------------------------------------------------

            print(
                f"Embedding dtype: {embeddings.dtype}"
            )

            if embeddings.dtype.name == "float32":

                print(
                    "Embedding dtype validation: PASS"
                )

            else:

                print(
                    "Embedding dtype validation: FAIL"
                )

            # -------------------------------------------------
            # Verify normalization
            # -------------------------------------------------

            norms = (
                embeddings
                * embeddings
            ).sum(axis=1) ** 0.5

            max_norm_error = abs(
                norms - 1.0
            ).max()

            print(
                "\nMaximum normalization error:"
            )

            print(
                f"{max_norm_error:.8f}"
            )

            if max_norm_error < 1e-5:

                print(
                    "Embedding normalization: PASS"
                )

            else:

                print(
                    "Embedding normalization: FAIL"
                )

            # -------------------------------------------------
            # Show first embedding
            # -------------------------------------------------

            print(
                "\nFirst chunk:"
            )

            print(
                f"Chunk ID: "
                f"{chunks[0].chunk_id}"
            )

            print(
                f"Section: "
                f"{chunks[0].section}"
            )

            print(
                f"Word count: "
                f"{chunks[0].word_count}"
            )

            print(
                "\nFirst 10 embedding values:"
            )

            print(
                embeddings[0][:10]
            )

    print("\n" + "=" * 80)
    print(
        "EMBEDDING TEST COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()