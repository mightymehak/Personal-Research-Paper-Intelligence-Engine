from pathlib import Path

from backend.app.services.chunker import chunk_paper
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.pdf_parser import parse_pdf


PDF_DIRECTORY = Path("data/raw")


def count_tokens(tokenizer, text: str) -> int:
    """
    Count the actual tokenizer tokens without truncation.

    This tells us how many tokens exist in the complete chunk,
    before SentenceTransformer applies its maximum sequence length.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    return len(encoded["input_ids"])


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

    print("=" * 80)
    print("PAPERMIND TOKENIZATION ANALYSIS")
    print("=" * 80)

    embedding_service = EmbeddingService()

    tokenizer = embedding_service.model.tokenizer

    model_max_tokens = embedding_service.model.max_seq_length

    print("\nEmbedding model:")
    print(f"  {embedding_service.model_name}")

    print("\nEmbedding dimension:")
    print(f"  {embedding_service.embedding_dimension}")

    print("\nSentenceTransformer max sequence length:")
    print(f"  {model_max_tokens} tokens")

    print("\nTokenizer:")
    print(f"  {tokenizer.__class__.__name__}")

    overall_token_counts = []
    overall_oversized_chunks = 0
    overall_total_chunks = 0

    for library_directory in libraries:

        print("\n" + "=" * 80)
        print(f"LIBRARY: {library_directory.name}")
        print("=" * 80)

        pdf_files = [
            pdf
            for pdf in library_directory.rglob("*.pdf")
            if "__MACOSX" not in pdf.parts
            and not pdf.name.startswith("._")
        ]

        for pdf_file in pdf_files:

            print("\n" + "-" * 80)
            print(f"FILE: {pdf_file.name}")
            print("-" * 80)

            try:
                paper = parse_pdf(pdf_file)
                chunks = chunk_paper(paper)

                print(f"\nTitle: {paper.title}")
                print(f"Chunks: {len(chunks)}")

                if not chunks:
                    print("No chunks generated.")
                    continue

                token_counts = []

                for chunk in chunks:
                    tokens = count_tokens(
                        tokenizer,
                        chunk.text,
                    )

                    token_counts.append(tokens)
                    overall_token_counts.append(tokens)

                total_chunks = len(token_counts)

                oversized_chunks = [
                    count
                    for count in token_counts
                    if count > model_max_tokens
                ]

                oversized_count = len(
                    oversized_chunks
                )

                overall_total_chunks += total_chunks
                overall_oversized_chunks += oversized_count

                minimum_tokens = min(token_counts)
                maximum_tokens = max(token_counts)
                average_tokens = (
                    sum(token_counts)
                    / total_chunks
                )

                print("\nToken statistics:")
                print(
                    f"  Minimum tokens: "
                    f"{minimum_tokens}"
                )
                print(
                    f"  Maximum tokens: "
                    f"{maximum_tokens}"
                )
                print(
                    f"  Average tokens: "
                    f"{average_tokens:.2f}"
                )

                print(
                    "\nModel limit:"
                )
                print(
                    f"  Maximum sequence length: "
                    f"{model_max_tokens} tokens"
                )

                print(
                    "\nPotential truncation:"
                )
                print(
                    f"  Chunks exceeding limit: "
                    f"{oversized_count}/{total_chunks}"
                )

                percentage = (
                    oversized_count
                    / total_chunks
                    * 100
                )

                print(
                    f"  Percentage exceeding limit: "
                    f"{percentage:.2f}%"
                )

                # Token distribution
                ranges = [
                    ("0-64", 0, 64),
                    ("65-128", 65, 128),
                    ("129-192", 129, 192),
                    ("193-256", 193, 256),
                    ("257-384", 257, 384),
                    ("385-512", 385, 512),
                    ("513+", 513, float("inf")),
                ]

                print("\nToken distribution:")

                for label, lower, upper in ranges:
                    count = sum(
                        1
                        for value in token_counts
                        if lower <= value <= upper
                    )

                    print(
                        f"  {label:>8}: "
                        f"{count:>3} chunks"
                    )

                # Show largest chunks
                largest_indices = sorted(
                    range(len(token_counts)),
                    key=lambda index: token_counts[index],
                    reverse=True,
                )[:5]

                print(
                    "\nLargest chunks:"
                )

                for index in largest_indices:
                    chunk = chunks[index]

                    print(
                        f"  Chunk {chunk.chunk_index}: "
                        f"{chunk.word_count} words, "
                        f"{token_counts[index]} tokens, "
                        f"section='{chunk.section}'"
                    )

                # Show chunks that exceed model limit
                if oversized_count > 0:

                    print(
                        "\nChunks exceeding model limit:"
                    )

                    for index, token_count in enumerate(
                        token_counts
                    ):
                        if token_count > model_max_tokens:

                            chunk = chunks[index]

                            print(
                                f"  Chunk {chunk.chunk_index}: "
                                f"{chunk.word_count} words → "
                                f"{token_count} tokens "
                                f"(section='{chunk.section}')"
                            )

                else:

                    print(
                        "\nNo chunks exceed the model "
                        "sequence length."
                    )

            except Exception as error:

                print(
                    "\nTokenization analysis FAILED"
                )

                print(
                    f"Error: {error}"
                )

    print("\n" + "=" * 80)
    print("OVERALL TOKENIZATION SUMMARY")
    print("=" * 80)

    if overall_token_counts:

        overall_min = min(
            overall_token_counts
        )

        overall_max = max(
            overall_token_counts
        )

        overall_avg = (
            sum(overall_token_counts)
            / len(overall_token_counts)
        )

        overall_percentage = (
            overall_oversized_chunks
            / overall_total_chunks
            * 100
        )

        print(
            f"\nTotal chunks analyzed: "
            f"{overall_total_chunks}"
        )

        print(
            f"Minimum tokens: "
            f"{overall_min}"
        )

        print(
            f"Maximum tokens: "
            f"{overall_max}"
        )

        print(
            f"Average tokens: "
            f"{overall_avg:.2f}"
        )

        print(
            f"\nChunks exceeding model limit: "
            f"{overall_oversized_chunks}"
            f"/{overall_total_chunks}"
        )

        print(
            f"Percentage exceeding model limit: "
            f"{overall_percentage:.2f}%"
        )

        if overall_oversized_chunks == 0:

            print(
                "\nRESULT: PASS"
            )

            print(
                "No chunks exceed the model's "
                "maximum sequence length."
            )

        else:

            print(
                "\nRESULT: ATTENTION REQUIRED"
            )

            print(
                "Some chunks are longer than the "
                "model's maximum sequence length "
                "and may be truncated during "
                "embedding generation."
            )

    else:

        print(
            "\nNo chunks were analyzed."
        )

    print("\n" + "=" * 80)
    print("TOKENIZATION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()