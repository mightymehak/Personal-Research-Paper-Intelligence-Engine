from pathlib import Path

from backend.app.services.chunker import chunk_paper
from backend.app.services.pdf_parser import parse_pdf


PDF_DIRECTORY = Path("data/raw")

# The chunker targets approximately 200 tokens.
# We validate against the same hard target.
MAX_CHUNK_TOKENS = 200


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

    overall_success = True

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

        if not pdf_files:
            print("No PDF files found in this library.")
            continue

        for pdf_file in pdf_files:

            print("\n" + "-" * 80)
            print(f"FILE: {pdf_file.name}")
            print("-" * 80)

            try:

                # -------------------------------------------------
                # Parse PDF
                # -------------------------------------------------

                paper = parse_pdf(pdf_file)

                # -------------------------------------------------
                # Generate chunks
                # -------------------------------------------------

                chunks = chunk_paper(paper)

                print(f"\nTitle: {paper.title}")
                print(f"Total chunks: {len(chunks)}")

                if not chunks:
                    print("No chunks generated.")
                    overall_success = False
                    continue

                # -------------------------------------------------
                # Word-count statistics
                # -------------------------------------------------

                word_counts = [
                    chunk.word_count
                    for chunk in chunks
                ]

                print("\nWord statistics:")

                print(
                    f"  Minimum words: "
                    f"{min(word_counts)}"
                )

                print(
                    f"  Maximum words: "
                    f"{max(word_counts)}"
                )

                print(
                    f"  Average words: "
                    f"{sum(word_counts) / len(word_counts):.2f}"
                )

                # -------------------------------------------------
                # Token-count statistics
                # -------------------------------------------------

                token_counts = [
                    chunk.token_count
                    for chunk in chunks
                ]

                print("\nToken statistics:")

                print(
                    f"  Minimum tokens: "
                    f"{min(token_counts)}"
                )

                print(
                    f"  Maximum tokens: "
                    f"{max(token_counts)}"
                )

                print(
                    f"  Average tokens: "
                    f"{sum(token_counts) / len(token_counts):.2f}"
                )

                # -------------------------------------------------
                # Token limit validation
                # -------------------------------------------------

                oversized_token_chunks = [
                    chunk
                    for chunk in chunks
                    if chunk.token_count
                    > MAX_CHUNK_TOKENS
                ]

                print(
                    "\nToken limit validation:"
                )

                print(
                    f"  Maximum allowed: "
                    f"{MAX_CHUNK_TOKENS} tokens"
                )

                print(
                    f"  Oversized chunks: "
                    f"{len(oversized_token_chunks)}"
                )

                if oversized_token_chunks:

                    print(
                        "\nWARNING: Oversized token chunks:"
                    )

                    for chunk in oversized_token_chunks:

                        print(
                            f"  - {chunk.chunk_id}: "
                            f"{chunk.token_count} tokens "
                            f"({chunk.word_count} words)"
                        )

                    token_limit_pass = False
                    overall_success = False

                else:

                    print(
                        "Token limit validation: PASS"
                    )

                    token_limit_pass = True

                # -------------------------------------------------
                # Section distribution
                # -------------------------------------------------

                sections = {}

                for chunk in chunks:

                    sections.setdefault(
                        chunk.section,
                        0,
                    )

                    sections[chunk.section] += 1

                print("\nChunks by section:")

                for section, count in sections.items():

                    print(
                        f"  - {section}: {count}"
                    )

                # -------------------------------------------------
                # First 3 chunks
                # -------------------------------------------------

                print("\nFirst 3 chunks:")

                for chunk in chunks[:3]:

                    print("\n" + "." * 70)

                    print(
                        f"Chunk ID: "
                        f"{chunk.chunk_id}"
                    )

                    print(
                        f"Section: "
                        f"{chunk.section}"
                    )

                    print(
                        f"Index: "
                        f"{chunk.chunk_index}"
                    )

                    print(
                        f"Word count: "
                        f"{chunk.word_count}"
                    )

                    print(
                        f"Token count: "
                        f"{chunk.token_count}"
                    )

                    print("\nText:")

                    print(
                        chunk.text[:700]
                    )

                    if len(chunk.text) > 700:
                        print("...")

                # -------------------------------------------------
                # Last chunk
                # -------------------------------------------------

                print("\nLast chunk:")

                last_chunk = chunks[-1]

                print("\n" + "." * 70)

                print(
                    f"Chunk ID: "
                    f"{last_chunk.chunk_id}"
                )

                print(
                    f"Section: "
                    f"{last_chunk.section}"
                )

                print(
                    f"Index: "
                    f"{last_chunk.chunk_index}"
                )

                print(
                    f"Word count: "
                    f"{last_chunk.word_count}"
                )

                print(
                    f"Token count: "
                    f"{last_chunk.token_count}"
                )

                print("\nText:")

                print(
                    last_chunk.text[:700]
                )

                if len(last_chunk.text) > 700:
                    print("...")

                # -------------------------------------------------
                # Chunk index validation
                # -------------------------------------------------

                expected_indices = list(
                    range(len(chunks))
                )

                actual_indices = [
                    chunk.chunk_index
                    for chunk in chunks
                ]

                if actual_indices == expected_indices:

                    print(
                        "\nChunk index validation: PASS"
                    )

                    index_validation = True

                else:

                    print(
                        "\nChunk index validation: FAIL"
                    )

                    index_validation = False
                    overall_success = False

                # -------------------------------------------------
                # Empty-text validation
                # -------------------------------------------------

                empty_chunks = [
                    chunk
                    for chunk in chunks
                    if not chunk.text.strip()
                ]

                if not empty_chunks:

                    print(
                        "Empty chunk validation: PASS"
                    )

                    empty_validation = True

                else:

                    print(
                        "Empty chunk validation: FAIL"
                    )

                    for chunk in empty_chunks:

                        print(
                            f"  - {chunk.chunk_id}"
                        )

                    empty_validation = False
                    overall_success = False

                # -------------------------------------------------
                # Word-count validation
                # -------------------------------------------------

                incorrect_word_counts = []

                for chunk in chunks:

                    actual_count = len(
                        chunk.text.split()
                    )

                    if actual_count != chunk.word_count:

                        incorrect_word_counts.append(
                            chunk.chunk_id
                        )

                if not incorrect_word_counts:

                    print(
                        "Word count validation: PASS"
                    )

                    word_count_validation = True

                else:

                    print(
                        "Word count validation: FAIL"
                    )

                    for chunk_id in (
                        incorrect_word_counts
                    ):

                        print(
                            f"  - {chunk_id}"
                        )

                    word_count_validation = False
                    overall_success = False

                # -------------------------------------------------
                # Token-count validation
                # -------------------------------------------------

                incorrect_token_counts = []

                for chunk in chunks:

                    if chunk.token_count <= 0:

                        incorrect_token_counts.append(
                            chunk.chunk_id
                        )

                if not incorrect_token_counts:

                    print(
                        "Token count validation: PASS"
                    )

                    token_count_validation = True

                else:

                    print(
                        "Token count validation: FAIL"
                    )

                    for chunk_id in (
                        incorrect_token_counts
                    ):

                        print(
                            f"  - {chunk_id}"
                        )

                    token_count_validation = False
                    overall_success = False

                # -------------------------------------------------
                # Token limit vs model limit
                # -------------------------------------------------

                model_max_tokens = 256

                exceeding_model_limit = [
                    chunk
                    for chunk in chunks
                    if chunk.token_count
                    > model_max_tokens
                ]

                print(
                    "\nModel limit validation:"
                )

                print(
                    f"  Model maximum: "
                    f"{model_max_tokens} tokens"
                )

                print(
                    f"  Chunks exceeding model limit: "
                    f"{len(exceeding_model_limit)}"
                )

                if not exceeding_model_limit:

                    print(
                        "Model limit validation: PASS"
                    )

                    model_limit_validation = True

                else:

                    print(
                        "Model limit validation: FAIL"
                    )

                    for chunk in (
                        exceeding_model_limit
                    ):

                        print(
                            f"  - {chunk.chunk_id}: "
                            f"{chunk.token_count} tokens"
                        )

                    model_limit_validation = False
                    overall_success = False

                # -------------------------------------------------
                # Overall paper validation
                # -------------------------------------------------

                paper_success = (
                    token_limit_pass
                    and index_validation
                    and empty_validation
                    and word_count_validation
                    and token_count_validation
                    and model_limit_validation
                )

                if paper_success:

                    print(
                        "\nChunk validation: SUCCESS"
                    )

                else:

                    print(
                        "\nChunk validation: FAILED"
                    )

            except Exception as error:

                print(
                    "\nChunking status: FAILED"
                )

                print(
                    f"Error: {error}"
                )

                overall_success = False

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("CHUNKING TEST COMPLETE")
    print("=" * 80)

    if overall_success:

        print(
            "\nOVERALL RESULT: SUCCESS"
        )

        print(
            "All chunks passed token, word, index, "
            "empty-text, and model-limit validation."
        )

    else:

        print(
            "\nOVERALL RESULT: FAILED"
        )

        print(
            "One or more validation checks failed."
        )


if __name__ == "__main__":
    main()