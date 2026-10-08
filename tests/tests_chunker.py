from pathlib import Path

from backend.app.services.chunker import chunk_paper
from backend.app.services.pdf_parser import parse_pdf


PDF_DIRECTORY = Path("data/raw")

MAX_CHUNK_SIZE = 500


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
                    continue

                # -------------------------------------------------
                # Word-count statistics
                # -------------------------------------------------

                word_counts = [
                    chunk.word_count
                    for chunk in chunks
                ]

                print(
                    f"Minimum chunk size: "
                    f"{min(word_counts)} words"
                )

                print(
                    f"Maximum chunk size: "
                    f"{max(word_counts)} words"
                )

                print(
                    f"Average chunk size: "
                    f"{sum(word_counts) / len(word_counts):.2f} words"
                )

                # -------------------------------------------------
                # Check for oversized chunks
                # -------------------------------------------------

                oversized_chunks = [
                    chunk
                    for chunk in chunks
                    if chunk.word_count > MAX_CHUNK_SIZE
                ]

                print(
                    f"Oversized chunks "
                    f"(>{MAX_CHUNK_SIZE} words): "
                    f"{len(oversized_chunks)}"
                )

                if oversized_chunks:

                    print(
                        "\nWARNING: Oversized chunks detected:"
                    )

                    for chunk in oversized_chunks:

                        print(
                            f"  - {chunk.chunk_id}: "
                            f"{chunk.word_count} words"
                        )

                else:

                    print(
                        "All chunks are within the "
                        f"{MAX_CHUNK_SIZE}-word limit."
                    )

                # -------------------------------------------------
                # Section distribution
                # -------------------------------------------------

                sections = {}

                for chunk in chunks:

                    sections.setdefault(
                        chunk.section,
                        0
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
                        f"Chunk ID: {chunk.chunk_id}"
                    )

                    print(
                        f"Section: {chunk.section}"
                    )

                    print(
                        f"Index: {chunk.chunk_index}"
                    )

                    print(
                        f"Word count: {chunk.word_count}"
                    )

                    print("\nText:")

                    print(chunk.text[:700])

                    if len(chunk.text) > 700:
                        print("...")

                # -------------------------------------------------
                # Last chunk
                # -------------------------------------------------

                print("\nLast chunk:")

                last_chunk = chunks[-1]

                print("\n" + "." * 70)

                print(
                    f"Chunk ID: {last_chunk.chunk_id}"
                )

                print(
                    f"Section: {last_chunk.section}"
                )

                print(
                    f"Index: {last_chunk.chunk_index}"
                )

                print(
                    f"Word count: {last_chunk.word_count}"
                )

                print("\nText:")

                print(last_chunk.text[:700])

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

                else:

                    print(
                        "\nChunk index validation: FAIL"
                    )

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

                else:

                    print(
                        "Empty chunk validation: FAIL"
                    )

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

                else:

                    print(
                        "Word count validation: FAIL"
                    )

                    for chunk_id in incorrect_word_counts:

                        print(
                            f"  - {chunk_id}"
                        )

                # -------------------------------------------------
                # Overall paper validation
                # -------------------------------------------------

                if (
                    not oversized_chunks
                    and not empty_chunks
                    and actual_indices == expected_indices
                    and not incorrect_word_counts
                ):

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

    print("\n" + "=" * 80)
    print("CHUNKING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()