from pathlib import Path

from backend.app.services.pdf_parser import parse_pdf


PDF_DIRECTORY = Path("data/raw")


def main():
    """
    Test the PDF parser on every valid PDF
    found inside every research library.
    """

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

        # Recursively search for PDFs inside the library.
        # This supports ZIPs containing folders such as:
        #
        # library_xxxxx/
        # └── papers/
        #     ├── paper1.pdf
        #     └── paper2.pdf
        #
        # Ignore macOS-generated metadata files.
        pdf_files = [
            pdf
            for pdf in library_directory.rglob("*.pdf")
            if "__MACOSX" not in pdf.parts
            and not pdf.name.startswith("._")
        ]

        if not pdf_files:
            print("No PDF files found in this library.")
            continue

        print(f"\nFound {len(pdf_files)} PDF file(s).")

        for pdf_file in pdf_files:

            print("\n" + "-" * 80)
            print(f"FILE: {pdf_file.name}")
            print(f"PATH: {pdf_file}")
            print("-" * 80)

            try:

                paper = parse_pdf(pdf_file)

                print("\nPaper ID:")
                print(paper.paper_id)

                print("\nTitle:")
                if paper.title:
                    print(paper.title)
                else:
                    print("  No title detected.")

                print("\nAuthors:")
                if paper.authors:
                    for author in paper.authors:
                        print(f"  - {author}")
                else:
                    print("  No authors detected.")

                print("\nAbstract:")
                if paper.abstract:
                    print(paper.abstract[:1000])

                    if len(paper.abstract) > 1000:
                        print("  ...")
                else:
                    print("  No abstract detected.")

                print("\nKeywords:")
                if paper.keywords:
                    for keyword in paper.keywords:
                        print(f"  - {keyword}")
                else:
                    print("  No keywords detected.")

                print("\nSections:")
                if paper.sections:
                    for section_name, content in paper.sections.items():
                        print(
                            f"  - {section_name}: "
                            f"{len(content)} characters"
                        )
                else:
                    print("  No sections detected.")

                print("\nReferences:")
                print(
                    f"  {len(paper.references)} "
                    f"references detected."
                )

                print("\nFull text:")
                print(
                    f"  {len(paper.full_text)} "
                    f"characters"
                )

                print("\nParser status: SUCCESS")

            except Exception as error:

                print("\nParser status: FAILED")
                print(f"Error: {error}")

    print("\n" + "=" * 80)
    print("PDF PARSER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()