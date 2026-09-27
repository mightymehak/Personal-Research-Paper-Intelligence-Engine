import os
import uuid
import zipfile
from pathlib import Path


BASE_DATA_DIR = Path("data/raw")


def create_library_directory() -> tuple[str, Path]:
    """
    Creates a unique directory for a research paper library.

    Returns:
        tuple: (library_id, library_directory)
    """

    library_id = f"library_{uuid.uuid4().hex[:8]}"

    library_dir = BASE_DATA_DIR / library_id
    library_dir.mkdir(parents=True, exist_ok=True)

    return library_id, library_dir


def validate_zip_file(zip_path: Path) -> list[str]:
    """
    Validates that the ZIP file contains only supported PDF files.

    Returns:
        List of PDF filenames found inside the ZIP.

    Raises:
        ValueError: If the ZIP is invalid or contains unsupported files.
    """

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Uploaded file is not a valid ZIP archive.")

    with zipfile.ZipFile(zip_path, "r") as zip_file:

        members = zip_file.namelist()

        # Ignore directories
        files = [
            member
            for member in members
            if not member.endswith("/")
        ]

        if not files:
            raise ValueError("The ZIP archive is empty.")

        unsupported_files = [
            file
            for file in files
            if not file.lower().endswith(".pdf")
        ]

        if unsupported_files:
            raise ValueError(
                f"ZIP contains unsupported files: {unsupported_files}"
            )

        pdf_files = [
            file
            for file in files
            if file.lower().endswith(".pdf")
        ]

        if not pdf_files:
            raise ValueError("No PDF files found in the ZIP archive.")

        return pdf_files


def safe_extract_zip(zip_path: Path, destination: Path) -> list[Path]:
    """
    Safely extracts PDF files from a ZIP archive.

    Prevents path traversal attacks such as:
        ../../malicious_file.pdf

    Returns:
        List of extracted PDF paths.
    """

    extracted_files = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:

        for member in zip_file.infolist():

            if member.is_dir():
                continue

            filename = Path(member.filename)

            # Only allow PDFs
            if filename.suffix.lower() != ".pdf":
                continue

            # Prevent path traversal
            target_path = (destination / filename).resolve()
            destination_path = destination.resolve()

            if not str(target_path).startswith(str(destination_path)):
                raise ValueError(
                    f"Unsafe file path detected: {member.filename}"
                )

            # Create parent directories if needed
            target_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with zip_file.open(member) as source:
                with open(target_path, "wb") as target:
                    target.write(source.read())

            extracted_files.append(target_path)

    return extracted_files