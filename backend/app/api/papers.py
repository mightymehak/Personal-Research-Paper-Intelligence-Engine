from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.services.zip_handler import (
    create_library_directory,
    validate_zip_file,
    safe_extract_zip,
)


router = APIRouter(
    prefix="/papers",
    tags=["Papers"]
)


@router.post("/upload")
async def upload_papers(
    file: UploadFile = File(...)
):
    """
    Upload a ZIP file containing research papers.
    """

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded."
        )

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP files are allowed."
        )

    library_id = None

    try:

        # Create library directory
        library_id, library_dir = create_library_directory()

        # Temporary ZIP path
        zip_path = library_dir / file.filename

        # Save uploaded ZIP
        contents = await file.read()

        with open(zip_path, "wb") as buffer:
            buffer.write(contents)

        # Validate ZIP
        pdf_files = validate_zip_file(zip_path)

        # Extract PDFs
        extracted_files = safe_extract_zip(
            zip_path,
            library_dir
        )

        # Remove original ZIP after extraction
        zip_path.unlink()

        return {
            "status": "success",
            "library_id": library_id,
            "papers_found": len(extracted_files),
            "papers": [
                path.name
                for path in extracted_files
            ]
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process ZIP: {str(error)}"
        )