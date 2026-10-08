from __future__ import annotations

import re
from typing import List

from backend.app.models.chunk import PaperChunk
from backend.app.models.paper import Paper


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 75

# Very small fragments are usually not useful as independent
# retrieval chunks. They will be merged with neighboring content
# when possible.
MIN_FRAGMENT_SIZE = 50


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving the actual textual content.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text: str) -> int:
    """
    Count whitespace-separated words.

    NOTE:
    This is intentionally a word count rather than a true tokenizer
    count. In Phase 2B, this will be replaced by the tokenizer used
    by the Sentence Transformer embedding model.
    """

    if not text:
        return 0

    return len(text.split())


def split_long_paragraph(
    paragraph: str,
    chunk_size: int,
) -> List[str]:
    """
    Split a paragraph that is larger than the maximum chunk size.

    Long paragraphs cannot preserve paragraph boundaries, so they are
    split by words.

    No overlap is introduced here. Overlap is handled at the
    chunk-to-chunk level by the main chunking function.
    """

    words = paragraph.split()

    if not words:
        return []

    chunks = []

    for start in range(0, len(words), chunk_size):
        end = start + chunk_size

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def get_last_words(
    text: str,
    count: int,
) -> str:
    """
    Return the last `count` words from a piece of text.
    """

    if count <= 0:
        return ""

    words = text.split()

    if len(words) <= count:
        return text

    return " ".join(words[-count:])


def split_section_into_chunks(
    section_text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split one paper section into retrieval-friendly chunks.

    Strategy:

    1. Normalize the section.
    2. Split the section into paragraphs.
    3. Keep paragraphs together whenever possible.
    4. Split exceptionally large paragraphs.
    5. Enforce a hard maximum chunk size.
    6. Add approximately `overlap` words from the previous chunk.
    7. Avoid unnecessary tiny fragments.
    """

    if not section_text or not section_text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    # ---------------------------------------------------------
    # Step 1: Split section into paragraphs
    # ---------------------------------------------------------

    raw_paragraphs = re.split(
        r"\n\s*\n",
        section_text,
    )

    paragraphs = []

    for paragraph in raw_paragraphs:

        paragraph = normalize_text(paragraph)

        if paragraph:
            paragraphs.append(paragraph)

    if not paragraphs:
        return []

    # ---------------------------------------------------------
    # Step 2: Handle exceptionally long paragraphs
    # ---------------------------------------------------------

    processed_paragraphs = []

    for paragraph in paragraphs:

        if word_count(paragraph) <= chunk_size:
            processed_paragraphs.append(paragraph)

        else:
            processed_paragraphs.extend(
                split_long_paragraph(
                    paragraph,
                    chunk_size,
                )
            )

    # ---------------------------------------------------------
    # Step 3: Build chunks from paragraphs
    # ---------------------------------------------------------

    chunks = []

    current_parts: List[str] = []
    current_word_count = 0

    for paragraph in processed_paragraphs:

        paragraph_words = word_count(paragraph)

        # -----------------------------------------------------
        # If the paragraph itself fits into the current chunk
        # -----------------------------------------------------

        if (
            current_parts
            and current_word_count + paragraph_words <= chunk_size
        ):
            current_parts.append(paragraph)
            current_word_count += paragraph_words
            continue

        # -----------------------------------------------------
        # If adding the paragraph would exceed the limit,
        # finalize the current chunk first.
        # -----------------------------------------------------

        if current_parts:

            current_chunk = " ".join(current_parts).strip()

            if current_chunk:
                chunks.append(current_chunk)

            # -------------------------------------------------
            # Create overlap from the END of the finalized chunk.
            #
            # This is the important fix.
            #
            # We take exactly approximately `overlap` words
            # rather than accidentally including an entire
            # previous paragraph.
            # -------------------------------------------------

            overlap_text = get_last_words(
                current_chunk,
                overlap,
            )

            current_parts = []

            if overlap_text:
                current_parts.append(overlap_text)

            current_word_count = word_count(
                overlap_text
            )

        # -----------------------------------------------------
        # Add the new paragraph.
        # -----------------------------------------------------

        current_parts.append(paragraph)

        current_word_count += paragraph_words

        # -----------------------------------------------------
        # Safety check.
        #
        # This should normally only happen when the paragraph
        # was already split above.
        # -----------------------------------------------------

        if current_word_count > chunk_size:

            current_chunk = " ".join(
                current_parts
            ).strip()

            current_words = current_chunk.split()

            # Keep the most recent `chunk_size` words.
            current_chunk = " ".join(
                current_words[-chunk_size:]
            )

            chunks.append(current_chunk)

            overlap_text = get_last_words(
                current_chunk,
                overlap,
            )

            current_parts = [overlap_text]
            current_word_count = word_count(
                overlap_text
            )

    # ---------------------------------------------------------
    # Step 4: Final chunk
    # ---------------------------------------------------------

    if current_parts:

        final_chunk = " ".join(
            current_parts
        ).strip()

        if final_chunk:
            chunks.append(final_chunk)

    # ---------------------------------------------------------
    # Step 5: Clean up accidental duplicates / empty chunks
    # ---------------------------------------------------------

    cleaned_chunks = []

    for chunk in chunks:

        chunk = normalize_text(chunk)

        if not chunk:
            continue

        cleaned_chunks.append(chunk)

    # ---------------------------------------------------------
    # Step 6: Handle tiny fragments
    #
    # We don't delete legitimate short sections. This function
    # only merges a tiny chunk when it is clearly an accidental
    # fragment created by splitting.
    # ---------------------------------------------------------

    final_chunks = []

    for chunk in cleaned_chunks:

        current_count = word_count(chunk)

        # Keep the first chunk even if it is short.
        if not final_chunks:
            final_chunks.append(chunk)
            continue

        # If this is a very small fragment and adding it to the
        # previous chunk would still respect the hard maximum,
        # merge it.
        if (
            current_count < MIN_FRAGMENT_SIZE
            and word_count(final_chunks[-1]) + current_count
            <= chunk_size
        ):
            final_chunks[-1] = (
                final_chunks[-1] + " " + chunk
            ).strip()

        else:
            final_chunks.append(chunk)

    return final_chunks


def chunk_paper(
    paper: Paper,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[PaperChunk]:
    """
    Convert a parsed Paper into section-aware PaperChunk objects.

    References are intentionally excluded from the retrieval
    chunks. They will be useful later for citation-graph
    functionality.
    """

    chunks: List[PaperChunk] = []

    chunk_index = 0

    for section_name, section_text in paper.sections.items():

        section_chunks = split_section_into_chunks(
            section_text=section_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_text in section_chunks:

            chunk_text = normalize_text(chunk_text)

            if not chunk_text:
                continue

            chunks.append(
                PaperChunk(
                    chunk_id=(
                        f"{paper.paper_id}"
                        f"_chunk_{chunk_index}"
                    ),
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    section=section_name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    word_count=word_count(chunk_text),
                )
            )

            chunk_index += 1

    return chunks