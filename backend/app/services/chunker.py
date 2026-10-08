from __future__ import annotations

import re
from typing import List

from sentence_transformers import SentenceTransformer

from backend.app.models.chunk import PaperChunk
from backend.app.models.paper import Paper


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_TOKENS = 200
DEFAULT_CHUNK_OVERLAP = 35

MIN_FRAGMENT_SIZE = 50


# Load the tokenizer once when this module is imported.
#
# We only need the tokenizer here, not the embedding model itself.
_embedding_model = SentenceTransformer(MODEL_NAME)
_tokenizer = _embedding_model.tokenizer


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving the textual content.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text: str) -> int:
    """
    Count whitespace-separated words.
    """

    if not text:
        return 0

    return len(text.split())


def token_count(text: str) -> int:
    """
    Count tokens using the exact tokenizer used by
    all-MiniLM-L6-v2.

    Special tokens are included because they are part of
    the actual model input.
    """

    if not text:
        return 0

    encoded = _tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    return len(encoded["input_ids"])


def split_text_by_tokens(
    text: str,
    max_tokens: int,
) -> List[str]:
    """
    Split text into pieces that stay within max_tokens.

    Splitting is performed at word boundaries rather than
    arbitrary token boundaries so that chunks remain readable.
    """

    text = normalize_text(text)

    if not text:
        return []

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than 0"
        )

    words = text.split()

    chunks = []
    current_words = []

    for word in words:

        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)

        candidate_tokens = token_count(candidate_text)

        if (
            current_words
            and candidate_tokens > max_tokens
        ):
            chunks.append(
                " ".join(current_words)
            )

            current_words = [word]

        elif candidate_tokens > max_tokens:
            # A single unusual word/token sequence can itself
            # exceed the limit.
            #
            # Fall back to tokenizer-level splitting.
            encoded = _tokenizer(
                word,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )

            word_token_ids = encoded["input_ids"]

            for start in range(
                0,
                len(word_token_ids),
                max_tokens,
            ):
                token_slice = word_token_ids[
                    start:start + max_tokens
                ]

                decoded = _tokenizer.decode(
                    token_slice,
                    skip_special_tokens=True,
                ).strip()

                if decoded:
                    chunks.append(decoded)

            current_words = []

        else:
            current_words = candidate_words

    if current_words:
        chunks.append(
            " ".join(current_words)
        )

    return chunks


def get_last_tokens(
    text: str,
    max_tokens: int,
) -> str:
    """
    Return approximately the last max_tokens from text,
    while keeping the overlap readable by preferring
    word boundaries.
    """

    if not text or max_tokens <= 0:
        return ""

    words = text.split()

    selected_words = []

    for word in reversed(words):

        candidate_words = [
            word
        ] + selected_words

        candidate_text = " ".join(
            candidate_words
        )

        if (
            token_count(candidate_text)
            <= max_tokens
        ):
            selected_words = candidate_words
        else:
            break

    return " ".join(selected_words)


def split_section_into_chunks(
    section_text: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a section into token-aware retrieval chunks.

    Strategy:

    1. Normalize the section.
    2. Split into paragraphs.
    3. Preserve paragraph boundaries whenever possible.
    4. Split exceptionally large paragraphs by tokens.
    5. Build chunks up to chunk_size tokens.
    6. Add overlap from the previous chunk.
    7. Guarantee that chunks stay within chunk_size tokens.
    """

    if not section_text or not section_text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

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

        paragraph = normalize_text(
            paragraph
        )

        if paragraph:
            paragraphs.append(
                paragraph
            )

    if not paragraphs:
        return []

    # ---------------------------------------------------------
    # Step 2: Split paragraphs that exceed token limit
    # ---------------------------------------------------------

    processed_paragraphs = []

    for paragraph in paragraphs:

        if (
            token_count(paragraph)
            <= chunk_size
        ):
            processed_paragraphs.append(
                paragraph
            )

        else:
            processed_paragraphs.extend(
                split_text_by_tokens(
                    paragraph,
                    chunk_size,
                )
            )

    # ---------------------------------------------------------
    # Step 3: Build base chunks without overlap
    # ---------------------------------------------------------

    base_chunks = []

    current_parts: List[str] = []
    current_tokens = 0

    for paragraph in processed_paragraphs:

        paragraph_tokens = token_count(
            paragraph
        )

        # Paragraph fits into current chunk.
        if (
            current_parts
            and current_tokens
            + paragraph_tokens
            <= chunk_size
        ):
            current_parts.append(
                paragraph
            )

            current_tokens += (
                paragraph_tokens
            )

            continue

        # Finalize current chunk.
        if current_parts:

            current_chunk = " ".join(
                current_parts
            ).strip()

            if current_chunk:
                base_chunks.append(
                    current_chunk
                )

        # Start a new chunk.
        current_parts = [
            paragraph
        ]

        current_tokens = (
            paragraph_tokens
        )

    # Final chunk.
    if current_parts:

        final_chunk = " ".join(
            current_parts
        ).strip()

        if final_chunk:
            base_chunks.append(
                final_chunk
            )

    # ---------------------------------------------------------
    # Step 4: Add token overlap
    # ---------------------------------------------------------

    final_chunks = []

    for index, base_chunk in enumerate(
        base_chunks
    ):

        if index == 0:

            final_chunks.append(
                normalize_text(
                    base_chunk
                )
            )

            continue

        previous_chunk = final_chunks[-1]

        overlap_text = get_last_tokens(
            previous_chunk,
            overlap,
        )

        if not overlap_text:

            final_chunks.append(
                base_chunk
            )

            continue

        overlap_tokens = token_count(
            overlap_text
        )

        available_tokens = (
            chunk_size
            - overlap_tokens
        )

        # The current base chunk should already
        # be <= chunk_size. We now need to leave
        # room for the overlap.
        current_tokens = token_count(
            base_chunk
        )

        if current_tokens <= available_tokens:

            combined = (
                overlap_text
                + " "
                + base_chunk
            )

            final_chunks.append(
                normalize_text(
                    combined
                )
            )

        else:

            # Keep only as much of the current
            # chunk as fits after the overlap.
            current_words = base_chunk.split()

            selected_words = []

            for word in current_words:

                candidate_words = (
                    selected_words
                    + [word]
                )

                candidate_text = " ".join(
                    candidate_words
                )

                candidate_tokens = (
                    token_count(
                        candidate_text
                    )
                )

                if (
                    candidate_tokens
                    <= available_tokens
                ):
                    selected_words.append(
                        word
                    )
                else:
                    break

            current_part = " ".join(
                selected_words
            )

            if current_part:

                combined = (
                    overlap_text
                    + " "
                    + current_part
                )

                final_chunks.append(
                    normalize_text(
                        combined
                    )
                )

            else:
                final_chunks.append(
                    base_chunk
                )

    # ---------------------------------------------------------
    # Step 5: Final validation
    # ---------------------------------------------------------

    validated_chunks = []

    for chunk in final_chunks:

        chunk = normalize_text(
            chunk
        )

        if not chunk:
            continue

        count = token_count(
            chunk
        )

        if count <= chunk_size:

            validated_chunks.append(
                chunk
            )

        else:

            # Absolute safety fallback.
            validated_chunks.extend(
                split_text_by_tokens(
                    chunk,
                    chunk_size,
                )
            )

    return validated_chunks


def chunk_paper(
    paper: Paper,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[PaperChunk]:
    """
    Convert a parsed Paper into section-aware,
    token-aware PaperChunk objects.

    References are intentionally excluded from
    retrieval chunks.
    """

    chunks: List[PaperChunk] = []

    chunk_index = 0

    for section_name, section_text in (
        paper.sections.items()
    ):

        section_chunks = (
            split_section_into_chunks(
                section_text=section_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        for chunk_text in section_chunks:

            chunk_text = normalize_text(
                chunk_text
            )

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
                    word_count=word_count(
                        chunk_text
                    ),
                    token_count=token_count(
                        chunk_text
                    ),
                )
            )

            chunk_index += 1

    return chunks