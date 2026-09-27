from __future__ import annotations

import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz

from backend.app.models.paper import Paper


# ============================================================================
# CONFIGURATION
# ============================================================================

COMMON_SECTION_NAMES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "related studies",
    "literature review",
    "literature survey",
    "preliminaries",
    "problem statement",
    "problem formulation",
    "research question",
    "research questions",
    "methodology",
    "methods",
    "method",
    "materials and methods",
    "approach",
    "proposed method",
    "proposed approach",
    "system design",
    "system architecture",
    "architecture",
    "implementation",
    "design",
    "experimental setup",
    "experiments",
    "experimental results",
    "results",
    "evaluation",
    "evaluation methodology",
    "discussion",
    "results and discussion",
    "analysis",
    "limitations",
    "threats to validity",
    "future work",
    "future directions",
    "lessons and future work",
    "conclusion",
    "conclusions",
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
    "benchmarks",
    "benchmark",
    "challenges and open problems",
    "current systems",
    "query processing",
    "storage and indexing",
    "optimization and execution",
    "query optimization and execution",
    "indexing",
    "query execution",
    "query optimization",
    "similarity scores",
    "query types and basic operators",
    "hybrid operators",
    "plan enumeration",
    "plan selection",
    "references",
    "bibliography",
    "appendix",
    "appendices",
}


REFERENCE_SECTION_NAMES = {
    "references",
    "bibliography",
}


AFFILIATION_WORDS = {
    "university",
    "department",
    "faculty",
    "school",
    "college",
    "institute",
    "laboratory",
    "laboratories",
    "lab",
    "research",
    "center",
    "centre",
    "hospital",
    "company",
    "corporation",
    "inc",
    "ltd",
    "llc",
    "campus",
}


LOCATION_WORDS = {
    "usa",
    "us",
    "united states",
    "canada",
    "uk",
    "united kingdom",
    "india",
    "china",
    "germany",
    "france",
    "italy",
    "spain",
    "australia",
    "california",
    "new york",
    "texas",
    "london",
    "toronto",
    "west lafayette",
    "indiana",
}


# These are strong signals that something is NOT a section heading.
NON_SECTION_START_WORDS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "finally",
    "however",
    "therefore",
    "thus",
    "hence",
    "moreover",
    "furthermore",
    "although",
    "because",
    "while",
    "when",
    "where",
    "given",
    "during",
    "using",
    "used",
    "consider",
    "considering",
    "suppose",
    "assuming",
    "note",
    "notice",
    "e.g",
    "eg",
    "for example",
    "in this",
    "in our",
    "in the",
    "the",
    "this",
    "these",
    "those",
    "such",
    "some",
    "many",
    "most",
    "each",
    "another",
    "another",
    "as",
    "and",
    "or",
    "on",
    "from",
    "for",
    "with",
    "without",
    "based",
    "according",
    "during",
    "after",
    "before",
    "results",
    "discussion",
}


# ============================================================================
# BASIC TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving paragraph structure.
    """

    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    lines = [line.rstrip() for line in text.splitlines()]

    cleaned_lines: List[str] = []
    previous_blank = False

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            if not previous_blank:
                cleaned_lines.append("")

            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    text = "\n".join(cleaned_lines)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """
    Collapse arbitrary whitespace.
    """

    return re.sub(r"\s+", " ", text).strip()


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for comparisons.
    """

    text = text.lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_heading(text: str) -> str:
    """
    Normalize a section heading while removing numbering.
    """

    text = normalize_whitespace(text)

    text = re.sub(
        r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[\s.)-]+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip(" .:-")


# ============================================================================
# PDF LAYOUT EXTRACTION
# ============================================================================

def extract_pdf_pages(
    pdf_path: Path,
) -> List[Dict[str, Any]]:
    """
    Extract layout-aware PDF information.

    Each line stores:
        - text
        - bbox
        - font size
        - maximum font size
        - bold flag
        - spans
    """

    pages: List[Dict[str, Any]] = []

    with fitz.open(pdf_path) as document:

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            page_dict = page.get_text("dict")

            page_width = float(
                page.rect.width
            )

            page_height = float(
                page.rect.height
            )

            blocks: List[Dict[str, Any]] = []

            for block in page_dict.get(
                "blocks",
                [],
            ):

                if block.get("type") != 0:
                    continue

                block_lines: List[
                    Dict[str, Any]
                ] = []

                for raw_line in block.get(
                    "lines",
                    [],
                ):

                    spans: List[
                        Dict[str, Any]
                    ] = []

                    text_parts: List[str] = []

                    for span in raw_line.get(
                        "spans",
                        [],
                    ):

                        span_text = span.get(
                            "text",
                            "",
                        )

                        if not span_text:
                            continue

                        font_size = float(
                            span.get(
                                "size",
                                0.0,
                            )
                        )

                        font_name = str(
                            span.get(
                                "font",
                                "",
                            )
                        )

                        flags = int(
                            span.get(
                                "flags",
                                0,
                            )
                        )

                        is_bold = (
                            bool(flags & 16)
                            or "bold"
                            in font_name.lower()
                        )

                        spans.append(
                            {
                                "text": span_text,
                                "size": font_size,
                                "font": font_name,
                                "flags": flags,
                                "bold": is_bold,
                                "bbox": tuple(
                                    span.get(
                                        "bbox",
                                        (0, 0, 0, 0),
                                    )
                                ),
                            }
                        )

                        text_parts.append(
                            span_text
                        )

                    if not spans:
                        continue

                    line_text = "".join(
                        text_parts
                    ).strip()

                    if not line_text:
                        continue

                    line_bbox = tuple(
                        raw_line.get(
                            "bbox",
                            (0, 0, 0, 0),
                        )
                    )

                    sizes = [
                        span["size"]
                        for span in spans
                        if span["size"] > 0
                    ]

                    average_font_size = (
                        sum(sizes) / len(sizes)
                        if sizes
                        else 0.0
                    )

                    max_font_size = (
                        max(sizes)
                        if sizes
                        else 0.0
                    )

                    block_lines.append(
                        {
                            "text": line_text,
                            "bbox": line_bbox,
                            "spans": spans,
                            "font_size": average_font_size,
                            "max_font_size": max_font_size,
                            "bold": any(
                                span["bold"]
                                for span in spans
                            ),
                        }
                    )

                if not block_lines:
                    continue

                block_text = "\n".join(
                    line["text"]
                    for line in block_lines
                )

                blocks.append(
                    {
                        "text": block_text,
                        "bbox": tuple(
                            block.get(
                                "bbox",
                                (0, 0, 0, 0),
                            )
                        ),
                        "lines": block_lines,
                    }
                )

            pages.append(
                {
                    "page_number": page_number,
                    "width": page_width,
                    "height": page_height,
                    "blocks": blocks,
                }
            )

    return pages


# ============================================================================
# PAGE HELPERS
# ============================================================================

def get_all_page_lines(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten all page lines.
    """

    lines: List[Dict[str, Any]] = []

    for page in pages:

        for block in page["blocks"]:

            for line in block["lines"]:

                item = dict(line)

                item["page_number"] = (
                    page["page_number"]
                )

                item["page_width"] = (
                    page["width"]
                )

                item["page_height"] = (
                    page["height"]
                )

                lines.append(item)

    return lines


def extract_plain_text_from_pages(
    pages: List[Dict[str, Any]],
) -> str:
    """
    Reconstruct readable text.
    """

    page_texts: List[str] = []

    for page in pages:

        block_texts: List[str] = []

        for block in page["blocks"]:

            text = block["text"].strip()

            if text:
                block_texts.append(text)

        if block_texts:
            page_texts.append(
                "\n\n".join(
                    block_texts
                )
            )

    return clean_text(
        "\n\n".join(
            page_texts
        )
    )


# ============================================================================
# HEADER / FOOTER DETECTION
# ============================================================================

def _normalized_edge_text(
    text: str,
) -> str:

    text = normalize_for_matching(
        text
    )

    text = re.sub(
        r"\bpage\s+\d+\b",
        "page",
        text,
    )

    text = re.sub(
        r"\b\d+\s*/\s*\d+\b",
        "",
        text,
    )

    text = re.sub(
        r"\b\d+\b",
        "",
        text,
    )

    return normalize_whitespace(
        text
    )


def detect_repeated_headers_footers(
    pages: List[Dict[str, Any]],
) -> Tuple[Set[str], Set[str]]:
    """
    Detect repeated running headers and footers.
    """

    if len(pages) <= 1:
        return set(), set()

    top_counter: Counter = Counter()
    bottom_counter: Counter = Counter()

    for page in pages:

        height = page["height"]

        for block in page["blocks"]:

            bbox = block["bbox"]

            text = normalize_whitespace(
                block["text"]
            )

            if not text:
                continue

            normalized = (
                _normalized_edge_text(
                    text
                )
            )

            if len(normalized) < 4:
                continue

            if bbox[1] <= height * 0.12:
                top_counter[
                    normalized
                ] += 1

            if bbox[3] >= height * 0.88:
                bottom_counter[
                    normalized
                ] += 1

    repeated_headers = {
        text
        for text, count
        in top_counter.items()
        if count >= 2
    }

    repeated_footers = {
        text
        for text, count
        in bottom_counter.items()
        if count >= 2
    }

    return (
        repeated_headers,
        repeated_footers,
    )


def remove_repeated_headers_footers(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    (
        repeated_headers,
        repeated_footers,
    ) = detect_repeated_headers_footers(
        pages
    )

    if not repeated_headers and not repeated_footers:
        return pages

    cleaned_pages: List[
        Dict[str, Any]
    ] = []

    for page in pages:

        new_page = dict(page)

        new_blocks: List[
            Dict[str, Any]
        ] = []

        height = page["height"]

        for block in page["blocks"]:

            text = normalize_whitespace(
                block["text"]
            )

            normalized = (
                _normalized_edge_text(
                    text
                )
            )

            bbox = block["bbox"]

            is_header = (
                bbox[1] <= height * 0.12
                and normalized
                in repeated_headers
            )

            is_footer = (
                bbox[3] >= height * 0.88
                and normalized
                in repeated_footers
            )

            if is_header or is_footer:
                continue

            new_blocks.append(
                block
            )

        new_page["blocks"] = (
            new_blocks
        )

        cleaned_pages.append(
            new_page
        )

    return cleaned_pages


# ============================================================================
# TITLE EXTRACTION
# ============================================================================

def _looks_like_title(
    text: str,
) -> bool:

    text = normalize_whitespace(
        text
    )

    if not text:
        return False

    if len(text) < 8:
        return False

    if len(text) > 300:
        return False

    lower = text.lower()

    bad_patterns = [
        "abstract",
        "keywords",
        "key words",
        "contents",
        "copyright",
        "permission",
        "proceedings",
        "doi:",
        "preprint",
        "manuscript no",
        "received",
        "accepted",
        "published",
        "arxiv.org",
    ]

    if any(
        pattern in lower
        for pattern in bad_patterns
    ):
        return False

    if not re.search(
        r"[A-Za-z]",
        text,
    ):
        return False

    return True


def _estimate_body_font_size(
    pages: List[Dict[str, Any]],
) -> float:

    sizes: List[float] = []

    for page in pages[:10]:

        for block in page["blocks"]:

            for line in block["lines"]:

                text = normalize_whitespace(
                    line["text"]
                )

                if len(text) < 30:
                    continue

                size = float(
                    line.get(
                        "font_size",
                        0.0,
                    )
                )

                if size > 0:
                    sizes.append(
                        round(
                            size,
                            1,
                        )
                    )

    if not sizes:
        return 10.0

    return float(
        Counter(
            sizes
        ).most_common(1)[0][0]
    )


def extract_title(
    pages: List[Dict[str, Any]],
) -> str:

    if not pages:
        return ""

    body_size = (
        _estimate_body_font_size(
            pages
        )
    )

    candidates: List[
        Dict[str, Any]
    ] = []

    for page_index, page in enumerate(
        pages[:3]
    ):

        page_width = page["width"]

        for block in page["blocks"]:

            text = normalize_whitespace(
                block["text"]
            )

            if not _looks_like_title(
                text
            ):
                continue

            bbox = block["bbox"]

            vertical_position = (
                bbox[1]
                / max(
                    page["height"],
                    1,
                )
            )

            if vertical_position > 0.55:
                continue

            sizes = [
                line["font_size"]
                for line in block["lines"]
                if line["font_size"] > 0
            ]

            if not sizes:
                continue

            average_size = (
                sum(sizes)
                / len(sizes)
            )

            max_size = max(
                sizes
            )

            bold = any(
                line["bold"]
                for line in block["lines"]
            )

            center_x = (
                bbox[0] + bbox[2]
            ) / 2

            center_distance = abs(
                center_x
                - page_width / 2
            )

            centered_score = max(
                0.0,
                1.0
                - (
                    center_distance
                    / max(
                        page_width / 2,
                        1,
                    )
                ),
            )

            ratio = (
                average_size
                / max(
                    body_size,
                    1,
                )
            )

            score = (
                min(
                    ratio,
                    3.0,
                )
                * 4.0
            )

            if bold:
                score += 2.0

            score += (
                centered_score
                * 2.0
            )

            score += max(
                0.0,
                3.0 - page_index,
            )

            if 20 <= len(text) <= 180:
                score += 2.0

            candidates.append(
                {
                    "text": text,
                    "score": score,
                    "font_size": max_size,
                    "page": page_index,
                }
            )

    if candidates:

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["font_size"],
                -item["page"],
            ),
            reverse=True,
        )

        title = candidates[0][
            "text"
        ]

        if _looks_like_title(
            title
        ):
            return title

    for page in pages[:2]:

        for block in page["blocks"]:

            text = normalize_whitespace(
                block["text"]
            )

            if _looks_like_title(
                text
            ):
                return text

    return ""


# ============================================================================
# AUTHOR EXTRACTION
# ============================================================================

def _is_affiliation_text(
    text: str,
) -> bool:

    normalized = normalize_for_matching(
        text
    )

    if not normalized:
        return False

    words = set(
        normalized.split()
    )

    if words.intersection(
        AFFILIATION_WORDS
    ):
        return True

    if any(
        location in normalized
        for location in LOCATION_WORDS
    ):
        return True

    if "@" in text:
        return True

    if re.search(
        r"\b\d{5,6}\b",
        text,
    ):
        return True

    return False


def _looks_like_person_name(
    text: str,
) -> bool:

    text = normalize_whitespace(
        text
    )

    if not text:
        return False

    if len(text) > 100:
        return False

    if _is_affiliation_text(
        text
    ):
        return False

    if re.search(
        r"[@:/]",
        text,
    ):
        return False

    words = text.split()

    if not 2 <= len(words) <= 6:
        return False

    if any(
        not re.search(
            r"[A-Za-z]",
            word,
        )
        for word in words
    ):
        return False

    if len(words) >= 4 and re.search(
        r"\b(the|and|for|with|from|using|"
        r"based|system|database|vector|"
        r"search|paper|concepts)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False

    capitalized_words = sum(
        bool(
            re.match(
                r"^[A-Z][A-Za-z'`-]*$",
                word.strip(
                    ".,;:"
                ),
            )
        )
        for word in words
    )

    return capitalized_words >= 2


def _clean_author_candidate(
    text: str,
) -> str:

    text = normalize_whitespace(
        text
    )

    text = re.sub(
        r"\b(corresponding author|"
        r"email|e-mail)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[†‡*]+",
        "",
        text,
    )

    text = re.sub(
        r"(?<=\D)\d+$",
        "",
        text,
    )

    return text.strip(
        " ,;|.-"
    )


def _split_authors(
    text: str,
) -> List[str]:

    text = normalize_whitespace(
        text
    )

    if not text:
        return []

    text = re.sub(
        r"^(authors?|by)\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Important: many papers use a middle dot
    # between authors.
    text = text.replace(
        "·",
        ";",
    )

    text = text.replace(
        "•",
        ";",
    )

    text = re.sub(
        r"\s+and\s+",
        ";",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*&\s*",
        ";",
        text,
    )

    # Remove affiliation tail.
    text = re.split(
        r"\b(?:department|school|faculty|"
        r"university|institute|college|"
        r"laboratory|lab)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    if ";" not in text:

        comma_parts = re.split(
            r",\s*",
            text,
        )

        if len(comma_parts) > 1:

            if all(
                _looks_like_person_name(
                    part
                )
                for part in comma_parts
            ):
                text = ";".join(
                    comma_parts
                )

    candidates = re.split(
        r";|\n",
        text,
    )

    authors: List[str] = []

    for candidate in candidates:

        candidate = (
            _clean_author_candidate(
                candidate
            )
        )

        if _looks_like_person_name(
            candidate
        ):
            authors.append(
                candidate
            )

    return authors


def extract_authors(
    pages: List[Dict[str, Any]],
    title: str,
) -> List[str]:

    if not pages:
        return []

    first_page = pages[0]

    title_normalized = (
        normalize_for_matching(
            title
        )
    )

    title_bottom: Optional[
        float
    ] = None

    for block in first_page[
        "blocks"
    ]:

        block_text = (
            normalize_for_matching(
                block["text"]
            )
        )

        if (
            title_normalized
            and title_normalized
            in block_text
        ):

            title_bottom = (
                block["bbox"][3]
            )

            break

    if title_bottom is None:
        title_bottom = (
            first_page["height"]
            * 0.15
        )

    authors: List[str] = []

    for block in first_page[
        "blocks"
    ]:

        bbox = block["bbox"]

        if (
            bbox[1]
            < title_bottom - 2
        ):
            continue

        if (
            bbox[1]
            > first_page["height"]
            * 0.60
        ):
            continue

        for line in block[
            "lines"
        ]:

            text = normalize_whitespace(
                line["text"]
            )

            if not text:
                continue

            normalized = (
                normalize_for_matching(
                    text
                )
            )

            # ----------------------------------------------------------
            # Hard stop conditions.
            # ----------------------------------------------------------

            if re.match(
                r"^(abstract|keywords?|"
                r"key words?|ccs concepts?)\b",
                normalized,
            ):
                return authors[:30]

            if (
                "acm reference format"
                in normalized
            ):
                return authors[:30]

            if re.match(
                r"^(received|accepted|"
                r"published|copyright)\b",
                normalized,
            ):
                continue

            if _is_affiliation_text(
                text
            ):
                continue

            extracted = _split_authors(
                text
            )

            for author in extracted:

                if (
                    author
                    and author
                    not in authors
                ):
                    authors.append(
                        author
                    )

    return authors[:30]


# ============================================================================
# ABSTRACT
# ============================================================================

def extract_abstract(
    pages: List[Dict[str, Any]],
) -> str:

    full_text = (
        extract_plain_text_from_pages(
            pages[:3]
        )
    )

    if not full_text:
        return ""

    patterns = [
        r"(?is)\babstract\b\s*[:.\-]?\s*(.*?)(?=\n\s*(?:keywords?|key words?)\b)",

        r"(?is)\babstract\b\s*[:.\-]?\s*(.*?)(?=\n\s*(?:\d+[\.)]?\s*)?introduction\b)",

        r"(?is)\babstract\b\s*[:.\-]?\s*(.*?)(?=\n\s*(?:\d+[\.)]?\s*)?[A-Z][A-Za-z\s&-]{2,50}\n)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            full_text,
        )

        if not match:
            continue

        abstract = clean_text(
            match.group(1)
        )

        if len(abstract) >= 80:
            return abstract

    lines = full_text.splitlines()

    started = False

    abstract_lines: List[
        str
    ] = []

    for line in lines:

        normalized = (
            normalize_for_matching(
                line
            )
        )

        if not started:

            if re.match(
                r"^abstract\b",
                normalized,
            ):

                started = True

                remainder = re.sub(
                    r"^abstract\b\s*[:.\-]?\s*",
                    "",
                    line,
                    flags=re.IGNORECASE,
                )

                if remainder.strip():
                    abstract_lines.append(
                        remainder
                    )

            continue

        if re.match(
            r"^(keywords?|key words?)\b",
            normalized,
        ):
            break

        if re.match(
            r"^(?:\d+[\.)]?\s*)?introduction\b",
            normalized,
        ):
            break

        abstract_lines.append(
            line
        )

    abstract = clean_text(
        "\n".join(
            abstract_lines
        )
    )

    return (
        abstract
        if len(abstract) >= 50
        else ""
    )


# ============================================================================
# KEYWORDS
# ============================================================================

def _split_keywords(
    text: str,
) -> List[str]:

    text = normalize_whitespace(
        text
    )

    if not text:
        return []

    text = re.sub(
        r"^(keywords?|key words?)\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.split(
        r"\b(?:acm reference format|"
        r"reference format|doi|copyright)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    parts = re.split(
        r"\s*[;,|]\s*"
        r"|\s+\u2022\s+"
        r"|\s+·\s+",
        text,
    )

    keywords: List[str] = []

    for part in parts:

        part = normalize_whitespace(
            part
        )

        part = part.strip(
            " .:-"
        )

        if not part:
            continue

        if len(part) < 2:
            continue

        if len(part) > 100:
            continue

        if "http" in part.lower():
            continue

        if re.search(
            r"\b(acm|copyright|permission|"
            r"published|doi)\b",
            part,
            flags=re.IGNORECASE,
        ):
            continue

        keywords.append(
            part
        )

    return keywords


def extract_keywords(
    pages: List[Dict[str, Any]],
) -> List[str]:

    full_text = (
        extract_plain_text_from_pages(
            pages[:3]
        )
    )

    if not full_text:
        return []

    patterns = [
        r"(?im)^\s*keywords?\s*[:\-]\s*(.+)$",
        r"(?im)^\s*key words?\s*[:\-]\s*(.+)$",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            full_text,
        )

        for match in matches:

            keywords = _split_keywords(
                match
            )

            if keywords:
                return keywords[:30]

    match = re.search(
        r"(?is)\b(?:keywords?|key words?)\b\s*[:\-]?\s*(.*?)(?=\n\s*(?:\d+[\.)]?\s*)?introduction\b)",
        full_text,
    )

    if match:

        keywords = _split_keywords(
            match.group(1)
        )

        if keywords:
            return keywords[:30]

    return []


# ============================================================================
# SECTION DETECTION
# ============================================================================

def _remove_section_number(text: str) -> str:
    text = normalize_whitespace(text)
    return re.sub(
        r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[\s.)-]+",
        "",
        text,
    ).strip()


def _is_url_or_identifier(text: str) -> bool:
    lower = text.lower()
    return any(
        pattern in lower
        for pattern in (
            "http://", "https://", "www.", "doi.org",
            "arxiv.org", "arxiv:", "isbn", "issn",
        )
    )


def _is_metadata_line(text: str) -> bool:
    lower = normalize_whitespace(text).lower()

    if re.search(r"\barxiv:\s*\d+\.\d+", lower):
        return True

    if re.search(r"\b\d{4}\b.*\b(?:v\d+|cs\.[a-z]{2})\b", lower):
        return True

    metadata_words = (
        "received", "accepted", "published", "copyright",
        "permission to make digital", "acm reference format",
        "ccs concepts",
    )

    return any(word in lower for word in metadata_words)


def _is_figure_or_table_caption(text: str) -> bool:
    normalized = normalize_for_matching(text)
    return any(
        re.match(pattern, normalized)
        for pattern in (
            r"^figure\s*\d+",
            r"^fig\s*\d+",
            r"^table\s*\d+",
            r"^algorithm\s*\d+",
        )
    )


def _is_definition_or_theorem(text: str) -> bool:
    normalized = normalize_for_matching(text)
    return any(
        re.match(pattern, normalized)
        for pattern in (
            r"^definition\s*\d*",
            r"^theorem\s*\d*",
            r"^lemma\s*\d*",
            r"^proposition\s*\d*",
            r"^corollary\s*\d*",
            r"^proof\b",
            r"^remark\b",
            r"^example\s*\d*",
            r"^observation\s*\d*",
        )
    )


def _is_algorithm_fragment(text: str) -> bool:
    normalized = normalize_for_matching(text)
    return any(
        re.match(pattern, normalized)
        for pattern in (
            r"^procedure\b", r"^end procedure\b", r"^end if\b",
            r"^end for\b", r"^end while\b", r"^for each\b",
            r"^while\b", r"^if\b", r"^else\b", r"^repeat\b",
            r"^return\b", r"^input\b", r"^output\b",
            r"^step\s+\d+",
        )
    )


def _is_math_or_formula(text: str) -> bool:
    if not text:
        return False

    # Strong mathematical symbols.
    math_symbols = sum(
        text.count(symbol)
        for symbol in (
            "=", "≤", "≥", "∈", "∑", "∏", "√", "→", "←",
            "∞", "∼", "⟨", "⟩", "∥", "∩", "∪", "×", "÷",
        )
    )

    if math_symbols >= 2:
        return True

    normalized = normalize_for_matching(text)

    if re.search(r"\b(?:argmin|argmax|sqrt|log|exp)\b", normalized):
        return True

    # Variable assignments / pseudocode / equations.
    if re.search(r"=|:=|<-|←|→", text):
        return True

    if re.search(r"\b[a-z]\d*\s*=\s*[a-z0-9]", text, flags=re.IGNORECASE):
        return True

    # Function-like mathematical expressions such as f(a,b), d(q,c), etc.
    if re.search(r"\b[a-zA-Z]{1,3}\s*\([^)]{0,30}\)", text):
        return True

    # Very short isolated mathematical identifiers.
    words = normalized.split()
    if len(words) <= 3 and re.fullmatch(r"[a-z](?:\s+[a-z](?:\s+[a-z])?)?", normalized):
        return True

    return False


def _is_bullet_or_list_item(text: str) -> bool:
    stripped = text.strip()

    if stripped.startswith(("•", "▪", "◦", "–", "—", "*", "·")):
        return True

    if re.match(r"^\(?[a-zA-Z]\)?[.)]\s+", stripped):
        return True

    if re.match(r"^\(?\d+\)?[.)]\s+", stripped):
        return True

    return False


def _is_reference_fragment(text: str) -> bool:
    text = normalize_whitespace(text)

    if not text:
        return False

    if re.match(r"^\[\d+\]", text):
        return True

    # PDF extraction can leave a citation marker at the beginning of a body
    # line, e.g. "V [133], each bucket...". This is not a heading.
    if re.match(r"^[A-Za-z]\s*\[\d+(?:\s*,\s*\d+)*\]", text):
        return True

    if re.match(r"^\(\d+\)", text):
        return True

    if re.match(r"^\d+\.\s+[A-Z][A-Za-z-]+,", text):
        return True

    if re.search(
        r"\b(?:proceedings of|IEEE|ACM SIG|journal|transactions on)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def _looks_like_person_name(text: str) -> bool:
    text = normalize_whitespace(text)

    if not text or _is_affiliation_text(text):
        return False

    # Multiple authors separated by middle dots are also not headings.
    if "·" in text or "•" in text:
        pieces = [p.strip() for p in re.split(r"[·•]", text) if p.strip()]
        if len(pieces) >= 2 and all(_looks_like_single_person_name(p) for p in pieces):
            return True

    return _looks_like_single_person_name(text)


def _looks_like_single_person_name(text: str) -> bool:
    text = normalize_whitespace(text)

    if not text or _is_affiliation_text(text):
        return False

    words = text.split()
    if not 2 <= len(words) <= 6:
        return False

    if any(not re.match(r"^[A-Z][A-Za-z'`-]*$", word.strip(".,;:")) for word in words):
        return False

    return True


def _is_affiliation_text(text: str) -> bool:
    normalized = normalize_for_matching(text)

    if not normalized:
        return False

    words = set(normalized.split())

    if words.intersection(AFFILIATION_WORDS):
        return True

    if any(location in normalized for location in LOCATION_WORDS):
        return True

    if "@" in text:
        return True

    return False


def _is_sentence_like_heading(text: str) -> bool:
    text = normalize_whitespace(text)
    words = text.split()

    if len(words) > 12:
        return True

    # A final full stop usually means prose. A heading followed by body
    # text is handled separately by _split_inline_heading().
    if re.search(r"[!?]$", text):
        return True

    normalized = normalize_for_matching(_remove_section_number(text))
    if not normalized:
        return True

    first_words = normalized.split()
    if first_words and first_words[0] in NON_SECTION_START_WORDS:
        return True

    prose_markers = {
        "the", "this", "these", "those", "which", "that", "where",
        "because", "therefore", "however", "while", "using", "used",
        "have", "has", "are", "is", "was", "were", "we", "our",
    }

    marker_count = len(set(first_words) & prose_markers)

    if len(words) >= 8 and marker_count >= 2:
        return True

    return False


def _has_heading_typography(line: Dict[str, Any], body_font_size: float) -> bool:
    font_size = float(line.get("font_size", 0.0))
    max_font_size = float(line.get("max_font_size", font_size))
    bold = bool(line.get("bold", False))

    significantly_larger = max_font_size >= body_font_size * 1.15
    clearly_larger = max_font_size >= body_font_size * 1.30

    # Important: bold alone is NOT enough. Academic PDFs frequently use
    # bold body labels, figure labels, and inline terms.
    return clearly_larger or (significantly_larger and bold)


def _is_number_only_heading(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?",
            normalize_whitespace(text),
        )
    )


def _is_known_section_name(normalized: str) -> bool:
    return normalized in COMMON_SECTION_NAMES


def _starts_with_known_section(normalized: str) -> bool:
    for name in COMMON_SECTION_NAMES:
        if normalized.startswith(name + " "):
            return True
    return False


def _looks_like_section_heading(line: Dict[str, Any], body_font_size: float) -> bool:
    """
    Conservative heading detector.

    The key principle is that semantic names alone are NOT enough to make a
    line a heading. The line must also be structurally plausible. This keeps
    body fragments such as mathematical expressions and sentence continuations
    out of the section list.
    """

    text = normalize_whitespace(line.get("text", ""))

    if not text or len(text) < 2 or len(text) > 120:
        return False

    normalized = normalize_for_matching(_remove_section_number(text))

    if not normalized:
        return False

    # ---------------------------------------------------------------
    # Absolute rejection rules.
    # ---------------------------------------------------------------
    if _is_number_only_heading(text):
        return False

    if _is_url_or_identifier(text):
        return False

    if _is_metadata_line(text):
        return False

    # ACM tutorial/program labels can be visually prominent but are not
    # research-paper sections. A standalone "TUTORIAL" is one such label.
    if normalize_for_matching(text) == "tutorial":
        return False

    if _is_figure_or_table_caption(text):
        return False

    if _is_definition_or_theorem(text):
        return False

    if _is_algorithm_fragment(text):
        return False

    if _is_bullet_or_list_item(text):
        return False

    if _is_reference_fragment(text):
        return False

    if _is_math_or_formula(text):
        return False

    # A line that becomes punctuation/formula residue after section-number
    # removal is almost certainly extracted from a figure/equation.
    residue = _remove_section_number(text).strip()
    if residue and residue[0] in "·.,;:)]}]":
        return False

    if _looks_like_person_name(text):
        return False

    if normalized in {"abstract", "keywords", "key words"}:
        return False

    if normalized.startswith("abstract ") or normalized.startswith("keywords "):
        return False

    # ---------------------------------------------------------------
    # Exact known heading.
    # ---------------------------------------------------------------
    # A semantic name by itself is NOT sufficient. Phrases such as
    # "query processing" and "storage and indexing" occur frequently in
    # ordinary body prose. Require actual heading structure here. Inline
    # forms such as "Query Processing. The query processor..." are handled
    # separately by _split_inline_heading().
    if _is_known_section_name(normalized):
        if len(normalized.split()) > 8:
            return False

        if _is_sentence_like_heading(text):
            return False

        numbered_here = bool(
            re.match(
                r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[\s.)-]+",
                text,
            )
        )

        uppercase_heading = (
            text == text.upper()
            and any(char.isalpha() for char in text)
        )

        # Bold short headings are common in two-column papers even when the
        # font size is identical to body text.
        strong_bold = bool(line.get("bold", False)) and len(normalized.split()) <= 5

        return bool(
            numbered_here
            or uppercase_heading
            or _has_heading_typography(line, body_font_size)
            or strong_bold
        )

    # ---------------------------------------------------------------
    # Numbered heading.
    # ---------------------------------------------------------------
    numbered = bool(
        re.match(
            r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[\s.)-]+",
            text,
        )
    )

    typography = _has_heading_typography(line, body_font_size)
    word_count = len(normalized.split())

    if numbered:
        if word_count > 10:
            return False
        if _is_sentence_like_heading(text):
            return False

        # Footnotes and inline numbered prose can also begin with a number.
        # Real section numbers normally use approximately body-sized text.
        font_size = float(line.get("font_size", 0.0))
        if body_font_size > 0 and font_size < body_font_size * 0.88:
            return False

        # "4 bits ..." is ordinary prose, not a numbered heading. A bare
        # number followed by whitespace is accepted only when the heading
        # itself starts like a title (e.g. "1 Introduction"). Numbered forms
        # with punctuation ("2.1 Query Processing") remain fully supported.
        number_match = re.match(r"^(\d+)(\s+)(.+)$", text)
        if number_match:
            remainder = number_match.group(3).strip()
            if remainder and remainder[0].islower():
                return False

        residue = _remove_section_number(text).strip()
        if residue and residue[0] in "·.,;:)]}]":
            return False

        return True

    # ---------------------------------------------------------------
    # Generic unnumbered heading.
    # ---------------------------------------------------------------
    # Bold, short lines are legitimate subsection headings in many academic
    # PDFs even when their font size is identical to body text. Accept that
    # pattern only when the wording is short and not sentence-like.
    if not typography and not (bool(line.get("bold", False)) and word_count <= 5):
        return False

    if word_count > 8:
        return False

    if _is_sentence_like_heading(text):
        return False

    first_word = normalized.split()[0]
    if first_word in NON_SECTION_START_WORDS:
        return False

    # Lowercase headings do occur in real papers (for example
    # "offline blocking" and "single plan"), but they must be short and
    # typographically explicit.
    if text[0].islower():
        if not bool(line.get("bold", False)):
            return False
        if word_count > 3:
            return False
        if any(symbol in text for symbol in ("(", ")", "[", "]", "{", "}")):
            return False

    # A heading should not end in a closing bracket left over from a split
    # equation, title fragment, or figure/table expression.
    if text.rstrip().endswith((")", "]", "}")):
        return False

    return True


def _estimate_body_font_size_from_lines(lines: List[Dict[str, Any]]) -> float:
    sizes: List[float] = []

    for line in lines:
        text = normalize_whitespace(line.get("text", ""))
        if len(text) < 30:
            continue

        size = float(line.get("font_size", 0.0))
        if size > 0:
            sizes.append(round(size, 1))

    if not sizes:
        return 10.0

    return float(Counter(sizes).most_common(1)[0][0])


def _find_reference_section_index(lines: List[Dict[str, Any]]) -> Optional[int]:
    body_font_size = _estimate_body_font_size_from_lines(lines)

    for index, line in enumerate(lines):
        text = normalize_whitespace(line["text"])
        normalized = normalize_for_matching(_remove_section_number(text))

        if normalized not in REFERENCE_SECTION_NAMES:
            continue

        if _has_heading_typography(line, body_font_size):
            return index

        # References are normally near the end of a paper. Do not require
        # typography if they occur on a later page.
        if line.get("page_number", 1) >= 2:
            return index

    return None


def _find_main_content_start(lines: List[Dict[str, Any]]) -> int:
    """
    Find the first real paper section, normally Introduction.

    This prevents the title, author names, affiliations, ACM metadata, and
    first-page material from being interpreted as sections.
    """

    # First locate the abstract.
    abstract_index: Optional[int] = None

    for index, line in enumerate(lines):
        normalized = normalize_for_matching(line["text"])
        if normalized == "abstract" or normalized.startswith("abstract "):
            abstract_index = index
            break

    search_start = abstract_index + 1 if abstract_index is not None else 0

    # Prefer an explicit Introduction heading.
    for index in range(search_start, len(lines)):
        line = lines[index]
        normalized = normalize_for_matching(_remove_section_number(line["text"]))

        if normalized == "introduction":
            return index

    # If there is no Introduction, find the first strong section heading after
    # the abstract/metadata region. Avoid the first-page author block.
    body_font_size = _estimate_body_font_size_from_lines(lines)

    for index in range(search_start, len(lines)):
        line = lines[index]

        if line.get("page_number", 1) == 1:
            continue

        if _looks_like_section_heading(line, body_font_size):
            return index

    return search_start


def _split_inline_heading(
    line: Dict[str, Any],
    body_font_size: float,
) -> Optional[Tuple[str, str]]:
    """
    Split lines such as:

        Query Processing. The query processor mainly deals with...
        Benefits of Decoupling. The overall benefit is...

    into a heading and body text.

    This is deliberately conservative. We only split when the prefix is a
    plausible heading and the remaining text clearly looks like prose.
    """

    text = normalize_whitespace(line.get("text", ""))
    if not text or len(text) < 10:
        return None

    # ---------------------------------------------------------------
    # First use known semantic headings. This handles the most reliable
    # cases without trying to guess arbitrary sentence boundaries.
    # ---------------------------------------------------------------
    known_names = sorted(COMMON_SECTION_NAMES, key=len, reverse=True)

    for name in known_names:
        pattern = re.compile(
            r"^" + re.escape(name) + r"\s*[:.]\s+(.+)$",
            flags=re.IGNORECASE,
        )

        match = pattern.match(text)
        if not match:
            continue

        remainder = normalize_whitespace(match.group(1))
        if len(remainder.split()) < 3:
            continue

        # Known heading names are only reliable when the PDF gives us some
        # structural evidence as well. This prevents ordinary prose such as
        # "query optimization. These systems..." from becoming a section.
        structural_signal = (
            bool(line.get("bold", False))
            or _has_heading_typography(line, body_font_size)
        )

        if not structural_signal:
            continue

        # Preserve the capitalization used by the source PDF rather than the
        # lowercase normalization used in COMMON_SECTION_NAMES.
        heading_text = text[:match.start(1)].rstrip(" .:-")
        return normalize_heading(heading_text), remainder

    # ---------------------------------------------------------------
    # Numbered heading followed by inline prose.
    # Example:
    #   2. Query Processing. The processor...
    # ---------------------------------------------------------------
    match = re.match(
        r"^(?P<number>(?:\d+(?:\.\d+)*|[IVXLC]+)[\s.)-]+)"
        r"(?P<heading>[A-Z][A-Za-z0-9&/()\- ]{1,70}?)"
        r"\s*[:.]\s+"
        r"(?P<body>[A-Z].+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        heading = normalize_heading(match.group("heading"))
        body = normalize_whitespace(match.group("body"))

        if (
            heading
            and 1 <= len(heading.split()) <= 10
            and len(body.split()) >= 3
            and not _is_sentence_like_heading(heading)
        ):
            return heading, body

    # ---------------------------------------------------------------
    # Generic typography-supported inline heading.
    # This supports headings not present in COMMON_SECTION_NAMES without
    # making every sentence ending in a period a heading.
    # ---------------------------------------------------------------
    # Some academic two-column PDFs render subsection headings at exactly
    # the body font size, but use bold to distinguish them. This is a strong
    # signal when the prefix is short and title-like.
    strong_inline_style = (
        bool(line.get("bold", False))
        and len(text.split()) <= 18
    )

    if not _has_heading_typography(line, body_font_size) and not strong_inline_style:
        return None

    generic = re.match(
        r"^(?P<heading>[A-Z][A-Za-z0-9&/()\- ]{1,70}?)"
        r"\s*[:.]\s+"
        r"(?P<body>[A-Z].+)$",
        text,
    )

    if not generic:
        return None

    heading = normalize_heading(generic.group("heading"))
    body = normalize_whitespace(generic.group("body"))

    if not heading or not body:
        return None

    if not 1 <= len(heading.split()) <= 8:
        return None

    if len(body.split()) < 4:
        return None

    if "," in heading or ";" in heading:
        return None

    if _is_sentence_like_heading(heading):
        return None

    if _is_figure_or_table_caption(heading):
        return None

    return heading, body


def extract_sections(pages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract sections conservatively while preserving the stable metadata
    extraction logic from the working parser.

    Important design choices:
      1. Start at the real paper body, not page 1.
      2. Keep References as a hard boundary.
      3. Detect inline headings separately from normal headings.
      4. Never classify mathematical/code fragments as headings.
      5. Require typography for unknown headings.
      6. Preserve the body text that appears on the same line as a heading.
    """

    if not pages:
        return {}

    lines = get_all_page_lines(pages)
    if not lines:
        return {}

    body_font_size = _estimate_body_font_size_from_lines(lines)

    reference_index = _find_reference_section_index(lines)
    content_end = reference_index if reference_index is not None else len(lines)

    # Crucial fix: don't scan title/author/affiliation metadata as sections.
    main_start = _find_main_content_start(lines[:content_end])
    content_lines = lines[main_start:content_end]

    if not content_lines:
        return {}

    # Each tuple is:
    #   (line index, heading, optional body already present on heading line)
    section_positions: List[Tuple[int, str, str]] = []

    for index, line in enumerate(content_lines):
        text = normalize_whitespace(line.get("text", ""))
        if not text:
            continue

        # Inline heading must be checked before ordinary heading detection.
        inline = _split_inline_heading(line, body_font_size)
        if inline:
            heading, inline_body = inline
            normalized_heading = normalize_for_matching(heading)

            if normalized_heading not in {
                "abstract",
                "keywords",
                "key words",
                *REFERENCE_SECTION_NAMES,
            }:
                section_positions.append(
                    (index, heading, inline_body)
                )
            continue

        if not _looks_like_section_heading(line, body_font_size):
            continue

        heading = normalize_heading(text)
        normalized_heading = normalize_for_matching(heading)

        if normalized_heading in REFERENCE_SECTION_NAMES:
            continue

        if normalized_heading in {
            "abstract",
            "keywords",
            "key words",
        }:
            continue

        section_positions.append(
            (index, heading, "")
        )

    if not section_positions:
        return {}

    sections: Dict[str, str] = {}

    for position, (start_index, heading, inline_body) in enumerate(section_positions):
        if position + 1 < len(section_positions):
            end_index = section_positions[position + 1][0]
        else:
            end_index = len(content_lines)

        content: List[str] = []

        # Preserve body text that shared the heading's line.
        if inline_body:
            content.append(inline_body)

        for line in content_lines[start_index + 1:end_index]:
            text = normalize_whitespace(line.get("text", ""))
            if text:
                content.append(text)

        section_content = clean_text("\n".join(content))

        if not section_content:
            continue

        if heading in sections:
            sections[heading] += "\n\n" + section_content
        else:
            sections[heading] = section_content

    return sections


# ============================================================================
# REFERENCES
# ============================================================================

def _looks_like_reference_line(
    text: str,
) -> bool:

    text = normalize_whitespace(
        text
    )

    if not text:
        return False

    patterns = [
        r"^\[\d+\]",
        r"^\(\d+\)",
        r"^\d+\.",
        r"^\d+\)",
    ]

    return any(
        re.match(
            pattern,
            text,
        )
        for pattern in patterns
    )


def _find_reference_heading_in_text(
    full_text: str,
) -> Optional[re.Match]:

    patterns = [
        r"(?im)^\s*(?:references|bibliography)\s*$",
        r"(?im)^\s*(?:\d+[\.)]?\s*)?(?:references|bibliography)\s*$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            full_text,
        )

        if match:
            return match

    return None


def extract_references(
    pages: List[Dict[str, Any]],
) -> List[str]:

    full_text = (
        extract_plain_text_from_pages(
            pages
        )
    )

    if not full_text:
        return []

    reference_match = (
        _find_reference_heading_in_text(
            full_text
        )
    )

    if not reference_match:
        return []

    reference_text = full_text[
        reference_match.end():
    ]

    reference_text = clean_text(
        reference_text
    )

    if not reference_text:
        return []

    lines = reference_text.splitlines()

    references: List[str] = []

    current_reference: List[
        str
    ] = []

    for line in lines:

        line = normalize_whitespace(
            line
        )

        if not line:
            continue

        if re.match(
            r"^(?:appendix|appendices)\b",
            line,
            flags=re.IGNORECASE,
        ):
            break

        if _looks_like_reference_line(
            line
        ):

            if current_reference:

                references.append(
                    normalize_whitespace(
                        " ".join(
                            current_reference
                        )
                    )
                )

            current_reference = [
                line
            ]

        else:

            if current_reference:
                current_reference.append(
                    line
                )

    if current_reference:

        references.append(
            normalize_whitespace(
                " ".join(
                    current_reference
                )
            )
        )

    if len(references) < 2:

        paragraphs = [
            normalize_whitespace(
                paragraph
            )
            for paragraph in re.split(
                r"\n{2,}",
                reference_text,
            )
            if normalize_whitespace(
                paragraph
            )
        ]

        if len(paragraphs) > len(
            references
        ):
            references = paragraphs

    cleaned: List[str] = []

    seen: Set[str] = set()

    for reference in references:

        reference = normalize_whitespace(
            reference
        )

        if len(reference) < 10:
            continue

        key = normalize_for_matching(
            reference
        )

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(
            reference
        )

    return cleaned


# ============================================================================
# FULL TEXT
# ============================================================================

def extract_full_text(
    pages: List[Dict[str, Any]],
) -> str:

    return extract_plain_text_from_pages(
        pages
    )


# ============================================================================
# MAIN PARSER
# ============================================================================

def parse_pdf(
    pdf_path: Path,
) -> Paper:

    pdf_path = Path(
        pdf_path
    )

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():

        raise ValueError(
            f"Expected a file but received: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            f"Expected a PDF file: {pdf_path}"
        )

    # ------------------------------------------------------------------
    # 1. Extract PDF layout.
    # ------------------------------------------------------------------

    pages = extract_pdf_pages(
        pdf_path
    )

    if not pages:

        raise ValueError(
            f"Could not extract any pages from PDF: "
            f"{pdf_path.name}"
        )

    # ------------------------------------------------------------------
    # 2. Remove repeated headers and footers.
    # ------------------------------------------------------------------

    pages = remove_repeated_headers_footers(
        pages
    )

    # ------------------------------------------------------------------
    # 3. Metadata.
    # ------------------------------------------------------------------

    title = extract_title(
        pages
    )

    authors = extract_authors(
        pages,
        title,
    )

    abstract = extract_abstract(
        pages
    )

    keywords = extract_keywords(
        pages
    )

    # ------------------------------------------------------------------
    # 4. Structured content.
    # ------------------------------------------------------------------

    sections = extract_sections(
        pages
    )

    references = extract_references(
        pages
    )

    full_text = extract_full_text(
        pages
    )

    # ------------------------------------------------------------------
    # 5. Paper model.
    # ------------------------------------------------------------------

    return Paper(
        paper_id=str(
            uuid.uuid4()
        ),
        filename=pdf_path.name,
        title=title,
        authors=authors,
        abstract=abstract,
        keywords=keywords,
        sections=sections,
        references=references,
        full_text=full_text,
    )