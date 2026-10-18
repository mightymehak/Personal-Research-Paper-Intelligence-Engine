
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks an existing list of retrieved chunks using a cross-encoder.

    The original retrieval score is preserved in `score`.
    The cross-encoder score is stored separately in `rerank_score`.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ) -> None:
        print(f"Loading reranker model: {model_name}")

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

        print("Cross-encoder reranker loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Score candidates against the query and return the best top_k.

        Each candidate must contain a non-empty `text` field.
        Existing candidate metadata is preserved.
        """

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not candidates:
            return []

        valid_candidates = []

        for candidate in candidates:
            text = candidate.get("text", "")

            if isinstance(text, str) and text.strip():
                valid_candidates.append(candidate)

        if not valid_candidates:
            return []

        pairs = [
            (query, candidate["text"])
            for candidate in valid_candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        ranked_candidates = []

        for candidate, score in zip(valid_candidates, scores):
            result = dict(candidate)

            # Preserve the original FAISS ranking.
            result["original_rank"] = candidate.get("rank")

            # Preserve the original vector similarity score.
            # Do not compare this directly with rerank_score.
            result["rerank_score"] = float(score)

            ranked_candidates.append(result)

        ranked_candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        # Assign new ranks while preserving original_rank.
        for new_rank, result in enumerate(ranked_candidates, start=1):
            result["rank"] = new_rank

        return ranked_candidates[:top_k]
