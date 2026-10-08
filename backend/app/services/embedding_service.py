from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.models.chunk import PaperChunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """
    Generates semantic embeddings for PaperMind chunks
    using a Sentence Transformer model.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model_name = model_name

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        self.max_seq_length = (
            self.model.max_seq_length
        )

        print(
            "Embedding dimension: "
            f"{self.embedding_dimension}"
        )

        print(
            "Maximum sequence length: "
            f"{self.max_seq_length} tokens"
        )

    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Convert a single text into a normalized embedding.
        """

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty text."
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype(
            np.float32
        )

    def encode_chunks(
        self,
        chunks: List[PaperChunk],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate normalized embeddings for multiple chunks.

        Returns:
            numpy array with shape:
            (number_of_chunks, embedding_dimension)
        """

        if not chunks:
            return np.empty(
                (
                    0,
                    self.embedding_dimension,
                ),
                dtype=np.float32,
            )

        # --------------------------------------------------
        # Validate that chunks fit inside model limit.
        # --------------------------------------------------

        oversized_chunks = [
            chunk
            for chunk in chunks
            if chunk.token_count
            > self.max_seq_length
        ]

        if oversized_chunks:

            details = "\n".join(
                (
                    f"  - {chunk.chunk_id}: "
                    f"{chunk.token_count} tokens"
                )
                for chunk in oversized_chunks[:10]
            )

            raise ValueError(
                "Cannot generate embeddings because "
                "some chunks exceed the model's "
                f"maximum sequence length "
                f"({self.max_seq_length} tokens).\n"
                f"{details}"
            )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype(
            np.float32
        )

        # --------------------------------------------------
        # Validate embedding shape.
        # --------------------------------------------------

        expected_shape = (
            len(chunks),
            self.embedding_dimension,
        )

        if embeddings.shape != expected_shape:

            raise ValueError(
                "Unexpected embedding shape. "
                f"Expected {expected_shape}, "
                f"got {embeddings.shape}."
            )

        # --------------------------------------------------
        # Validate normalization.
        # --------------------------------------------------

        norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                "Embedding normalization failed."
            )

        return embeddings