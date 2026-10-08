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

        print(
            "Embedding dimension: "
            f"{self.embedding_dimension}"
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
                (0, self.embedding_dimension),
                dtype=np.float32,
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

        return embeddings.astype(
            np.float32
        )