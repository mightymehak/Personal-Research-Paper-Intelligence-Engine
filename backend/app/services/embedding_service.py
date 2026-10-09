
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.models.chunk import PaperChunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )
        self.max_seq_length = self.model.max_seq_length

        print(f"Embedding dimension: {self.embedding_dimension}")
        print(f"Maximum sequence length: {self.max_seq_length}")

    def encode_text(self, text: str) -> np.ndarray:
        """Generate one normalized embedding."""
        if not text or not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty text."
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (self.embedding_dimension,):
            raise ValueError(
                f"Unexpected embedding shape: {embedding.shape}"
            )

        if not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4):
            raise ValueError("Embedding is not L2-normalized.")

        return embedding

    def encode_chunks(
        self,
        chunks: List[PaperChunk],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Generate normalized embeddings for paper chunks."""
        if not chunks:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        # Validate chunk lengths before encoding.
        oversized = [
            {
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
            if chunk.token_count > self.max_seq_length
        ]

        if oversized:
            raise ValueError(
                f"{len(oversized)} chunks exceed the model's "
                f"{self.max_seq_length}-token limit. "
                f"Examples: {oversized[:10]}"
            )

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (
            len(chunks),
            self.embedding_dimension,
        )

        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected embedding shape {expected_shape}, "
                f"got {embeddings.shape}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )

        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(
                "Some embeddings are not L2-normalized."
            )

        print(
            f"Successfully embedded {len(chunks)} chunks "
            f"into {self.embedding_dimension}-dimensional vectors."
        )

        return embeddings
