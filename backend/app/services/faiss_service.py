
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from backend.app.models.chunk import PaperChunk
from backend.app.services.embedding_service import EmbeddingService


class FaissVectorStore:
    """Build and search a FAISS index of research-paper chunks."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.embedding_dimension = (
            embedding_service.embedding_dimension
        )
        self.index = None
        self.chunks: List[PaperChunk] = []

    def build_index(self, chunks: List[PaperChunk]) -> None:
        """Embed chunks and construct an exact inner-product index."""
        if not chunks:
            raise ValueError(
                "Cannot build a FAISS index without chunks."
            )

        embeddings = self.embedding_service.encode_chunks(chunks)

        expected_shape = (
            len(chunks),
            self.embedding_dimension,
        )

        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected {expected_shape}, got {embeddings.shape}."
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid values.")

        # IndexFlatIP performs exact inner-product search.
        index = faiss.IndexFlatIP(self.embedding_dimension)
        index.add(np.ascontiguousarray(embeddings))

        self.index = index
        self.chunks = list(chunks)

        print("FAISS index built successfully.")
        print(f"Indexed chunks: {self.index.ntotal}")
        print(f"Vector dimensions: {self.index.d}")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the most similar chunks for a text query."""
        if self.index is None:
            raise RuntimeError(
                "FAISS index is not built or loaded."
            )

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.embedding_service.encode_text(query)
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.embedding_dimension:
            raise ValueError("Query embedding dimension mismatch.")

        k = min(top_k, self.index.ntotal)
        scores, positions = self.index.search(
            np.ascontiguousarray(query_embedding),
            k,
        )

        results = []

        for score, position in zip(scores[0], positions[0]):
            # FAISS uses -1 for missing result positions.
            if position < 0:
                continue

            chunk = self.chunks[int(position)]

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "word_count": chunk.word_count,
                    "token_count": chunk.token_count,
                }
            )

        return results

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and corresponding chunk metadata."""
        if self.index is None:
            raise RuntimeError("Build or load an index before saving.")

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                "Index size and chunk metadata count do not match."
            )

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "faiss.index"
        metadata_path = directory / "chunks.json"

        faiss.write_index(self.index, str(index_path))

        metadata = {
            "model_name": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_dimension,
            "chunks": [
                chunk.dict() for chunk in self.chunks
            ],
        }

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        print(f"FAISS index saved to: {index_path}")
        print(f"Chunk metadata saved to: {metadata_path}")

    def load(self, directory: str | Path) -> None:
        """Load a previously saved index and its chunk metadata."""
        directory = Path(directory)
        index_path = directory / "faiss.index"
        metadata_path = directory / "chunks.json"

        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Expected faiss.index and chunks.json in {directory}"
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata["model_name"] != self.embedding_service.model_name:
            raise ValueError(
                "The saved index uses a different embedding model."
            )

        if metadata["embedding_dimension"] != self.embedding_dimension:
            raise ValueError(
                "Saved embedding dimension does not match the model."
            )

        loaded_index = faiss.read_index(str(index_path))
        loaded_chunks = [
            PaperChunk(**item) for item in metadata["chunks"]
        ]

        if loaded_index.d != self.embedding_dimension:
            raise ValueError("Loaded FAISS index has the wrong dimension.")

        if loaded_index.ntotal != len(loaded_chunks):
            raise ValueError(
                "Loaded index and chunk metadata counts do not match."
            )

        self.index = loaded_index
        self.chunks = loaded_chunks

        print(f"Loaded {self.index.ntotal} indexed chunks.")
