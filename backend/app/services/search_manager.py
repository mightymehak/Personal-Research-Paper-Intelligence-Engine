
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.faiss_service import FaissVectorStore


PROCESSED_DIRECTORY = Path("data/processed")


class SearchManager:
    def __init__(self):
        self.embedding_service: Optional[EmbeddingService] = None
        self.stores: Dict[str, FaissVectorStore] = {}

    def load_indexes(self) -> None:
        """Load every valid saved library index at application startup."""
        if not PROCESSED_DIRECTORY.exists():
            print("No processed data directory found.")
            return

        index_directories = sorted(
            path
            for path in PROCESSED_DIRECTORY.glob("*/faiss")
            if (path / "faiss.index").is_file()
            and (path / "chunks.json").is_file()
        )

        if not index_directories:
            print("No saved FAISS indexes found.")
            return

        # Load the embedding model only once.
        self.embedding_service = EmbeddingService()

        for index_directory in index_directories:
            library_id = index_directory.parent.name

            try:
                store = FaissVectorStore(self.embedding_service)
                store.load(index_directory)
                self.stores[library_id] = store

                print(
                    f"Search enabled for {library_id}: "
                    f"{store.index.ntotal} chunks"
                )
            except Exception as exc:
                print(
                    f"Could not load index for {library_id}: {exc}"
                )

    def search(
        self,
        query: str,
        top_k: int = 5,
        library_id: Optional[str] = None,
    ) -> list:
        if not self.stores:
            raise RuntimeError("No searchable libraries are loaded.")

        if library_id is None:
            if len(self.stores) > 1:
                raise ValueError(
                    "Multiple libraries are available. "
                    "Specify library_id."
                )
            library_id = next(iter(self.stores))

        if library_id not in self.stores:
            raise KeyError(f"Unknown or unavailable library: {library_id}")

        results = self.stores[library_id].search(query, top_k=top_k)

        return [
            {"library_id": library_id, **result}
            for result in results
        ]


search_manager = SearchManager()
