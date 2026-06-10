"""Knowledge base CRUD + ChromaDB retrieval core manager."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import chromadb

from .chunker import Chunker
from .config import Settings
from .embedder import Embedder
from .reranker import Reranker


class KnowledgeManager:
    """Manage multiple knowledge bases with CRUD + search.

    Each knowledge base is stored under ``database/{name}/`` with a
    ``chroma_db/`` subdirectory for the ChromaDB PersistentClient data and
    an ``assets/`` subdirectory for the original uploaded files.
    """

    def __init__(self) -> None:
        self.settings = Settings()
        self.embedder = Embedder()
        self.reranker = Reranker(self.settings)
        max_tokens = (self.settings.embedding_input_token or 8192) - 200
        self.chunker = Chunker(max_tokens=max(max_tokens, 256))

        self.base_dir = Path(self.settings.database_path)
        self._clients: dict[str, chromadb.PersistentClient] = {}

    @staticmethod
    def _name_to_slug(name: str) -> str:
        """Generate a ChromaDB-safe collection name from any string.

        ChromaDB requires names in ``[a-zA-Z0-9._-]``, 3-512 chars.
        Non-ASCII characters (e.g. Chinese) are replaced with ``_``.
        If the result is too short, fall back to an MD5 suffix.
        """
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name).strip("_")
        if len(safe) < 3:
            h = hashlib.md5(name.encode()).hexdigest()[:8]
            return f"kb_{h}"
        return safe.lower()[:60]

    def _index_path(self) -> Path:
        return self.base_dir / "index.json"

    def _index(self) -> list[dict[str, Any]]:
        """Load the index file (returns empty list if missing)."""
        path = self._index_path()
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, index: list[dict[str, Any]]) -> None:
        """Persist the index to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self._index_path(), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _get_client(self, name: str) -> chromadb.PersistentClient:
        """Get or create a PersistentClient for *name*."""
        if name not in self._clients:
            client_path = self.base_dir / name / "chroma_db"
            client_path.mkdir(parents=True, exist_ok=True)
            self._clients[name] = chromadb.PersistentClient(path=str(client_path))
        return self._clients[name]

    def _collection(self, name: str) -> chromadb.Collection:
        """Get the ChromaDB collection for *name*, creating it if needed.

        Uses the slug from index.json as the actual ChromaDB collection name.
        """
        client = self._get_client(name)
        index = self._index()
        entry = next((e for e in index if e["name"] == name), None)
        if entry is None:
            raise ValueError(f"Knowledge base '{name}' not found")
        slug = entry.get("slug", name)
        try:
            return client.get_collection(name=slug)
        except chromadb.errors.NotFoundError:
            return client.create_collection(name=slug)

    def list(self) -> list[dict[str, Any]]:
        """Return all registered knowledge bases from index.json."""
        return self._index()

    def add_database(self, name: str, description: str = "") -> None:
        """Create a new knowledge base named *name*.

        Args:
            name: Unique identifier for the knowledge base.
            description: Optional human-readable description.

        Raises:
            ValueError: If a knowledge base with the same name already exists.
        """
        index = self._index()
        if any(entry["name"] == name for entry in index):
            raise ValueError(f"Knowledge base '{name}' already exists")

        slug = self._name_to_slug(name)

        assets_dir = self.base_dir / name / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        client = self._get_client(name)
        client.create_collection(name=slug)

        # Append to index
        from datetime import date
        index.append({
            "name": name,
            "slug": slug,
            "desc": description,
            "created": date.today().isoformat(),
        })
        self._save_index(index)

    def delete_database(self, name: str) -> None:
        """Delete a knowledge base and all its data.

        Args:
            name: The knowledge base to delete.
        """
        # Remove from index
        index = self._index()
        index = [entry for entry in index if entry["name"] != name]
        self._save_index(index)

        # Clear client cache
        self._clients.pop(name, None)

        # Delete the entire database directory
        db_path = self.base_dir / name
        if db_path.exists():
            shutil.rmtree(db_path)

    def add_file(self, collection_name: str, file_path: str) -> int:
        """Add a file to a knowledge base, chunking and embedding it.

        Args:
            collection_name: Target knowledge base name.
            file_path: Path to the file to add.

        Returns:
            Number of chunks the file was split into.

        Raises:
            ValueError: If the collection doesn't exist.
        """
        index = self._index()
        if not any(entry["name"] == collection_name for entry in index):
            raise ValueError(f"Knowledge base '{collection_name}' not found")

        src = Path(file_path)
        dst_dir = self.base_dir / collection_name / "assets"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        # Copy file to assets
        shutil.copy2(src, dst)

        # Image branch: embed directly via multimodal API, no chunking
        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        if src.suffix.lower() in IMAGE_EXTENSIONS:
            embedding = self.embedder.embed(text=src.stem, images=[str(dst)])
            chunk_id = uuid.uuid4().hex[:12]
            collection = self._collection(collection_name)
            self._validate_collection_dimension(collection, collection_name)
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[src.name],
                metadatas=[{
                    "file_name": src.name,
                    "file_path": str(dst),
                    "chunk_index": 0,
                    "token_count": 0,
                    "type": "image",
                }],
            )
            return 1

        # Chunk the file
        chunks = self.chunker.chunk_file(str(dst))
        if not chunks:
            return 0

        # Embed all chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # Generate unique IDs
        ids = [uuid.uuid4().hex[:12] for _ in chunks]

        # Store in ChromaDB
        collection = self._collection(collection_name)
        self._validate_collection_dimension(collection, collection_name)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "file_name": src.name,
                    "file_path": str(dst),
                    "chunk_index": c["index"],
                    "token_count": c["token_count"],
                }
                for c in chunks
            ],
        )

        return len(chunks)

    def delete_file(self, collection_name: str, file_name: str) -> None:
        """Delete a file's chunks from a knowledge base and remove the asset.

        Args:
            collection_name: Target knowledge base name.
            file_name: Name of the file to delete.
        """
        collection = self._collection(collection_name)
        collection.delete(where={"file_name": file_name})

        safe_name = Path(file_name).name
        asset_path = self.base_dir / collection_name / "assets" / safe_name
        if asset_path.exists():
            asset_path.unlink()

    def _validate_collection_dimension(
        self, collection: chromadb.Collection, name: str
    ) -> None:
        """Check existing collection vectors match current embedding dimension."""
        try:
            existing = collection.get(limit=1, include=["embeddings"])
            embeddings = existing.get("embeddings") or []
            if not embeddings:
                return
            db_dim = len(embeddings[0])
        except (KeyError, IndexError, TypeError):
            return

        current_dim = self.embedder.detected_dimension
        if current_dim is not None and db_dim != current_dim:
            raise ValueError(
                f"Knowledge base '{name}' was created with {db_dim}D embeddings, "
                f"current model produces {current_dim}D. "
                f"Use the original model or create a new knowledge base."
            )

    def search(self, query: str, collection_name: str = "", top_k: int | None = None) -> list[dict[str, Any]]:
        """Search across one or all knowledge bases, reranked.

        Args:
            query: The search query.
            collection_name: If provided, search only this collection.
                Otherwise search all registered collections.
            top_k: Max results to return. Overrides env RERANK_TOP_K if set.

        Returns:
            Sorted list of matching documents with metadata.
        """
        index = self._index()
        if collection_name:
            targets = [e for e in index if e["name"] == collection_name]
            if not targets:
                raise ValueError(f"Knowledge base '{collection_name}' not found")
        else:
            targets = index

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Collect results from all target collections
        all_docs: list[dict[str, Any]] = []
        for entry in targets:
            name = entry["name"]
            try:
                collection = self._collection(name)
                query_kwargs = {
                    "query_embeddings": [query_embedding],
                    "include": ["documents", "metadatas", "distances"],
                }
                if self.embedder.top_k is not None:
                    query_kwargs["n_results"] = self.embedder.top_k
                results = collection.query(**query_kwargs)

                for i in range(len(results["documents"][0])):
                    all_docs.append({
                        "collection": name,
                        "text": results["documents"][0][i],
                        "file_name": results["metadatas"][0][i].get("file_name", ""),
                        "file_path": results["metadatas"][0][i].get("file_path", ""),
                        "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                        "_raw_score": results["distances"][0][i],
                    })
            except Exception as exc:
                # Let a single collection's failure propagate — don't silently
                # swallow errors.
                raise RuntimeError(f"Search failed in collection '{name}'") from exc

        # Rerank if we have documents
        if all_docs:
            return self.reranker.rerank(query, all_docs, top_k)

        return []
