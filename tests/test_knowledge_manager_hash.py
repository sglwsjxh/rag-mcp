"""Smoke tests for knowledge_manager hash deduplication + delete cleanup."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Patch Embedder / Reranker before importing KnowledgeManager
embedder_mock = MagicMock()
embedder_mock.embed.return_value = [0.1] * 768
embedder_mock.embed_batch.return_value = [[0.1] * 768]
embedder_mock.top_k = None
embedder_mock.detected_dimension = 768

reranker_mock = MagicMock()
reranker_mock.rerank.side_effect = lambda q, docs, top_k=None: docs

with patch("rag_mcp.knowledge_manager.Embedder", return_value=embedder_mock):
    with patch("rag_mcp.knowledge_manager.Reranker", return_value=reranker_mock):
        from rag_mcp.knowledge_manager import KnowledgeManager


@pytest.fixture()
def km(tmp_path: Path) -> KnowledgeManager:
    """Create a KnowledgeManager backed by a temp directory."""
    km = KnowledgeManager.__new__(KnowledgeManager)
    km.settings = MagicMock()
    km.settings.database_path = str(tmp_path / "database")
    km.settings.embedding_input_token = 8192
    km.settings.rerank_api_key = None
    km.embedder = embedder_mock
    km.reranker = reranker_mock
    km.chunker = MagicMock()
    km.chunker.chunk_file.return_value = [
        {"text": "hello world", "index": 0, "token_count": 3},
    ]
    km.base_dir = Path(km.settings.database_path)
    km._clients = {}
    km._validate_collection_dimension = MagicMock()  # skip chromadb dimension check

    # Mock the ChromaDB collection returned by _collection()
    mock_collection = MagicMock()
    mock_collection.add = MagicMock()
    mock_collection.delete = MagicMock()
    original_collection = km._collection

    def patched_collection(name):
        return mock_collection

    km._collection = patched_collection
    return km


def _write_file(base: Path, name: str, content: str) -> Path:
    p = base / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ── Hash dedup ─────────────────────────────────────────────────────────


def test_add_file_first_time_creates_hash(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "test kb")
    src = _write_file(tmp_path, "doc.txt", "first add")

    result = km.add_file("kb1", str(src))
    assert result == 1  # 1 chunk

    hash_file = km._hash_path("kb1")
    assert hash_file.exists()
    data = json.loads(hash_file.read_text(encoding="utf-8"))
    assert "doc.txt" in data["file_hashes"]


def test_add_duplicate_content_returns_zero(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "test kb")
    src1 = _write_file(tmp_path, "doc.txt", "same content")
    km.add_file("kb1", str(src1))

    # Add a different filename with the same content
    src2 = _write_file(tmp_path, "doc_copy.txt", "same content")
    result = km.add_file("kb1", str(src2))
    assert result == 0  # duplicate content, skipped

    # Asset should NOT have been copied
    asset = km.base_dir / "kb1" / "assets" / "doc_copy.txt"
    assert not asset.exists()


def test_same_filename_same_content_skips(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "test kb")
    src = _write_file(tmp_path, "doc.txt", "same content")

    km.add_file("kb1", str(src))
    result = km.add_file("kb1", str(src))
    assert result == 0  # re-adding same file


def test_different_content_same_name_overwrites_hash(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "test kb")
    src1 = _write_file(tmp_path, "doc.txt", "original content")
    km.add_file("kb1", str(src1))

    src2 = _write_file(tmp_path, "doc.txt", "new content different")
    result = km.add_file("kb1", str(src2))
    # Different content = new hash = accepted
    assert result == 1


def test_cross_kb_same_content_allowed(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "kb1")
    km.add_database("kb2", "kb2")

    src = _write_file(tmp_path, "doc.txt", "shared content")

    r1 = km.add_file("kb1", str(src))
    r2 = km.add_file("kb2", str(src))

    assert r1 == 1
    assert r2 == 1  # different KBs, same content is fine


# ── Delete cleanup ─────────────────────────────────────────────────────


def test_delete_file_removes_hash_entry(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "test kb")
    src = _write_file(tmp_path, "doc.txt", "to delete")
    km.add_file("kb1", str(src))

    # Verify hash exists
    hashes = km._load_hashes("kb1")
    assert "doc.txt" in hashes

    km.delete_file("kb1", "doc.txt")

    # Hash entry should be gone
    hashes_after = km._load_hashes("kb1")
    assert "doc.txt" not in hashes_after


def test_readd_after_delete_works(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "test kb")
    src = _write_file(tmp_path, "doc.txt", "content")

    r1 = km.add_file("kb1", str(src))
    assert r1 == 1

    km.delete_file("kb1", "doc.txt")

    # Re-adding should work (hash was cleaned up)
    r2 = km.add_file("kb1", str(src))
    assert r2 == 1


def test_delete_nonexistent_file_does_not_crash(km: KnowledgeManager, tmp_path: Path):
    km.add_database("kb1", "test kb")
    # Should not raise even if file was never added
    km.delete_file("kb1", "ghost.txt")
