"""Smoke tests for grouped search (multi-collection) behavior."""

from unittest.mock import MagicMock, patch

import pytest

from rag_mcp.knowledge_manager import KnowledgeManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def km():
    """KnowledgeManager with all external deps mocked."""
    with (
        patch("rag_mcp.knowledge_manager.KnowledgeManager._index") as mock_index,
        patch("rag_mcp.knowledge_manager.Embedder") as mock_embedder_cls,
        patch("rag_mcp.knowledge_manager.Reranker") as mock_reranker_cls,
    ):
        # --- index ---
        mock_index.return_value = [
            {"name": "kb_a", "slug": "kb_a"},
            {"name": "kb_b", "slug": "kb_b"},
        ]

        # --- embedder ---
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        mock_embedder.top_k = None
        mock_embedder.detected_dimension = None
        mock_embedder_cls.return_value = mock_embedder

        # --- reranker ---
        mock_reranker = MagicMock()
        mock_reranker_cls.return_value = mock_reranker

        km = KnowledgeManager()

        # --- collection query mock (shared for all KBs) ---
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["alpha", "beta"]],
            "metadatas": [[
                {"file_name": "a.md", "file_path": "/a.md", "chunk_index": 0},
                {"file_name": "b.md", "file_path": "/b.md", "chunk_index": 1},
            ]],
            "distances": [[0.1, 0.2]],
        }
        km._collection = MagicMock(return_value=mock_col)

        yield km


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_search_full_returns_grouped_dict(km):
    """Full search (collection_name='') returns a dict keyed by collection."""
    controlled = [
        {"collection": "kb_a", "text": "d1", "score": 0.9,
         "file_name": "a.md", "file_path": "/a.md", "chunk_index": 0},
        {"collection": "kb_b", "text": "d2", "score": 0.8,
         "file_name": "b.md", "file_path": "/b.md", "chunk_index": 1},
        {"collection": "kb_a", "text": "d3", "score": 0.7,
         "file_name": "c.md", "file_path": "/c.md", "chunk_index": 2},
    ]
    km.reranker.rerank.return_value = controlled

    result = km.search("test query")

    assert isinstance(result, dict), "Full search should return a dict"
    assert set(result.keys()) == {"kb_a", "kb_b"}
    assert len(result["kb_a"]) == 2
    assert len(result["kb_b"]) == 1
    # Order within groups preserved
    assert result["kb_a"][0]["text"] == "d1"
    assert result["kb_a"][1]["text"] == "d3"
    assert result["kb_b"][0]["text"] == "d2"


def test_search_single_returns_flat_list(km):
    """Single-collection search returns a flat list."""
    km.reranker.rerank.return_value = [
        {"collection": "kb_a", "text": "d1", "score": 0.9,
         "file_name": "a.md", "file_path": "/a.md", "chunk_index": 0},
    ]

    result = km.search("test query", collection_name="kb_a")

    assert isinstance(result, list), "Single search should return a list"
    assert len(result) == 1
    assert result[0]["text"] == "d1"


def test_search_full_empty_returns_empty_dict(km):
    """Empty full search returns {}."""
    # Make collection query return nothing → no all_docs → empty path
    km._collection.return_value.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    result = km.search("test query")

    assert isinstance(result, dict)
    assert result == {}


def test_search_single_empty_returns_empty_list(km):
    """Empty single-collection search returns [] (backward compatible)."""
    km._collection.return_value.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    result = km.search("test query", collection_name="kb_a")

    assert isinstance(result, list)
    assert result == []


def test_group_order_preserves_rerank_sequence(km):
    """Group keys appear in order of first occurrence in reranked results."""
    controlled = [
        {"collection": "kb_b", "text": "first", "score": 0.9,
         "file_name": "x.md", "file_path": "/x.md", "chunk_index": 0},
        {"collection": "kb_a", "text": "second", "score": 0.8,
         "file_name": "y.md", "file_path": "/y.md", "chunk_index": 1},
        {"collection": "kb_b", "text": "third", "score": 0.7,
         "file_name": "z.md", "file_path": "/z.md", "chunk_index": 2},
    ]
    km.reranker.rerank.return_value = controlled

    result = km.search("test query")

    # kb_b appears first in reranked order, so it should be first key
    keys = list(result.keys())
    assert keys == ["kb_b", "kb_a"], (
        f"Expected [kb_b, kb_a], got {keys}"
    )
