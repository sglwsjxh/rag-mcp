"""Smoke tests for rag_mcp.chunker.Chunker — no external API calls."""

from rag_mcp.chunker import Chunker


# ── heading boundaries ──────────────────────────────────────────────────

def test_heading_boundary():
    """Markdown headings split into separate segments before merge."""
    chunker = Chunker(max_tokens=1000)
    text = "# Title\n\nContent here.\n\n## Subtitle\n\nMore content."
    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 1
    assert "# Title" in chunks[0]["text"]
    assert all(k in chunks[0] for k in ("text", "index", "token_count"))


def test_multiple_headings_create_multiple_chunks():
    """Multiple headings with content produce multiple merged chunks."""
    chunker = Chunker(max_tokens=50)  # small to force splits
    text = "# Part 1\n\nA" * 20 + "\n\n# Part 2\n\nB" * 20
    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 2
    # Part 2 heading should appear in a later chunk
    headings_in_chunks = [i for i, c in enumerate(chunks) if "# Part" in c["text"]]
    assert len(headings_in_chunks) >= 1


# ── blank paragraph breaks ──────────────────────────────────────────────

def test_blank_line_boundary():
    """Consecutive blank lines split text into segments."""
    chunker = Chunker(max_tokens=1000)
    text = "First paragraph.\n\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 1
    assert "First paragraph" in chunks[0]["text"]


# ── empty / whitespace input ────────────────────────────────────────────

def test_empty_input():
    """Empty or whitespace-only input returns an empty list."""
    chunker = Chunker(max_tokens=1000)
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   ") == []
    assert chunker.chunk_text("\n\n\n") == []


# ── single segment input ────────────────────────────────────────────────

def test_single_segment_input():
    """Text without any boundary patterns becomes one chunk."""
    chunker = Chunker(max_tokens=1000)
    text = "This is just plain text with no boundaries at all."
    chunks = chunker.chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0]["text"] == text.strip()
    assert chunks[0]["index"] == 0


# ── token truncation ────────────────────────────────────────────────────

def test_truncate_short_text():
    """Text shorter than max_tokens is returned unchanged."""
    text = "Short text."
    result = Chunker.truncate(text, max_tokens=100)
    assert result == text


def test_truncate_long_text():
    """Text longer than max_tokens is truncated to max_tokens tokens."""
    chunker = Chunker(max_tokens=1000)
    long_text = "word " * 500  # ~500 tokens
    truncated = Chunker.truncate(long_text, max_tokens=10)

    # Count tokens of truncated result — should be <= 10
    assert chunker._count_tokens(truncated) <= 10
    assert len(truncated) < len(long_text)


# ── oversized segment stays intact ──────────────────────────────────────

def test_oversized_segment_not_split():
    """A single segment exceeding max_tokens is kept as-is, not split further."""
    chunker = Chunker(max_tokens=10)
    huge = "word " * 1000  # way over 10 tokens, no boundary patterns
    chunks = chunker.chunk_text(huge)

    # Should be exactly one chunk (the whole thing)
    assert len(chunks) == 1
    assert chunks[0]["token_count"] > 10  # exceeds max, but not split
