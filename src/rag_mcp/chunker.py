"""Document chunking by natural boundaries with token-aware merging."""

from __future__ import annotations

import re


class Chunker:
    """Split text into token-bounded chunks using natural boundary detection.

    The chunker first splits text by natural boundaries (markdown headings,
    Chinese chapter markers, numbered lists, empty lines) then merges small
    segments together so each chunk approaches *max_tokens* without exceeding
    it.  Single segments that already exceed *max_tokens* are kept intact —
    the chunker never splits a boundary-anchored segment further.
    """

    def __init__(self, max_tokens: int = 131072) -> None:
        """Initialize the chunker.

        Args:
            max_tokens: Maximum token count per chunk.  Default is 131072.
        """
        self.max_tokens = max_tokens

        # Natural boundary patterns ordered by specificity (most specific first).
        self.boundary_patterns: list[re.Pattern[str]] = [
            re.compile(r"(?m)^#{1,6}\s+.+$"),              # Markdown headings
            re.compile(r"(?m)^第[一二三四五六七八九十百千]+[章节部篇]"),  # Chinese chapters
            re.compile(r"(?m)^\d+\.\d+[\s\.]"),            # Numbered 1.1, 1.2.3
            re.compile(r"\n\s*\n"),                         # Empty paragraph break
        ]

    # ── public API ─────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> list[dict]:
        """Chunk a raw text string by natural boundaries.

        Args:
            text: The raw text to split into chunks.

        Returns:
            A list of dicts, each with ``text``, ``index``, and ``token_count``.
        """
        segments = self._split_by_boundaries(text)
        chunks = self._merge_segments(segments)
        return [
            {"text": chunk, "index": i, "token_count": self._count_tokens(chunk)}
            for i, chunk in enumerate(chunks)
        ]

    def chunk_file(self, file_path: str) -> list[dict]:
        """Read a file and return its chunks.

        Args:
            file_path: Path to the file to chunk.

        Returns:
            A list of dicts, each with ``text``, ``index``, and ``token_count``.
        """
        from .utils import read_file  # local import, lazy

        text = read_file(file_path)
        return self.chunk_text(text)

    # ── internal ───────────────────────────────────────────────────────

    def _split_by_boundaries(self, text: str) -> list[str]:
        """Split *text* at natural boundary positions.

        Walks through each pattern, finds all match spans, and cuts the text
        at those positions.  The boundary marker itself is kept attached to
        the **following** segment.

        Returns:
            A list of non-empty segment strings.
        """
        # Collect all (start, end) match spans across every pattern.
        spans: list[tuple[int, int]] = []
        for pattern in self.boundary_patterns:
            spans.extend((m.start(), m.end()) for m in pattern.finditer(text))

        if not spans:
            # No boundaries found — return the whole text as one segment.
            stripped = text.strip()
            return [stripped] if stripped else []

        # Remove overlapping/contained spans (keep earliest first).
        spans = sorted(spans, key=lambda s: (s[0], -s[1]))
        filtered: list[tuple[int, int]] = []
        last_end = 0
        for start, end in spans:
            if start >= last_end:
                filtered.append((start, end))
                last_end = end

        # Cut text at each span.  The boundary text is attached to the next
        # segment so headings/chapter markers stay with the content below.
        parts: list[str] = []
        prev = 0
        for start, end in filtered:
            # Text before the boundary (the previous segment tail).
            before = text[prev:start]
            if before.strip():
                parts.append(before)
            # The boundary line itself becomes the start of the next segment.
            prev = start

        # Tail after the last boundary.
        tail = text[prev:]
        if tail.strip():
            parts.append(tail)

        return [p for p in parts if p.strip()]

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """Merge small segments so each chunk approaches *max_tokens*.

        A single segment that already exceeds *max_tokens* becomes its own
        chunk — the chunker does **not** split boundary-anchored segments.

        Returns:
            Merged chunk strings.
        """
        if not segments:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = self._count_tokens(seg)

            if seg_tokens > self.max_tokens:
                # Oversized segment — flush current, emit oversized alone.
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.append(seg)
                continue

            if current_tokens + seg_tokens <= self.max_tokens:
                current.append(seg)
                current_tokens += seg_tokens
            else:
                chunks.append("\n".join(current))
                current = [seg]
                current_tokens = seg_tokens

        if current:
            chunks.append("\n".join(current))

        return chunks

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Estimate token count using the ``cl100k_base`` encoding."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
