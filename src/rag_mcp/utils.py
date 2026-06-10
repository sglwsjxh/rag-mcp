"""Lightweight file I/O and image encoding utilities."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

# ── lazy third-party imports (avoid hard deps if unused) ──────────────

def _get_pymupdf():
    """Import fitz (pymupdf) on first use."""
    import fitz  # noqa: PLC0415
    return fitz


def _get_markdown_it():
    """Import markdown_it on first use."""
    from markdown_it import MarkdownIt  # noqa: PLC0415
    return MarkdownIt


def _get_beautifulsoup():
    """Import bs4 on first use."""
    from bs4 import BeautifulSoup  # noqa: PLC0415
    return BeautifulSoup


# ── text helpers ──────────────────────────────────────────────────────

def _read_text_file(path: Path) -> str:
    """Read a plain-text file."""
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    fitz = _get_pymupdf()
    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _read_markdown(path: Path) -> str:
    """Extract inline text tokens from a Markdown file."""
    MarkdownIt = _get_markdown_it()
    src = _read_text_file(path)
    md = MarkdownIt()
    tokens = md.parse(src)
    parts = []
    for token in tokens:
        if token.type == "inline":
            parts.append(token.content)
    return "".join(parts)


def _read_html(path: Path) -> str:
    """Extract visible text from HTML, preserving line breaks."""
    BeautifulSoup = _get_beautifulsoup()
    src = _read_text_file(path)
    soup = BeautifulSoup(src, "html.parser")
    return soup.get_text(separator="\n")


# ── public API ────────────────────────────────────────────────────────

SUPPORTED_TEXT_EXTENSIONS: tuple[str, ...] = (
    ".pdf", ".md", ".markdown",
    ".html", ".htm",
    ".txt", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".csv", ".log", ".rst",
)


def read_file(file_path: str) -> str:
    """Read a file and return its content as plain text.

    Dispatches to the appropriate parser based on file extension:

    - ``.pdf``  → PyMuPDF (fitz) page-by-page extraction
    - ``.md`` / ``.markdown``  → markdown-it-py inline token extraction
    - ``.html`` / ``.htm``  → BeautifulSoup body text
    - ``.txt``, ``.py``, ``.js``, etc.  → raw UTF-8 text
    - others  → attempt UTF-8 read as fallback

    Raises:
        FileNotFoundError: if *file_path* does not exist.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _read_pdf(path)
    if ext in (".md", ".markdown"):
        return _read_markdown(path)
    if ext in (".html", ".htm"):
        return _read_html(path)

    # Fallback: treat everything else as plain text
    return _read_text_file(path)


def encode_image_to_data_uri(path: str) -> str:
    """Encode an image file as a ``data:image/<fmt>;base64,...`` URI.

    Supported formats: jpg, jpeg, png, webp.  Unknown extensions fall
    back to ``png``.
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower().lstrip(".")
    fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")

    with open(path_obj, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return f"data:image/{fmt};base64,{b64}"
