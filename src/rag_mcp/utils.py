"""Lightweight file I/O and image encoding utilities."""

from __future__ import annotations

import base64
from pathlib import Path

# ── lazy third-party imports (avoid hard deps if unused) ──────────────

def _get_beautifulsoup():
    """Import bs4 on first use."""
    from bs4 import BeautifulSoup  # noqa: PLC0415
    return BeautifulSoup


# ── text helpers ──────────────────────────────────────────────────────

def _read_text_file(path: Path) -> str:
    """Read a plain-text file."""
    return path.read_text(encoding="utf-8")


def _read_html(path: Path) -> str:
    """Extract visible text from HTML, preserving line breaks."""
    BeautifulSoup = _get_beautifulsoup()
    src = _read_text_file(path)
    soup = BeautifulSoup(src, "html.parser")
    return soup.get_text(separator="\n")


# ── public API ────────────────────────────────────────────────────────

SUPPORTED_TEXT_EXTENSIONS: tuple[str, ...] = (
    ".html", ".htm",
    ".txt", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".csv", ".log", ".rst",
)


def read_file(file_path: str) -> str:
    """Read a file and return its content as plain text.

    Dispatches to the appropriate parser based on file extension:

    - ``.html`` / ``.htm``  → BeautifulSoup body text
    - Everything else (``.txt``, ``.md``, code files, etc.) → raw UTF-8 read

    Raises:
        FileNotFoundError: if *file_path* does not exist.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

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
