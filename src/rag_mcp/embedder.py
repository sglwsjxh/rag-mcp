"""OpenRouter Embedding API wrapper for text and multimodal inputs."""

from __future__ import annotations

from openai import OpenAI

from .config import Settings
from .utils import encode_image_to_data_uri

settings = Settings()


class Embedder:
    """Wrap OpenRouter Embedding API for text and multimodal (text+image) inputs.

    All methods are synchronous to keep the codebase consistent.
    """

    top_k: int = settings.embedding_top_k

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=settings.embedding_baseurl,
            api_key=settings.embedding_api_key,
        )
        self.model = settings.embedding_model
        self.detected_dimension: int | None = None

    # ── single / multimodal ───────────────────────────────────────────

    def embed(
        self,
        text: str = "",
        images: list[str] | None = None,
    ) -> list[float]:
        """Generate an embedding for text, optionally with images.

        No branching on whether images exist — always build the content array:

        - Pure text → ``[{"type": "text", "text": text}]``
        - With images → append ``{"type": "image_url", "image_url": {"url": ...}}``
          for each image. URLs starting with ``http://``, ``https://`` or
          ``data:image/`` are passed through; local paths are converted to
          ``data:image/...`` URIs via :func:`encode_image_to_data_uri`.

        Args:
            text: The text to embed.
            images: List of image paths or URLs.

        Returns:
            The embedding vector as ``list[float]``.
        """
        content: list[dict] = [{"type": "text", "text": text}]

        if images:
            for img in images:
                if img.startswith("http://") or img.startswith("https://") or img.startswith("data:image/"):
                    content.append({"type": "image_url", "image_url": {"url": img}})
                else:
                    data_uri = encode_image_to_data_uri(img)
                    content.append({"type": "image_url", "image_url": {"url": data_uri}})

        resp = self.client.embeddings.create(
            model=self.model,
            input=[{"content": content}],
            encoding_format="float",
        )

        vec = resp.data[0].embedding
        dim = len(vec)
        if self.detected_dimension is None:
            self.detected_dimension = dim
        elif dim != self.detected_dimension:
            raise ValueError(
                f"Embedding dimension changed from {self.detected_dimension} "
                f"to {dim}. Did you change the model?"
            )
        return vec

    # ── batch pure text ───────────────────────────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed pure texts, bypassing the content-array format.

        ``input`` is the raw ``list[str]`` — faster for ingestion pipelines.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )

        vecs: list[list[float]] = []
        for d in resp.data:
            v = d.embedding
            dim = len(v)
            if self.detected_dimension is None:
                self.detected_dimension = dim
            elif dim != self.detected_dimension:
                raise ValueError(
                    f"Embedding dimension changed from {self.detected_dimension} "
                    f"to {dim}. Did you change the model?"
                )
            vecs.append(v)
        return vecs
