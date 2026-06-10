from __future__ import annotations

import math

import httpx
from openai import OpenAI

from .chunker import Chunker


def _sigmoid(x: float) -> float:
    """Sigmoid function for converting logits to probabilities."""
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    """NVIDIA Rerank API wrapper."""

    def __init__(self, settings) -> None:
        self.enabled = bool(settings.rerank_api_key)
        if not self.enabled:
            return
        self.client = OpenAI(
            base_url=settings.rerank_baseurl,
            api_key=settings.rerank_api_key,
        )
        self.model = settings.rerank_model
        self.top_k = settings.rerank_top_k
        self.threshold = settings.rerank_score_threshold
        self.max_input_tokens = settings.rerank_input_token or 8192

    def rerank(self, query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query string.
            documents: List of dicts, each containing at least a "text" key.
            top_k: Max results to return. Overrides self.top_k if set.

        Returns:
            Ranked document list with "score" field appended, sorted by
            logit descending. Documents below self.threshold are filtered
            out. At most top_k (or self.top_k) documents are returned.
        """
        if not self.enabled:
            return self._fallback(documents)

        try:
            # Truncate passages before sending
            passages = [
                {"text": Chunker.truncate(d["text"], self.max_input_tokens)}
                for d in documents
            ]

            response = self.client.post(
                path=f"/retrieval/{self.model}/reranking",
                body={
                    "model": self.model,
                    "query": {"text": query},
                    "passages": passages,
                },
                cast_to=httpx.Response,
            )
            response.raise_for_status()
            data = response.json()

            rankings = data.get("rankings")
            if rankings is None:
                raise ValueError("Rerank response missing 'rankings' field")
            for r in rankings:
                if "index" not in r or "logit" not in r:
                    raise ValueError(f"Invalid ranking entry: {r}")

            results = sorted(rankings, key=lambda r: r["logit"], reverse=True)

            if self.threshold is not None:
                results = [
                    r for r in results
                    if _sigmoid(r["logit"]) >= self.threshold
                ]

            ranked: list[dict] = []
            limit = top_k if top_k is not None else self.top_k
            top = results[:limit] if limit is not None else results
            for r in top:
                doc = documents[r["index"]].copy()
                doc["score"] = _sigmoid(r["logit"])
                ranked.append(doc)

            return ranked

        except Exception:
            return self._fallback(documents)

    def _fallback(self, documents: list[dict]) -> list[dict]:
        docs = sorted(documents, key=lambda d: d.get("_raw_score", 0))
        for d in docs:
            d["score"] = 0.0
        return docs
