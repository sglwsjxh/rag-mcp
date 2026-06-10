from __future__ import annotations

import math
from dataclasses import dataclass

import httpx
from openai import OpenAI


@dataclass
class RerankResult:
    """Single rerank result."""

    index: int
    score: float


class Reranker:
    """NVIDIA Rerank API wrapper."""

    def __init__(self, settings) -> None:
        self.client = OpenAI(
            base_url=settings.rerank_baseurl,
            api_key=settings.rerank_api_key,
        )
        self.model = settings.rerank_model
        self.top_k = settings.rerank_top_k
        self.threshold = settings.rerank_score_threshold

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
        passages = [{"text": d["text"]} for d in documents]

        response = self.client.post(
            path=f"/retrieval/{self.model}/reranking",
            body={
                "model": self.model,
                "query": {"text": query},
                "passages": passages,
            },
            cast_to=httpx.Response,
        )
        data = response.json()

        # data = {"rankings": [{"index": 0, "logit": 0.95}, ...]}
        # logit is raw score; convert to probability via sigmoid for filtering.
        results = sorted(
            data["rankings"], key=lambda r: r["logit"], reverse=True
        )

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        if self.threshold is not None:
            results = [
                r for r in results
                if _sigmoid(r["logit"]) >= self.threshold
            ]

        limit = top_k if top_k is not None else self.top_k
        ranked: list[dict] = []
        for r in results[: limit] if limit is not None else results:
            doc = documents[r["index"]].copy()
            doc["score"] = _sigmoid(r["logit"])
            ranked.append(doc)

        return ranked
