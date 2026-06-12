"""Smoke test for reranker confidence annotation."""

import unittest
from unittest.mock import MagicMock, patch


class MockSettings:
    """Minimal settings stub for Reranker."""

    def __init__(self):
        self.rerank_api_key = "test-key"
        self.rerank_baseurl = "http://localhost:8080/v1"
        self.rerank_model = "test-model"
        self.rerank_top_k = None
        self.rerank_score_threshold = None
        self.rerank_input_token = None
        self.confidence_threshold = 0.3


class TestRerankerConfidence(unittest.TestCase):
    """Test that Reranker annotates each result with a confidence field."""

    def _make_documents(self, count=3):
        return [{"text": f"Document {i} content here"} for i in range(count)]

    @patch("rag_mcp.reranker.OpenAI")
    def test_high_confidence_when_score_above_threshold(self, mock_openai_cls):
        """Docs with score >= 0.3 get confidence='high'."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Simulate a rerank response with logit that maps to score ~0.5
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rankings": [
                {"index": 0, "logit": 0.405},  # sigmoid(0.405) ~ 0.6
                {"index": 1, "logit": 0.405},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        from rag_mcp.reranker import Reranker

        settings = MockSettings()
        settings.confidence_threshold = 0.3
        reranker = Reranker(settings)

        docs = self._make_documents(3)
        result = reranker.rerank("test query", docs)

        self.assertEqual(len(result), 2)
        for doc in result:
            self.assertIn("confidence", doc)
            self.assertEqual(doc["confidence"], "high")

    @patch("rag_mcp.reranker.OpenAI")
    def test_low_confidence_when_score_below_threshold(self, mock_openai_cls):
        """Docs with score < 0.3 get confidence='low'."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Simulate a rerank response with logit that maps to score ~0.2
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rankings": [
                {"index": 2, "logit": -1.099},  # sigmoid(-1.099) ~ 0.25
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        from rag_mcp.reranker import Reranker

        settings = MockSettings()
        settings.confidence_threshold = 0.3
        reranker = Reranker(settings)

        docs = self._make_documents(3)
        result = reranker.rerank("test query", docs)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["confidence"], "low")

    @patch("rag_mcp.reranker.OpenAI")
    def test_fallback_sets_unknown_confidence(self, mock_openai_cls):
        """When rerank is disabled, all docs get confidence='unknown'."""
        from rag_mcp.reranker import Reranker

        settings = MockSettings()
        settings.rerank_api_key = None  # disable rerank
        settings.confidence_threshold = 0.3
        reranker = Reranker(settings)

        docs = self._make_documents(3)
        result = reranker.rerank("test query", docs)

        self.assertEqual(len(result), 3)
        for doc in result:
            self.assertEqual(doc["confidence"], "unknown")
            self.assertEqual(doc["score"], 0.0)

    @patch("rag_mcp.reranker.OpenAI")
    def test_exception_triggers_fallback_unknown(self, mock_openai_cls):
        """When API raises, fallback gives confidence='unknown'."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_client.post.return_value = mock_response

        from rag_mcp.reranker import Reranker

        settings = MockSettings()
        settings.confidence_threshold = 0.3
        reranker = Reranker(settings)

        docs = self._make_documents(2)
        result = reranker.rerank("test query", docs)

        self.assertEqual(len(result), 2)
        for doc in result:
            self.assertEqual(doc["confidence"], "unknown")


if __name__ == "__main__":
    unittest.main()
