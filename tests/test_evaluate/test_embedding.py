"""Tests for EmbeddingEvaluator and compute_embedding_sim."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbeddingEvaluator:
    def _make_mock_st_module(self, encode_return: np.ndarray) -> MagicMock:
        """Build a mock sentence_transformers module with a mock SentenceTransformer class."""
        mock_model = MagicMock()
        mock_model.encode.return_value = encode_return

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        return mock_st_module

    def _clear_module_cache(self) -> None:
        for key in list(sys.modules.keys()):
            if "rosettastone.evaluate.embedding" in key:
                del sys.modules[key]

    def test_embedding_evaluator_returns_embedding_sim_key(self) -> None:
        """EmbeddingEvaluator.score() returns embedding_sim key."""
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            scores = EmbeddingEvaluator().score("expected text", "actual text")

        assert "embedding_sim" in scores

    def test_embedding_sim_in_valid_range(self) -> None:
        """embedding_sim is between -1.0 and 1.0 (cosine similarity range)."""
        embeddings = np.array([[1.0, 0.0], [0.9, 0.436]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            scores = EmbeddingEvaluator().score("hello world", "hello there")

        assert -1.0 <= scores["embedding_sim"] <= 1.0

    def test_identical_embeddings_give_similarity_one(self) -> None:
        """Identical embedding vectors should produce cosine similarity of 1.0."""
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            scores = EmbeddingEvaluator().score("same", "same")

        assert abs(scores["embedding_sim"] - 1.0) < 1e-6

    def test_orthogonal_embeddings_give_similarity_zero(self) -> None:
        """Orthogonal embedding vectors should produce cosine similarity of 0.0."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            scores = EmbeddingEvaluator().score("up", "sideways")

        assert abs(scores["embedding_sim"] - 0.0) < 1e-6

    def test_embedding_evaluator_uses_correct_model(self) -> None:
        """EmbeddingEvaluator instantiates SentenceTransformer with all-MiniLM-L6-v2."""
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            EmbeddingEvaluator().score("expected", "actual")

        mock_st_module.SentenceTransformer.assert_called_once_with("all-MiniLM-L6-v2")

    def test_embedding_evaluator_encodes_both_texts(self) -> None:
        """encode() is called with both expected and actual."""
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            EmbeddingEvaluator().score("expected text", "actual text")

        mock_model = mock_st_module.SentenceTransformer.return_value
        mock_model.encode.assert_called_once_with(["expected text", "actual text"])

    def test_compute_embedding_sim_returns_float(self) -> None:
        """compute_embedding_sim returns a float."""
        embeddings = np.array([[1.0, 0.0], [0.8, 0.6]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import compute_embedding_sim

            result = compute_embedding_sim("expected", "actual")

        assert isinstance(result, float)

    def test_embedding_importerror_raises(self) -> None:
        """When sentence_transformers is not installed, ImportError is raised."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import compute_embedding_sim

            with pytest.raises(ImportError):
                compute_embedding_sim("expected", "actual")

    def test_embedding_evaluator_importerror_propagates(self) -> None:
        """EmbeddingEvaluator.score() propagates ImportError when sentence_transformers unavailable."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            with pytest.raises(ImportError):
                EmbeddingEvaluator().score("expected", "actual")

    def test_embedding_returns_only_embedding_sim_key(self) -> None:
        """EmbeddingEvaluator.score() returns exactly one key: embedding_sim."""
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
        mock_st_module = self._make_mock_st_module(embeddings)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            self._clear_module_cache()
            from rosettastone.evaluate.embedding import EmbeddingEvaluator

            scores = EmbeddingEvaluator().score("a", "b")

        assert list(scores.keys()) == ["embedding_sim"]
