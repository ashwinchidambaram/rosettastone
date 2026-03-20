"""Tests for BERTScoreEvaluator and compute_bertscore."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestBERTScoreEvaluator:
    def test_bertscore_evaluator_returns_f1_score(self) -> None:
        """BERTScoreEvaluator.score() returns bertscore_f1 key with valid value."""
        import torch

        mock_bert_score_module = MagicMock()
        mock_bert_score_module.score.return_value = (
            torch.tensor([0.85]),
            torch.tensor([0.82]),
            torch.tensor([0.84]),
        )

        with patch.dict("sys.modules", {"bert_score": mock_bert_score_module}):
            # Force reimport to pick up mock
            if "rosettastone.evaluate.bertscore" in sys.modules:
                del sys.modules["rosettastone.evaluate.bertscore"]

            from rosettastone.evaluate.bertscore import BERTScoreEvaluator

            scores = BERTScoreEvaluator().score("expected text", "actual text")

        assert "bertscore_f1" in scores
        assert abs(scores["bertscore_f1"] - 0.84) < 1e-5

    def test_bertscore_f1_in_valid_range(self) -> None:
        """bertscore_f1 must be between 0.0 and 1.0."""
        import torch

        mock_bert_score_module = MagicMock()
        mock_bert_score_module.score.return_value = (
            torch.tensor([0.91]),
            torch.tensor([0.88]),
            torch.tensor([0.90]),
        )

        with patch.dict("sys.modules", {"bert_score": mock_bert_score_module}):
            if "rosettastone.evaluate.bertscore" in sys.modules:
                del sys.modules["rosettastone.evaluate.bertscore"]

            from rosettastone.evaluate.bertscore import BERTScoreEvaluator

            scores = BERTScoreEvaluator().score("hello world", "hello there")

        assert 0.0 <= scores["bertscore_f1"] <= 1.0

    def test_bertscore_evaluator_calls_score_with_correct_args(self) -> None:
        """Verifies bert_score.score is called with the right candidate/reference args."""
        import torch

        mock_bert_score_module = MagicMock()
        mock_bert_score_module.score.return_value = (
            torch.tensor([0.80]),
            torch.tensor([0.78]),
            torch.tensor([0.79]),
        )

        with patch.dict("sys.modules", {"bert_score": mock_bert_score_module}):
            if "rosettastone.evaluate.bertscore" in sys.modules:
                del sys.modules["rosettastone.evaluate.bertscore"]

            from rosettastone.evaluate.bertscore import BERTScoreEvaluator

            BERTScoreEvaluator().score("expected_text", "actual_text")

        mock_bert_score_module.score.assert_called_once()
        call_args = mock_bert_score_module.score.call_args
        # cands=[actual], refs=[expected]
        assert call_args[0][0] == ["actual_text"]
        assert call_args[0][1] == ["expected_text"]

    def test_compute_bertscore_returns_float(self) -> None:
        """compute_bertscore returns a float value."""
        import torch

        mock_bert_score_module = MagicMock()
        mock_bert_score_module.score.return_value = (
            torch.tensor([0.85]),
            torch.tensor([0.82]),
            torch.tensor([0.84]),
        )

        with patch.dict("sys.modules", {"bert_score": mock_bert_score_module}):
            if "rosettastone.evaluate.bertscore" in sys.modules:
                del sys.modules["rosettastone.evaluate.bertscore"]

            from rosettastone.evaluate.bertscore import compute_bertscore

            result = compute_bertscore("expected", "actual")

        assert isinstance(result, float)
        assert abs(result - 0.84) < 1e-5

    def test_bertscore_importerror_raises(self) -> None:
        """When bert_score is not installed, ImportError is raised."""
        with patch.dict("sys.modules", {"bert_score": None}):
            if "rosettastone.evaluate.bertscore" in sys.modules:
                del sys.modules["rosettastone.evaluate.bertscore"]

            from rosettastone.evaluate.bertscore import compute_bertscore

            with pytest.raises(ImportError):
                compute_bertscore("expected", "actual")

    def test_bertscore_evaluator_importerror_propagates(self) -> None:
        """BERTScoreEvaluator.score() propagates ImportError when bert_score unavailable."""
        with patch.dict("sys.modules", {"bert_score": None}):
            if "rosettastone.evaluate.bertscore" in sys.modules:
                del sys.modules["rosettastone.evaluate.bertscore"]

            from rosettastone.evaluate.bertscore import BERTScoreEvaluator

            with pytest.raises(ImportError):
                BERTScoreEvaluator().score("expected", "actual")

    def test_bertscore_returns_only_f1_key(self) -> None:
        """BERTScoreEvaluator.score() returns exactly one key: bertscore_f1."""
        import torch

        mock_bert_score_module = MagicMock()
        mock_bert_score_module.score.return_value = (
            torch.tensor([0.85]),
            torch.tensor([0.82]),
            torch.tensor([0.84]),
        )

        with patch.dict("sys.modules", {"bert_score": mock_bert_score_module}):
            if "rosettastone.evaluate.bertscore" in sys.modules:
                del sys.modules["rosettastone.evaluate.bertscore"]

            from rosettastone.evaluate.bertscore import BERTScoreEvaluator

            scores = BERTScoreEvaluator().score("a", "b")

        assert list(scores.keys()) == ["bertscore_f1"]
