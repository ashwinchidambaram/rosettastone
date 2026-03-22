"""Tests for PDF report generator."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.core.types import EvalResult, MigrationResult, PromptPair
from rosettastone.report.pdf_generator import _stats_to_dict, generate_pdf_report


def _make_result(**overrides) -> MigrationResult:
    """Create a sample MigrationResult for testing."""
    defaults = dict(
        config={
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
        },
        optimized_prompt="You are a helpful assistant.",
        baseline_results=[],
        validation_results=[
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="test prompt",
                    response="expected response",
                    source_model="openai/gpt-4o",
                ),
                new_response="actual response",
                scores={"bertscore_f1": 0.92},
                composite_score=0.92,
                is_win=True,
            ),
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="another prompt",
                    response="another expected",
                    source_model="openai/gpt-4o",
                ),
                new_response="another actual",
                scores={"bertscore_f1": 0.78},
                composite_score=0.78,
                is_win=False,
            ),
        ],
        confidence_score=0.85,
        baseline_score=0.70,
        improvement=0.15,
        cost_usd=0.0523,
        duration_seconds=42.7,
        warnings=["Sample size below recommended minimum"],
    )
    defaults.update(overrides)
    return MigrationResult(**defaults)


class TestStatsToDict:
    """Tests for _stats_to_dict helper."""

    def test_dict_passthrough(self):
        d = {"win_rate": 0.8, "sample_count": 10}
        assert _stats_to_dict(d) == d

    def test_dataclass_conversion(self):
        @dataclasses.dataclass
        class TypeStats:
            win_rate: float
            sample_count: int

        stats = TypeStats(win_rate=0.9, sample_count=5)
        result = _stats_to_dict(stats)
        assert result == {"win_rate": 0.9, "sample_count": 5}

    def test_unknown_type_returns_empty(self):
        assert _stats_to_dict("not a dict or dataclass") == {}
        assert _stats_to_dict(42) == {}
        assert _stats_to_dict(None) == {}


class TestGeneratePdfReport:
    """Tests for generate_pdf_report function."""

    def test_raises_importerror_without_weasyprint(self, tmp_path):
        """PDF generation raises ImportError with helpful message when weasyprint missing."""
        result = _make_result()

        with patch.dict("sys.modules", {"weasyprint": None}):
            with pytest.raises(ImportError, match="weasyprint is required"):
                generate_pdf_report(result, tmp_path)

    def test_calls_weasyprint_correctly(self, tmp_path):
        """With weasyprint mocked, function renders HTML and calls weasyprint.HTML."""
        result = _make_result()

        mock_weasyprint = MagicMock()
        mock_html_instance = MagicMock()
        mock_weasyprint.HTML.return_value = mock_html_instance

        with patch.dict("sys.modules", {"weasyprint": mock_weasyprint}):
            output_path = generate_pdf_report(result, tmp_path)

        # Verify weasyprint.HTML was called with an HTML string
        mock_weasyprint.HTML.assert_called_once()
        call_kwargs = mock_weasyprint.HTML.call_args
        html_string = (
            call_kwargs.kwargs.get("string") or call_kwargs.args[0]
            if call_kwargs.args
            else call_kwargs[1].get("string")
        )
        assert html_string is not None
        assert "<!DOCTYPE html>" in html_string
        assert "Migration Report" in html_string

        # Verify write_pdf was called with the output path
        mock_html_instance.write_pdf.assert_called_once_with(str(tmp_path / "migration_report.pdf"))

        assert output_path == tmp_path / "migration_report.pdf"

    def test_creates_output_directory(self, tmp_path):
        """PDF generation creates output directory if it doesn't exist."""
        result = _make_result()
        output_dir = tmp_path / "nested" / "dir"

        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML.return_value = MagicMock()

        with patch.dict("sys.modules", {"weasyprint": mock_weasyprint}):
            generate_pdf_report(result, output_dir)

        assert output_dir.exists()

    def test_html_contains_no_pii(self, tmp_path):
        """Generated HTML does not contain prompt/response content in per-case section."""
        result = _make_result()

        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML.return_value = MagicMock()

        with patch.dict("sys.modules", {"weasyprint": mock_weasyprint}):
            generate_pdf_report(result, tmp_path)

        call_kwargs = mock_weasyprint.HTML.call_args
        html_string = call_kwargs.kwargs.get("string") or call_kwargs[1].get("string")
        # Template should not expose raw prompt/response content
        assert "expected response" not in html_string
        assert "actual response" not in html_string
