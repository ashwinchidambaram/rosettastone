"""Tests for executive narrative generator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rosettastone.core.types import EvalResult, MigrationResult, PromptPair
from rosettastone.report.narrative import (
    EXECUTIVE_PROMPT,
    _basic_summary,
    _format_per_type,
    _format_safety,
    generate_executive_narrative,
)


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
                    response="expected",
                    source_model="openai/gpt-4o",
                ),
                new_response="actual",
                scores={"bertscore_f1": 0.92},
                composite_score=0.92,
                is_win=True,
            ),
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="another",
                    response="expected2",
                    source_model="openai/gpt-4o",
                ),
                new_response="actual2",
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


class TestFormatPerType:
    """Tests for _format_per_type helper."""

    def test_empty_dict(self):
        assert _format_per_type({}) == "No per-type breakdown available."

    def test_with_dict_stats(self):
        per_type = {
            "json": {
                "win_rate": 0.85,
                "sample_count": 20,
                "mean": 0.88,
                "p10": 0.72,
                "p50": 0.89,
                "p90": 0.96,
            },
            "short_text": {
                "win_rate": 0.92,
                "sample_count": 15,
                "mean": 0.91,
                "p10": 0.80,
                "p50": 0.93,
                "p90": 0.98,
            },
        }
        result = _format_per_type(per_type)
        assert "json" in result
        assert "short_text" in result
        assert "85.0%" in result
        assert "92.0%" in result
        assert "samples=20" in result
        assert "samples=15" in result

    def test_with_dataclass_stats(self):
        import dataclasses

        @dataclasses.dataclass
        class TypeStats:
            win_rate: float
            sample_count: int
            mean: float
            p10: float
            p50: float
            p90: float

        per_type = {
            "classification": TypeStats(
                win_rate=0.75, sample_count=8, mean=0.80, p10=0.65, p50=0.82, p90=0.91
            ),
        }
        result = _format_per_type(per_type)
        assert "classification" in result
        assert "75.0%" in result

    def test_skips_unknown_types(self):
        per_type = {"bad": "not a dict or dataclass"}
        result = _format_per_type(per_type)
        assert result == "No per-type data."


class TestFormatSafety:
    """Tests for _format_safety helper."""

    def test_no_warnings(self):
        assert _format_safety([]) == "No safety issues found."

    def test_dict_warnings(self):
        warnings = [
            {"severity": "HIGH", "message": "Toxicity detected"},
            {"severity": "LOW", "message": "Minor style drift"},
        ]
        result = _format_safety(warnings)
        assert "[HIGH]" in result
        assert "Toxicity detected" in result
        assert "[LOW]" in result
        assert "Minor style drift" in result

    def test_object_warnings(self):
        class SafetyWarning:
            def __init__(self, severity, message):
                self.severity = severity
                self.message = message

        warnings = [SafetyWarning("CRITICAL", "Harmful content")]
        result = _format_safety(warnings)
        assert "[CRITICAL]" in result
        assert "Harmful content" in result

    def test_string_warnings(self):
        warnings = ["Something went wrong"]
        result = _format_safety(warnings)
        assert "Something went wrong" in result

    def test_mixed_warning_types(self):
        warnings = [
            {"severity": "HIGH", "message": "Dict warning"},
            "Plain string warning",
        ]
        result = _format_safety(warnings)
        assert "[HIGH]" in result
        assert "Plain string warning" in result


class TestGenerateExecutiveNarrative:
    """Tests for generate_executive_narrative function."""

    def test_local_only_uses_template_fallback(self):
        """With local_only=True, the function skips LLM and uses template."""
        result = _make_result()
        narrative = generate_executive_narrative(result, local_only=True)

        assert isinstance(narrative, str)
        assert len(narrative) > 0
        # Should contain key information
        assert "openai/gpt-4o" in narrative
        assert "anthropic/claude-sonnet-4" in narrative

    def test_local_only_contains_metrics(self):
        """Template fallback includes key metrics."""
        result = _make_result()
        narrative = generate_executive_narrative(result, local_only=True)

        assert "85.0%" in narrative  # confidence_score
        assert "70.0%" in narrative  # baseline_score

    def test_local_only_with_empty_results(self):
        """Template fallback handles empty validation results."""
        result = _make_result(
            validation_results=[],
            confidence_score=0.0,
            baseline_score=0.0,
            improvement=0.0,
        )
        narrative = generate_executive_narrative(result, local_only=True)
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_llm_failure_falls_back_to_template(self):
        """When LiteLLM call fails, function falls back to template."""
        result = _make_result()

        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = RuntimeError("API error")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            narrative = generate_executive_narrative(result)

        assert isinstance(narrative, str)
        assert len(narrative) > 0
        # Should still contain key info from template fallback
        assert "openai/gpt-4o" in narrative


class TestBasicSummary:
    """Tests for _basic_summary function."""

    def test_produces_readable_output(self):
        result = _make_result()
        summary = _basic_summary(result, {})

        assert "openai/gpt-4o" in summary
        assert "anthropic/claude-sonnet-4" in summary
        assert "85.0%" in summary  # confidence
        assert "70.0%" in summary  # baseline
        assert "+15.0%" in summary  # improvement
        assert "1/2" in summary  # win rate
        assert "$0.0523" in summary  # cost
        assert "42.7s" in summary  # duration

    def test_handles_zero_results(self):
        result = _make_result(validation_results=[])
        summary = _basic_summary(result, {})

        assert "0/0" in summary
        assert isinstance(summary, str)

    def test_handles_missing_config_fields(self):
        result = _make_result(config={})
        summary = _basic_summary(result, {})

        assert "unknown" in summary


class TestExecutivePrompt:
    """Tests for the EXECUTIVE_PROMPT constant."""

    def test_no_pii_instruction_present(self):
        """Prompt explicitly instructs LLM to not include raw prompt/response content."""
        assert "Do NOT include any raw prompt or response content" in EXECUTIVE_PROMPT

    def test_prompt_has_placeholder_fields(self):
        """Prompt contains all expected format placeholders."""
        expected_fields = [
            "source_model",
            "target_model",
            "recommendation",
            "confidence_score",
            "baseline_score",
            "improvement",
            "cost_usd",
            "duration_seconds",
            "total_test_cases",
            "wins",
            "per_type_summary",
            "safety_summary",
            "warnings_summary",
        ]
        for field in expected_fields:
            assert f"{{{field}" in EXECUTIVE_PROMPT, f"Missing placeholder: {field}"
