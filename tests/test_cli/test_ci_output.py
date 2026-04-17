"""Tests for src/rosettastone/cli/ci_output.py — format_ci_json, format_pr_comment, format_quality_diff."""

from __future__ import annotations

import json

from rosettastone.cli.ci_output import format_ci_json, format_pr_comment, format_quality_diff
from tests.factories import migration_result_factory

# ---------------------------------------------------------------------------
# format_ci_json
# ---------------------------------------------------------------------------


class TestFormatCiJson:
    def test_format_ci_json_returns_valid_json(self):
        """Output must be parseable by json.loads()."""
        result = migration_result_factory()
        output = format_ci_json(result)
        # Should not raise
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_format_ci_json_includes_expected_keys(self):
        """Output JSON must contain all documented top-level keys."""
        result = migration_result_factory()
        parsed = json.loads(format_ci_json(result))
        expected_keys = {
            "recommendation",
            "confidence_score",
            "baseline_score",
            "improvement",
            "cost_usd",
            "duration_seconds",
            "warnings",
            "safety_warnings",
            "per_type_scores",
        }
        assert expected_keys.issubset(parsed.keys())

    def test_format_ci_json_floats_rounded(self):
        """Float values must be rounded to at most 4 decimal places."""
        result = migration_result_factory(
            confidence_score=0.123456789,
            baseline_score=0.987654321,
            improvement=0.111111111,
            cost_usd=1.999999999,
            duration_seconds=60.000000001,
        )
        parsed = json.loads(format_ci_json(result))

        float_fields = [
            "confidence_score",
            "baseline_score",
            "improvement",
            "cost_usd",
            "duration_seconds",
        ]
        for field in float_fields:
            value = parsed[field]
            # Check that rounding to 4 places doesn't change the value
            assert round(value, 4) == value, f"{field}={value} has more than 4 decimal places"

    def test_format_ci_json_with_none_recommendation(self):
        """recommendation=None must not crash and must appear as null in JSON."""
        result = migration_result_factory(recommendation=None)
        output = format_ci_json(result)
        parsed = json.loads(output)
        assert parsed["recommendation"] is None

    def test_format_ci_json_recommendation_go(self):
        """recommendation='GO' is serialised correctly."""
        result = migration_result_factory(recommendation="GO")
        parsed = json.loads(format_ci_json(result))
        assert parsed["recommendation"] == "GO"

    def test_format_ci_json_warnings_list(self):
        """Warnings list is preserved in output."""
        result = migration_result_factory(warnings=["Token budget exceeded", "Low win rate"])
        parsed = json.loads(format_ci_json(result))
        assert parsed["warnings"] == ["Token budget exceeded", "Low win rate"]

    def test_format_ci_json_empty_warnings(self):
        """Empty warnings produces an empty list, not null."""
        result = migration_result_factory(warnings=[])
        parsed = json.loads(format_ci_json(result))
        assert parsed["warnings"] == []

    def test_format_ci_json_per_type_scores(self):
        """per_type_scores dict is included in output."""
        per_type = {"short_text": {"avg_score": 0.85, "win_rate": 0.9}}
        result = migration_result_factory(per_type_scores=per_type)
        parsed = json.loads(format_ci_json(result))
        assert parsed["per_type_scores"] == per_type


# ---------------------------------------------------------------------------
# format_pr_comment
# ---------------------------------------------------------------------------


class TestFormatPrComment:
    def test_format_pr_comment_contains_header(self):
        """Output must contain the RosettaStone migration check header."""
        result = migration_result_factory()
        output = format_pr_comment(result, "openai/gpt-4o", "anthropic/claude-sonnet-4")
        assert "RosettaStone Migration Check" in output

    def test_format_pr_comment_contains_model_names(self):
        """Source and target model strings must appear in the output."""
        source = "openai/gpt-4o"
        target = "anthropic/claude-sonnet-4"
        result = migration_result_factory()
        output = format_pr_comment(result, source, target)
        assert source in output
        assert target in output

    def test_format_pr_comment_recommendation_emoji_go(self):
        """GO recommendation produces the checkmark emoji."""
        result = migration_result_factory(recommendation="GO")
        output = format_pr_comment(result, "src", "tgt")
        assert "✅" in output

    def test_format_pr_comment_recommendation_emoji_conditional(self):
        """CONDITIONAL recommendation produces the warning emoji."""
        result = migration_result_factory(recommendation="CONDITIONAL")
        output = format_pr_comment(result, "src", "tgt")
        assert "⚠️" in output

    def test_format_pr_comment_recommendation_emoji_no_go(self):
        """NO_GO recommendation produces the cross emoji."""
        result = migration_result_factory(recommendation="NO_GO")
        output = format_pr_comment(result, "src", "tgt")
        assert "❌" in output

    def test_format_pr_comment_recommendation_emoji_none(self):
        """None recommendation produces the question-mark emoji."""
        result = migration_result_factory(recommendation=None)
        output = format_pr_comment(result, "src", "tgt")
        assert "❓" in output

    def test_format_pr_comment_with_warnings(self):
        """Warnings section appears when result.warnings is non-empty."""
        result = migration_result_factory(warnings=["Token budget exceeded"])
        output = format_pr_comment(result, "src", "tgt")
        assert "Warnings" in output
        assert "Token budget exceeded" in output

    def test_format_pr_comment_without_warnings(self):
        """Warnings section is absent when result.warnings is empty."""
        result = migration_result_factory(warnings=[])
        output = format_pr_comment(result, "src", "tgt")
        # The word "Warnings" should not appear (no warnings section)
        assert "### ⚠️ Warnings" not in output

    def test_format_pr_comment_safety_warnings_section(self):
        """Safety warnings section appears when result.safety_warnings is non-empty."""
        result = migration_result_factory(safety_warnings=["Context window exceeded."])
        output = format_pr_comment(result, "src", "tgt")
        assert "Safety" in output
        assert "Context window exceeded." in output

    def test_format_pr_comment_no_safety_warnings_section_when_empty(self):
        """Safety warnings section is absent when result.safety_warnings is empty."""
        result = migration_result_factory(safety_warnings=[])
        output = format_pr_comment(result, "src", "tgt")
        assert "### 🔒 Safety" not in output

    def test_format_pr_comment_contains_score_table(self):
        """Output includes a markdown table with Confidence, Baseline, and Improvement rows."""
        result = migration_result_factory(
            confidence_score=0.92,
            baseline_score=0.70,
            improvement=0.22,
        )
        output = format_pr_comment(result, "src", "tgt")
        assert "Confidence" in output
        assert "Baseline" in output
        assert "Improvement" in output

    def test_format_pr_comment_contains_cost(self):
        """Output includes cost in dollars."""
        result = migration_result_factory(cost_usd=3.50)
        output = format_pr_comment(result, "src", "tgt")
        assert "$" in output
        assert "3.50" in output

    def test_format_pr_comment_returns_string(self):
        """Return type is a str."""
        result = migration_result_factory()
        output = format_pr_comment(result, "src", "tgt")
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# format_quality_diff
# ---------------------------------------------------------------------------


class TestFormatQualityDiff:
    def test_format_quality_diff_current_only_shows_scores(self):
        """When baseline is None, current scores are shown."""
        result = migration_result_factory(
            confidence_score=0.88,
            baseline_score=0.72,
            improvement=0.16,
        )
        output = format_quality_diff(result, baseline=None)
        assert "Current Scores" in output
        assert "Confidence" in output
        assert "Baseline Score" in output
        assert "Improvement" in output

    def test_format_quality_diff_current_only_contains_header(self):
        """Output has a Quality Report header."""
        result = migration_result_factory()
        output = format_quality_diff(result)
        assert "Quality Report" in output

    def test_format_quality_diff_with_baseline_shows_comparison(self):
        """When baseline is provided, shows a comparison table."""
        current = migration_result_factory(
            confidence_score=0.88,
            baseline_score=0.72,
            improvement=0.16,
        )
        baseline = migration_result_factory(
            confidence_score=0.80,
            baseline_score=0.68,
            improvement=0.12,
        )
        output = format_quality_diff(current, baseline=baseline)
        assert "Score Comparison" in output
        assert "Current" in output
        assert "Baseline" in output
        assert "Delta" in output

    def test_format_quality_diff_positive_delta_rendered(self):
        """Positive delta (improvement) is shown with + sign."""
        current = migration_result_factory(
            confidence_score=0.90, baseline_score=0.70, improvement=0.20
        )
        base = migration_result_factory(
            confidence_score=0.80, baseline_score=0.65, improvement=0.15
        )
        output = format_quality_diff(current, baseline=base)
        assert "+" in output  # positive delta indicator

    def test_format_quality_diff_negative_delta(self):
        """Negative delta (regression) is shown with - sign."""
        current = migration_result_factory(
            confidence_score=0.75, baseline_score=0.60, improvement=0.05
        )
        base = migration_result_factory(
            confidence_score=0.85, baseline_score=0.70, improvement=0.15
        )
        output = format_quality_diff(current, baseline=base)
        assert "-" in output  # negative delta indicator

    def test_format_quality_diff_per_type_breakdown_included(self):
        """Per-type scores appear when current.per_type_scores is non-empty."""
        per_type = {"short_text": {"avg_score": 0.88}}
        result = migration_result_factory(per_type_scores=per_type)
        output = format_quality_diff(result)
        assert "Per-Type Breakdown" in output
        assert "short_text" in output

    def test_format_quality_diff_no_per_type_breakdown_when_empty(self):
        """Per-type section absent when current.per_type_scores is empty."""
        result = migration_result_factory(per_type_scores={})
        output = format_quality_diff(result)
        assert "Per-Type Breakdown" not in output

    def test_format_quality_diff_returns_string(self):
        """Return type is always a str."""
        result = migration_result_factory()
        output = format_quality_diff(result)
        assert isinstance(output, str)

    def test_format_quality_diff_default_baseline_is_none(self):
        """Calling without baseline arg (default None) shows current-only report."""
        result = migration_result_factory()
        output = format_quality_diff(result)
        assert "Current Scores" in output
        assert "Score Comparison" not in output
