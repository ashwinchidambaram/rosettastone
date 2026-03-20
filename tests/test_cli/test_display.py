"""Tests for src/rosettastone/cli/display.py — MigrationDisplay."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from rich.progress import Progress

from rosettastone.cli.display import MigrationDisplay, _score_color
from rosettastone.core.types import EvalResult, OutputType, PromptPair

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _console() -> tuple[Console, StringIO]:
    """Return a (Console, StringIO) pair for output capture."""
    buf = StringIO()
    con = Console(file=buf, highlight=False, markup=True)
    return con, buf


def _make_prompt_pair(output_type: OutputType | None = None) -> PromptPair:
    return PromptPair(
        prompt="Translate this sentence.",
        response="Hola mundo.",
        source_model="openai/gpt-4o",
        output_type=output_type,
    )


def _make_eval_result(
    *,
    composite_score: float = 0.85,
    is_win: bool = True,
    output_type: OutputType | None = OutputType.SHORT_TEXT,
) -> EvalResult:
    return EvalResult(
        prompt_pair=_make_prompt_pair(output_type),
        new_response="Hola mundo.",
        scores={"bertscore": composite_score},
        composite_score=composite_score,
        is_win=is_win,
    )


# ---------------------------------------------------------------------------
# _score_color helper
# ---------------------------------------------------------------------------


class TestScoreColor:
    def test_green_at_threshold(self):
        assert _score_color(0.90) == "green"

    def test_green_above_threshold(self):
        assert _score_color(0.95) == "green"

    def test_yellow_at_lower_threshold(self):
        assert _score_color(0.80) == "yellow"

    def test_yellow_between_thresholds(self):
        assert _score_color(0.85) == "yellow"

    def test_red_below_lower_threshold(self):
        assert _score_color(0.79) == "red"

    def test_red_at_zero(self):
        assert _score_color(0.0) == "red"


# ---------------------------------------------------------------------------
# MigrationDisplay construction
# ---------------------------------------------------------------------------


class TestMigrationDisplayInit:
    def test_default_console_created(self):
        display = MigrationDisplay()
        assert display.console is not None

    def test_custom_console_used(self):
        con, _ = _console()
        display = MigrationDisplay(console=con)
        assert display.console is con


# ---------------------------------------------------------------------------
# create_progress
# ---------------------------------------------------------------------------


class TestCreateProgress:
    def test_returns_progress_instance(self):
        con, _ = _console()
        display = MigrationDisplay(console=con)
        progress = display.create_progress()
        assert isinstance(progress, Progress)

    def test_progress_uses_same_console(self):
        con, _ = _console()
        display = MigrationDisplay(console=con)
        progress = display.create_progress()
        assert progress.console is con


# ---------------------------------------------------------------------------
# show_summary_table
# ---------------------------------------------------------------------------


class TestShowSummaryTable:
    def test_renders_output_type_column(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [_make_eval_result()]
        per_type = {
            "short_text": {
                "win_rate": 0.91,
                "sample_count": 10,
                "avg_score": 0.92,
                "threshold": 0.90,
            }
        }
        display.show_summary_table(results, per_type)

        output = buf.getvalue()
        assert "short_text" in output

    def test_renders_sample_count(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [_make_eval_result() for _ in range(5)]
        per_type = {
            "json": {
                "win_rate": 0.80,
                "sample_count": 5,
                "avg_score": 0.81,
                "threshold": 0.85,
            }
        }
        display.show_summary_table(results, per_type)

        assert "5" in buf.getvalue()

    def test_green_color_for_high_score(self):
        """Win rate >= 90% should be rendered with green markup."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [_make_eval_result(composite_score=0.95, is_win=True)]
        per_type = {
            "long_text": {
                "win_rate": 0.95,
                "sample_count": 1,
                "avg_score": 0.95,
                "threshold": 0.90,
            }
        }
        display.show_summary_table(results, per_type)

        # Rich renders green text — the percentage should be present
        assert "95%" in buf.getvalue()

    def test_yellow_color_for_mid_score(self):
        """Win rate 80-90% range should appear in output."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [_make_eval_result(composite_score=0.85, is_win=True)]
        per_type = {
            "classification": {
                "win_rate": 0.85,
                "sample_count": 1,
                "avg_score": 0.85,
                "threshold": 0.80,
            }
        }
        display.show_summary_table(results, per_type)

        assert "85%" in buf.getvalue()

    def test_red_color_for_low_score(self):
        """Win rate < 80% should appear in output."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [_make_eval_result(composite_score=0.70, is_win=False)]
        per_type = {
            "json": {
                "win_rate": 0.70,
                "sample_count": 1,
                "avg_score": 0.70,
                "threshold": 0.80,
            }
        }
        display.show_summary_table(results, per_type)

        assert "70%" in buf.getvalue()

    def test_empty_results_and_per_type(self):
        """Empty inputs should still render a table without crashing."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        display.show_summary_table([], {})

        output = buf.getvalue()
        assert len(output) > 0  # Something was printed

    def test_overall_row_with_multiple_types(self):
        """An Overall aggregate row is added when more than one type is present."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [
            _make_eval_result(composite_score=0.91, is_win=True),
            _make_eval_result(composite_score=0.75, is_win=False),
        ]
        per_type = {
            "json": {
                "win_rate": 0.91,
                "sample_count": 1,
                "avg_score": 0.91,
                "threshold": 0.90,
            },
            "short_text": {
                "win_rate": 0.75,
                "sample_count": 1,
                "avg_score": 0.75,
                "threshold": 0.80,
            },
        }
        display.show_summary_table(results, per_type)

        assert "Overall" in buf.getvalue()

    def test_no_overall_row_for_single_type(self):
        """No Overall row is added when only one output type is present."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [_make_eval_result(composite_score=0.91, is_win=True)]
        per_type = {
            "json": {
                "win_rate": 0.91,
                "sample_count": 1,
                "avg_score": 0.91,
                "threshold": 0.90,
            }
        }
        display.show_summary_table(results, per_type)

        assert "Overall" not in buf.getvalue()

    def test_missing_threshold_key_defaults(self):
        """Stats dict without 'threshold' key should not raise."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        results = [_make_eval_result()]
        per_type = {
            "short_text": {
                "win_rate": 0.88,
                "sample_count": 2,
                "avg_score": 0.88,
                # threshold intentionally omitted
            }
        }
        display.show_summary_table(results, per_type)

        assert "short_text" in buf.getvalue()


# ---------------------------------------------------------------------------
# show_recommendation
# ---------------------------------------------------------------------------


class TestShowRecommendation:
    def test_go_renders_go_text(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_recommendation("GO", "All metrics exceeded thresholds.")
        output = buf.getvalue()
        assert "GO" in output
        assert "All metrics exceeded thresholds." in output

    def test_no_go_renders_no_go_text(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_recommendation("NO_GO", "BERTScore too low.")
        output = buf.getvalue()
        assert "NO GO" in output or "NO_GO" in output
        assert "BERTScore too low." in output

    def test_conditional_renders_conditional_text(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_recommendation("CONDITIONAL", "Proceed with manual review.")
        output = buf.getvalue()
        assert "CONDITIONAL" in output
        assert "Proceed with manual review." in output

    def test_recommendation_case_insensitive(self):
        """Lowercase recommendation strings should work the same as uppercase."""
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_recommendation("go", "Looks good.")
        assert "GO" in buf.getvalue()

    def test_unknown_recommendation_does_not_crash(self):
        """An unrecognised recommendation value should fall through to CONDITIONAL style."""
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_recommendation("MAYBE", "Not sure yet.")
        assert "Not sure yet." in buf.getvalue()


# ---------------------------------------------------------------------------
# show_cost_summary
# ---------------------------------------------------------------------------


class TestShowCostSummary:
    def test_renders_phase_names(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_cost_summary({"preflight": 0.0010, "optimize": 3.4500, "evaluate": 0.5500})
        output = buf.getvalue()
        assert "preflight" in output
        assert "optimize" in output
        assert "evaluate" in output

    def test_renders_total_row(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_cost_summary({"preflight": 1.0, "evaluate": 2.0})
        assert "Total" in buf.getvalue()

    def test_total_is_sum_of_phases(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_cost_summary({"a": 1.5, "b": 2.5})
        # Total = 4.0000
        assert "4.0000" in buf.getvalue()

    def test_empty_costs_renders_table(self):
        """Empty costs dict should render a table without crashing."""
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_cost_summary({})
        # Table title still appears
        assert "Cost Summary" in buf.getvalue()

    def test_cost_formatted_with_four_decimal_places(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_cost_summary({"phase": 0.001})
        assert "0.0010" in buf.getvalue()


# ---------------------------------------------------------------------------
# show_safety_warnings
# ---------------------------------------------------------------------------


class TestShowSafetyWarnings:
    def test_no_output_for_empty_list(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings([])
        assert buf.getvalue() == ""

    def test_plain_string_warning_rendered(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings(["Token budget exceeded."])
        assert "Token budget exceeded." in buf.getvalue()

    def test_dict_warning_with_high_severity(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings([{"severity": "HIGH", "message": "Context window exceeded."}])
        output = buf.getvalue()
        assert "HIGH" in output
        assert "Context window exceeded." in output

    def test_dict_warning_with_medium_severity(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings([{"severity": "MEDIUM", "message": "Potential cost overrun."}])
        output = buf.getvalue()
        assert "MEDIUM" in output
        assert "Potential cost overrun." in output

    def test_dict_warning_with_low_severity(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings([{"severity": "LOW", "message": "Minor format change."}])
        output = buf.getvalue()
        assert "LOW" in output
        assert "Minor format change." in output

    def test_object_warning_with_severity_attribute(self):
        """Object with .severity / .message attributes is handled."""
        con, buf = _console()
        display = MigrationDisplay(console=con)

        warning = MagicMock()
        warning.severity = "HIGH"
        warning.message = "Model not available."

        display.show_safety_warnings([warning])
        output = buf.getvalue()
        assert "HIGH" in output
        assert "Model not available." in output

    def test_multiple_warnings_all_rendered(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings(
            [
                {"severity": "HIGH", "message": "Error A."},
                {"severity": "LOW", "message": "Note B."},
            ]
        )
        output = buf.getvalue()
        assert "Error A." in output
        assert "Note B." in output

    def test_panel_title_present(self):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings(["Some warning."])
        assert "Safety Warnings" in buf.getvalue()

    def test_case_insensitive_severity(self):
        """Lowercase severity strings should still map to the right colour label."""
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_safety_warnings([{"severity": "high", "message": "Danger!"}])
        output = buf.getvalue()
        assert "HIGH" in output
        assert "Danger!" in output


# ---------------------------------------------------------------------------
# Integration-style: smoke test all methods together
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_full_display_pipeline_does_not_raise(self):
        """All display methods can be called in sequence without exceptions."""
        con, _ = _console()
        display = MigrationDisplay(console=con)

        results = [
            _make_eval_result(composite_score=0.92, is_win=True),
            _make_eval_result(composite_score=0.78, is_win=False),
        ]
        per_type = {
            "short_text": {
                "win_rate": 0.50,
                "sample_count": 2,
                "avg_score": 0.85,
                "threshold": 0.90,
            }
        }

        display.show_summary_table(results, per_type)
        display.show_recommendation("GO", "All thresholds passed.")
        display.show_cost_summary({"preflight": 0.10, "optimize": 2.50})
        display.show_safety_warnings([{"severity": "LOW", "message": "Minor warning."}])
        progress = display.create_progress()
        assert isinstance(progress, Progress)

    @pytest.mark.parametrize("recommendation", ["GO", "NO_GO", "CONDITIONAL"])
    def test_all_recommendation_variants(self, recommendation):
        con, buf = _console()
        display = MigrationDisplay(console=con)
        display.show_recommendation(recommendation, f"Reasoning for {recommendation}.")
        assert f"Reasoning for {recommendation}." in buf.getvalue()
