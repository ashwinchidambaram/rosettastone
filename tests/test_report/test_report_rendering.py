"""Phase A observability rendering tests for the markdown report template."""

from __future__ import annotations

from rosettastone.core.types import EvalResult, MigrationResult, PromptPair
from rosettastone.report.markdown import generate_markdown_report


def _base_result(**kwargs) -> MigrationResult:
    """Return a MigrationResult with sensible defaults."""
    defaults = dict(
        config={"source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4"},
        optimized_prompt="You are a helpful assistant.",
        baseline_results=[],
        validation_results=[
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="test", response="expected", source_model="openai/gpt-4o"
                ),
                new_response="actual",
                scores={"bertscore_f1": 0.85},
                composite_score=0.85,
                is_win=True,
                details={"output_type": "short_text"},
            ),
        ],
        confidence_score=1.0,
        baseline_score=0.7,
        improvement=0.3,
        cost_usd=5.0,
        duration_seconds=60.0,
        warnings=[],
    )
    defaults.update(kwargs)
    return MigrationResult(**defaults)


# ---------------------------------------------------------------------------
# Feature 1: Pipeline Timing section
# ---------------------------------------------------------------------------


def test_pipeline_timing_section_renders_when_stage_timing_non_empty(tmp_path):
    """Pipeline Timing section appears when stage_timing has entries."""
    result = _base_result(
        stage_timing={"ingest": 2.5, "baseline_eval": 10.3, "optimize": 45.1},
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()

    assert "## Pipeline Timing" in content
    assert "ingest" in content
    assert "baseline_eval" in content
    assert "optimize" in content
    # Duration values formatted with 1 decimal
    assert "2.5s" in content
    assert "10.3s" in content
    assert "45.1s" in content


def test_pipeline_timing_section_absent_when_empty(tmp_path):
    """Pipeline Timing section is NOT rendered when stage_timing is empty."""
    result = _base_result(stage_timing={})
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "## Pipeline Timing" not in content


def test_pipeline_timing_absent_by_default(tmp_path):
    """Pipeline Timing section absent when stage_timing field not provided."""
    result = _base_result()  # stage_timing defaults to {}
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "## Pipeline Timing" not in content


# ---------------------------------------------------------------------------
# Feature 3: Per-Type CI column
# ---------------------------------------------------------------------------


def test_ci_column_appears_with_valid_ci_values(tmp_path):
    """95% CI column renders percentage range when CI bounds are non-zero."""
    result = _base_result(
        recommendation="GO",
        recommendation_reasoning="ok",
        per_type_scores={
            "short_text": {
                "win_rate": 0.90,
                "mean": 0.88,
                "median": 0.90,
                "p10": 0.75,
                "p50": 0.90,
                "p90": 0.98,
                "min_score": 0.70,
                "max_score": 1.0,
                "sample_count": 35,
                "confidence_interval": (0.62, 0.81),
            }
        },
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "95% CI" in content
    assert "62%" in content
    assert "81%" in content


def test_ci_shows_dash_when_both_bounds_are_zero(tmp_path):
    """95% CI column shows '—' when CI is (0.0, 0.0) (no data)."""
    result = _base_result(
        recommendation="CONDITIONAL",
        recommendation_reasoning="insufficient data",
        per_type_scores={
            "json": {
                "win_rate": 0.60,
                "mean": 0.65,
                "median": 0.62,
                "p10": 0.50,
                "p50": 0.62,
                "p90": 0.80,
                "min_score": 0.40,
                "max_score": 0.90,
                "sample_count": 5,
                "confidence_interval": (0.0, 0.0),
            }
        },
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    # The CI column exists (header) but shows '—' for this row
    assert "95% CI" in content
    assert "| — |" in content or "|—|" in content or " — |" in content


# ---------------------------------------------------------------------------
# Feature 5: Evaluation Reliability section
# ---------------------------------------------------------------------------


def test_eval_reliability_section_renders_when_eval_runs_gt_1(tmp_path):
    """Evaluation Reliability section appears when eval_runs > 1."""
    result = _base_result(eval_runs=3, non_deterministic_count=0)
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "## Evaluation Reliability" in content
    assert "Evaluation runs: 3" in content


def test_eval_reliability_section_renders_when_non_deterministic_gt_0(tmp_path):
    """Evaluation Reliability section appears when non_deterministic_count > 0."""
    result = _base_result(eval_runs=1, non_deterministic_count=2)
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "## Evaluation Reliability" in content
    assert "Non-deterministic pairs: 2" in content


def test_eval_reliability_section_absent_when_baseline(tmp_path):
    """Evaluation Reliability section absent when eval_runs==1 and non_deterministic_count==0."""
    result = _base_result(eval_runs=1, non_deterministic_count=0)
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    # The entire section (header + body) should be absent because the guard
    # {% if eval_runs > 1 or non_deterministic_count > 0 %} wraps both.
    assert "## Evaluation Reliability" not in content
    assert "Evaluation runs:" not in content
    assert "Non-deterministic pairs:" not in content
