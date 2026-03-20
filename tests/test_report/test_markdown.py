"""Tests for report generation subsystem (markdown template, file creation, edge cases)."""

from __future__ import annotations

from rosettastone.core.types import EvalResult, MigrationResult, PromptPair
from rosettastone.report.markdown import generate_markdown_report

# ── shared fixtures ───────────────────────────────────────────────────────────


def _base_result(**kwargs) -> MigrationResult:
    """Return a MigrationResult with sensible defaults, optionally overriding fields."""
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
        warnings=["Test warning"],
    )
    defaults.update(kwargs)
    return MigrationResult(**defaults)


# ── original tests (preserved, updated fixtures) ─────────────────────────────


def test_report_generates_files(tmp_path):
    """Report creates both markdown and prompt files."""
    result = _base_result()
    output_path = generate_markdown_report(result, tmp_path)
    assert output_path.exists()
    assert (tmp_path / "optimized_prompt.txt").exists()
    content = output_path.read_text()
    assert "openai/gpt-4o" in content
    assert "100.0%" in content  # confidence score formatted as percentage
    assert "Test warning" in content


def test_report_creates_output_dir(tmp_path):
    """Report creates output_dir if it doesn't exist."""
    output_dir = tmp_path / "nested" / "output"
    result = MigrationResult(
        config={},
        optimized_prompt="test",
        baseline_results=[],
        validation_results=[],
        confidence_score=0.0,
        baseline_score=0.0,
        improvement=0.0,
        cost_usd=0.0,
        duration_seconds=0.0,
        warnings=[],
    )
    output_path = generate_markdown_report(result, output_dir)
    assert output_dir.exists()
    assert output_path.exists()


def test_report_with_empty_results(tmp_path):
    """Template handles empty validation_results without error."""
    result = MigrationResult(
        config={},
        optimized_prompt="",
        baseline_results=[],
        validation_results=[],
        confidence_score=0.0,
        baseline_score=0.0,
        improvement=0.0,
        cost_usd=0.0,
        duration_seconds=0.0,
        warnings=["DRY RUN"],
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "DRY RUN" in content


def test_optimized_prompt_written(tmp_path):
    """optimized_prompt.txt contains the prompt text."""
    result = MigrationResult(
        config={},
        optimized_prompt="Custom system prompt here",
        baseline_results=[],
        validation_results=[],
        confidence_score=0.0,
        baseline_score=0.0,
        improvement=0.0,
        cost_usd=0.0,
        duration_seconds=0.0,
        warnings=[],
    )
    generate_markdown_report(result, tmp_path)
    prompt_content = (tmp_path / "optimized_prompt.txt").read_text()
    assert prompt_content == "Custom system prompt here"


# ── new recommendation-type rendering tests ───────────────────────────────────


def test_report_renders_go_recommendation(tmp_path):
    """Template renders without error when recommendation=GO."""
    result = _base_result(
        recommendation="GO",
        recommendation_reasoning="All types meet thresholds.",
        per_type_scores={
            "short_text": {
                "win_rate": 0.95,
                "mean": 0.92,
                "median": 0.93,
                "p10": 0.80,
                "p50": 0.93,
                "p90": 0.99,
                "min_score": 0.75,
                "max_score": 1.0,
                "sample_count": 20,
                "confidence_interval": (0.85, 1.0),
            }
        },
        safety_warnings=[],
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "GO" in content
    assert "All types meet thresholds." in content
    assert "short_text" in content


def test_report_renders_no_go_recommendation(tmp_path):
    """Template renders without error when recommendation=NO_GO."""
    result = _base_result(
        recommendation="NO_GO",
        recommendation_reasoning="HIGH severity safety finding: PII leak detected.",
        per_type_scores={},
        safety_warnings=[{"severity": "HIGH", "message": "PII leak detected"}],
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "NO_GO" in content
    assert "PII leak detected" in content
    assert "HIGH" in content


def test_report_renders_conditional_recommendation(tmp_path):
    """Template renders without error when recommendation=CONDITIONAL."""
    result = _base_result(
        recommendation="CONDITIONAL",
        recommendation_reasoning="json win rate 60.0% < threshold 95.0%.",
        per_type_scores={
            "json": {
                "win_rate": 0.60,
                "mean": 0.72,
                "median": 0.70,
                "p10": 0.55,
                "p50": 0.70,
                "p90": 0.88,
                "min_score": 0.40,
                "max_score": 0.95,
                "sample_count": 10,
                "confidence_interval": (0.29, 0.87),
            }
        },
        safety_warnings=[],
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "CONDITIONAL" in content
    assert "json" in content


def test_report_renders_with_no_recommendation(tmp_path):
    """Template renders gracefully when recommendation fields are absent (old-style result)."""
    result = _base_result()  # no recommendation fields set
    output_path = generate_markdown_report(result, tmp_path)
    assert output_path.exists()
    content = output_path.read_text()
    # Should not crash; basic content should be present.
    assert "Migration Report" in content


def test_report_per_type_table_present(tmp_path):
    """Per-Output-Type Breakdown table appears when per_type_scores is populated."""
    result = _base_result(
        recommendation="GO",
        recommendation_reasoning="ok",
        per_type_scores={
            "long_text": {
                "win_rate": 0.80,
                "mean": 0.82,
                "median": 0.81,
                "p10": 0.70,
                "p50": 0.81,
                "p90": 0.95,
                "min_score": 0.60,
                "max_score": 1.0,
                "sample_count": 12,
                "confidence_interval": (0.53, 0.96),
            }
        },
        safety_warnings=[],
    )
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "Per-Output-Type Breakdown" in content
    assert "long_text" in content
    assert "80.0%" in content


def test_report_safety_no_issues_when_empty(tmp_path):
    """Safety Findings section shows 'No issues found' when list is empty."""
    result = _base_result(safety_warnings=[])
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "No issues found" in content


def test_report_safety_warnings_shown(tmp_path):
    """Safety Findings section lists plain-string warnings."""
    result = _base_result(safety_warnings=["Rate limit exceeded during eval"])
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "Rate limit exceeded" in content


def test_report_cost_summary_present(tmp_path):
    """Cost Summary section is rendered."""
    result = _base_result(cost_usd=1.2345)
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "Cost Summary" in content
    assert "1.2345" in content


def test_report_worst_regressions_shown(tmp_path):
    """Worst Regressions shows up to 5 lowest-scoring results."""
    results = [
        EvalResult(
            prompt_pair=PromptPair(prompt=f"q{i}", response=f"expected {i}", source_model="m"),
            new_response=f"got {i}",
            scores={"s": float(i) / 10},
            composite_score=float(i) / 10,
            is_win=float(i) / 10 >= 0.8,
            details={"output_type": "short_text"},
        )
        for i in range(1, 8)
    ]
    result = _base_result(validation_results=results)
    output_path = generate_markdown_report(result, tmp_path)
    content = output_path.read_text()
    assert "Worst Regressions" in content
    # Bottom score 0.1 should appear
    assert "0.1000" in content
