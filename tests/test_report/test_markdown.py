"""Tests for report generation subsystem (markdown template, file creation, edge cases)."""

from rosettastone.core.types import EvalResult, MigrationResult, PromptPair
from rosettastone.report.markdown import generate_markdown_report


def test_report_generates_files(tmp_path):
    """Report creates both markdown and prompt files."""
    result = MigrationResult(
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
            ),
        ],
        confidence_score=1.0,
        baseline_score=0.7,
        improvement=0.3,
        cost_usd=5.0,
        duration_seconds=60.0,
        warnings=["Test warning"],
    )
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
