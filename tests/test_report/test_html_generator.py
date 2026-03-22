"""Tests for HTML report generator."""

from __future__ import annotations

from rosettastone.core.types import EvalResult, MigrationResult, PromptPair
from rosettastone.report.html_generator import generate_html_report


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
                    prompt="What is 2+2?",
                    response="4",
                    source_model="openai/gpt-4o",
                ),
                new_response="The answer is 4.",
                scores={"bertscore_f1": 0.95, "embedding_similarity": 0.91},
                composite_score=0.93,
                is_win=True,
            ),
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="Explain gravity briefly.",
                    response="Gravity is the force that attracts objects toward each other.",
                    source_model="openai/gpt-4o",
                ),
                new_response="Gravity pulls things together.",
                scores={"bertscore_f1": 0.82},
                composite_score=0.82,
                is_win=True,
            ),
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="Translate hello to French.",
                    response="Bonjour",
                    source_model="openai/gpt-4o",
                ),
                new_response="Salut",
                scores={"bertscore_f1": 0.65},
                composite_score=0.65,
                is_win=False,
            ),
        ],
        confidence_score=0.87,
        baseline_score=0.72,
        improvement=0.15,
        cost_usd=0.0312,
        duration_seconds=38.5,
        warnings=["Sample size below recommended minimum"],
    )
    defaults.update(overrides)
    return MigrationResult(**defaults)


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    def test_produces_html_file(self, tmp_path):
        """HTML report generates a .html file that exists on disk."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)

        assert output_path.exists()
        assert output_path.suffix == ".html"
        assert output_path.name == "migration_report.html"

    def test_html_is_valid_structure(self, tmp_path):
        """Generated HTML has basic structural elements."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "<style>" in content

    def test_contains_key_sections(self, tmp_path):
        """HTML report contains all key content sections."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        assert "Executive Summary" in content
        assert "Recommendation" in content
        assert "Key Metrics" in content
        assert "Score Distribution" in content
        assert "Per-Case Results" in content

    def test_contains_model_names(self, tmp_path):
        """HTML report contains source and target model names."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        assert "openai/gpt-4o" in content
        assert "anthropic/claude-sonnet-4" in content

    def test_contains_metrics(self, tmp_path):
        """HTML report contains formatted metric values."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        assert "87.0%" in content  # confidence_score
        assert "72.0%" in content  # baseline_score
        assert "0.0312" in content  # cost_usd
        assert "38.5" in content  # duration_seconds

    def test_contains_warnings(self, tmp_path):
        """HTML report includes warning messages."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        assert "Sample size below recommended minimum" in content

    def test_no_pii_in_output(self, tmp_path):
        """HTML report does not contain raw prompt or response content."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        # The template should not display raw prompt/response text (PII safety)
        assert "What is 2+2?" not in content
        assert "Explain gravity briefly." not in content
        assert "Translate hello to French." not in content
        assert "The answer is 4." not in content
        assert "Gravity pulls things together." not in content

    def test_empty_results(self, tmp_path):
        """HTML report handles empty validation results gracefully."""
        result = _make_result(
            validation_results=[],
            confidence_score=0.0,
            baseline_score=0.0,
            improvement=0.0,
            cost_usd=0.0,
            duration_seconds=0.0,
            warnings=[],
        )
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        assert output_path.exists()
        assert "<!DOCTYPE html>" in content
        assert "No validation results" in content

    def test_creates_output_directory(self, tmp_path):
        """HTML generator creates output directory if it does not exist."""
        result = _make_result()
        output_dir = tmp_path / "nested" / "deep" / "dir"

        output_path = generate_html_report(result, output_dir)
        assert output_dir.exists()
        assert output_path.exists()

    def test_chart_js_included(self, tmp_path):
        """HTML report includes Chart.js for interactive charts."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        assert "chart.js" in content.lower() or "Chart" in content

    def test_color_coding_applied(self, tmp_path):
        """HTML report applies color coding classes based on scores."""
        result = _make_result()
        output_path = generate_html_report(result, tmp_path)
        content = output_path.read_text()

        # Should have color classes for different score ranges
        assert "score-green" in content or "score-yellow" in content or "score-red" in content
