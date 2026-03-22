"""Tests for executive narrative prompt template and formatting utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rosettastone.report.executive_prompt import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    _format_per_type_block,
    _format_safety_block,
    _format_warnings_block,
    format_executive_prompt,
)


def _make_prompt_kwargs(**overrides):
    """Build default kwargs for format_executive_prompt, with optional overrides."""
    defaults = {
        "source_model": "openai/gpt-4o",
        "target_model": "anthropic/claude-sonnet-4",
        "recommendation": "GO",
        "confidence_score": 0.942,
        "baseline_score": 0.85,
        "improvement": 0.073,
        "cost_usd": 2.3412,
        "duration_seconds": 127.3,
        "total_test_cases": 156,
        "wins": 142,
        "per_type_scores": {
            "json": {
                "win_rate": 0.97,
                "sample_count": 82,
                "mean": 0.95,
                "median": 0.96,
                "p10": 0.88,
                "p50": 0.96,
                "p90": 0.99,
                "confidence_interval": (0.91, 0.99),
            },
        },
        "safety_warnings": [],
        "warnings": [],
    }
    defaults.update(overrides)
    return defaults


class TestFormatExecutivePrompt:
    """Tests for the top-level format_executive_prompt function."""

    def test_format_executive_prompt_returns_messages(self):
        """Verify it returns a list of dicts with role/content keys."""
        messages = format_executive_prompt(**_make_prompt_kwargs())

        assert isinstance(messages, list)
        assert len(messages) > 0
        for msg in messages:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("system", "user", "assistant")
            assert isinstance(msg["content"], str)

    def test_messages_structure(self):
        """Verify messages list has system + (2*n few-shot pairs) + user message."""
        messages = format_executive_prompt(**_make_prompt_kwargs())

        n_examples = len(FEW_SHOT_EXAMPLES)
        expected_length = 1 + (2 * n_examples) + 1  # system + pairs + user

        assert len(messages) == expected_length

        # First message is system
        assert messages[0]["role"] == "system"

        # Few-shot pairs alternate user/assistant
        for i in range(n_examples):
            user_idx = 1 + (2 * i)
            assistant_idx = 2 + (2 * i)
            assert messages[user_idx]["role"] == "user"
            assert messages[assistant_idx]["role"] == "assistant"

        # Last message is the actual user prompt
        assert messages[-1]["role"] == "user"
        assert "executive summary" in messages[-1]["content"].lower()


class TestSystemPrompt:
    """Tests for the SYSTEM_PROMPT constant."""

    def test_system_prompt_prohibits_content(self):
        """Verify SYSTEM_PROMPT contains the NEVER include raw prompt text instruction."""
        assert "NEVER include raw prompt text" in SYSTEM_PROMPT
        assert "response text" in SYSTEM_PROMPT


class TestFewShotExamples:
    """Tests for few-shot example quality."""

    def test_few_shot_examples_count(self):
        """Verify there are at least 3 examples covering GO, CONDITIONAL, NO_GO."""
        assert len(FEW_SHOT_EXAMPLES) >= 3

        scenarios = [ex["scenario"] for ex in FEW_SHOT_EXAMPLES]
        scenario_text = " ".join(scenarios).upper()

        assert "GO" in scenario_text
        assert "CONDITIONAL" in scenario_text
        assert "NO_GO" in scenario_text

    def test_few_shot_examples_no_pii(self):
        """Verify none of the examples contain things that look like actual prompts/responses.

        Examples should only reference metrics, not contain raw user prompts or model outputs.
        Checks for common prompt patterns like "You are a", "As an AI", quoted instructions, etc.
        """
        # Patterns that would suggest raw prompt/response content
        pii_patterns = [
            r"(?i)^you are a\b",  # system prompt openings
            r"(?i)^as an AI\b",
            r"(?i)\"(please|help me|write a|generate a|translate)",  # quoted instructions
            r"(?i)Dear\s+(Sir|Madam|Customer)",  # letter openings
        ]

        for example in FEW_SHOT_EXAMPLES:
            output = example["output"]
            for pattern in pii_patterns:
                matches = re.findall(pattern, output, re.MULTILINE)
                assert not matches, (
                    f"Few-shot example '{example['scenario']}' may contain raw content: "
                    f"matched pattern '{pattern}' with: {matches}"
                )

    def test_few_shot_examples_have_required_keys(self):
        """Each example has scenario, input_summary, and output."""
        for example in FEW_SHOT_EXAMPLES:
            assert "scenario" in example
            assert "input_summary" in example
            assert "output" in example
            assert len(example["output"]) > 100, "Example output seems too short"


class TestFormatPerTypeBlock:
    """Tests for _format_per_type_block."""

    def test_format_per_type_block_with_data(self):
        """Verify it formats per-type scores correctly."""
        scores = {
            "json": {
                "win_rate": 0.97,
                "sample_count": 82,
                "mean": 0.953,
                "median": 0.961,
                "p10": 0.882,
                "p50": 0.961,
                "p90": 0.993,
                "confidence_interval": (0.91, 0.99),
            },
            "classification": {
                "win_rate": 0.93,
                "sample_count": 48,
                "mean": 0.912,
                "median": 0.910,
                "p10": 0.801,
                "p50": 0.910,
                "p90": 0.980,
                "confidence_interval": [0.82, 0.98],
            },
        }
        result = _format_per_type_block(scores)

        # Check both types are present (sorted order: classification before json)
        assert "**classification**" in result
        assert "**json**" in result

        # Check key metrics appear
        assert "win_rate=97.0%" in result
        assert "samples=82" in result
        assert "CI(95%)=[0.910, 0.990]" in result
        assert "win_rate=93.0%" in result
        assert "samples=48" in result

    def test_format_per_type_block_empty(self):
        """Verify empty dict returns 'No per-type breakdown available.'."""
        result = _format_per_type_block({})
        assert result == "No per-type breakdown available."

    def test_format_per_type_block_with_dataclass(self):
        """Verify it handles dataclass stats objects."""

        @dataclass
        class TypeStats:
            win_rate: float
            sample_count: int
            mean: float
            median: float
            p10: float
            p50: float
            p90: float
            confidence_interval: tuple[float, float]

        scores = {
            "json": TypeStats(
                win_rate=0.95,
                sample_count=50,
                mean=0.94,
                median=0.95,
                p10=0.85,
                p50=0.95,
                p90=0.99,
                confidence_interval=(0.88, 0.98),
            ),
        }
        result = _format_per_type_block(scores)
        assert "**json**" in result
        assert "win_rate=95.0%" in result
        assert "samples=50" in result

    def test_format_per_type_block_missing_ci(self):
        """Handles missing confidence_interval gracefully."""
        scores = {
            "json": {
                "win_rate": 0.9,
                "sample_count": 10,
                "mean": 0.9,
                "median": 0.9,
                "p10": 0.8,
                "p50": 0.9,
                "p90": 0.95,
                # no confidence_interval key
            },
        }
        result = _format_per_type_block(scores)
        assert "CI(95%)=[0.000, 0.000]" in result


class TestFormatSafetyBlock:
    """Tests for _format_safety_block."""

    def test_format_safety_block_with_dict_warnings(self):
        """Verify safety formatting with dict warnings."""
        warnings = [
            {"severity": "HIGH", "message": "System prompt leakage detected"},
            {"severity": "MEDIUM", "message": "Inconsistent formatting in 8% of responses"},
        ]
        result = _format_safety_block(warnings)

        assert "[HIGH] System prompt leakage detected" in result
        assert "[MEDIUM] Inconsistent formatting in 8% of responses" in result

    def test_format_safety_block_with_object_warnings(self):
        """Verify safety formatting with object-style warnings."""

        class SafetyWarning:
            def __init__(self, severity: str, message: str):
                self.severity = severity
                self.message = message

        warnings = [SafetyWarning("HIGH", "Prompt injection vulnerability")]
        result = _format_safety_block(warnings)
        assert "[HIGH] Prompt injection vulnerability" in result

    def test_format_safety_block_with_string_warnings(self):
        """Verify safety formatting with plain string warnings."""
        warnings = ["Some unexpected warning format"]
        result = _format_safety_block(warnings)
        assert "[INFO] Some unexpected warning format" in result

    def test_format_safety_block_empty(self):
        """Verify returns 'No safety issues found.' for empty list."""
        result = _format_safety_block([])
        assert result == "No safety issues found."


class TestFormatWarningsBlock:
    """Tests for _format_warnings_block."""

    def test_format_warnings_block(self):
        """Verify pipeline warnings formatting."""
        warnings = [
            "Token budget exceeded for 3 test cases",
            "Rate limit encountered — retried 2 times",
        ]
        result = _format_warnings_block(warnings)
        assert "- Token budget exceeded for 3 test cases" in result
        assert "- Rate limit encountered" in result

    def test_format_warnings_block_empty(self):
        """Verify empty warnings return 'No pipeline warnings.'."""
        result = _format_warnings_block([])
        assert result == "No pipeline warnings."


class TestPromptContentIntegrity:
    """Integration tests verifying the full prompt has the right structure and values."""

    def test_user_prompt_contains_model_names(self):
        """Verify the formatted user prompt includes the source and target models."""
        messages = format_executive_prompt(**_make_prompt_kwargs())
        user_msg = messages[-1]["content"]

        assert "openai/gpt-4o" in user_msg
        assert "anthropic/claude-sonnet-4" in user_msg

    def test_user_prompt_contains_metrics(self):
        """Verify formatted prompt includes key numerical metrics."""
        messages = format_executive_prompt(**_make_prompt_kwargs())
        user_msg = messages[-1]["content"]

        assert "94.2%" in user_msg  # confidence
        assert "156" in user_msg  # total test cases
        assert "142" in user_msg  # wins
        assert "$2.3412" in user_msg  # cost
        assert "127.3s" in user_msg  # duration

    def test_user_prompt_prohibits_raw_content(self):
        """Verify the user prompt ends with the no-raw-content reminder."""
        messages = format_executive_prompt(**_make_prompt_kwargs())
        user_msg = messages[-1]["content"]

        assert "no raw content" in user_msg.lower()

    def test_zero_test_cases_no_division_error(self):
        """Verify win_rate calculation handles zero test cases gracefully."""
        messages = format_executive_prompt(**_make_prompt_kwargs(total_test_cases=0, wins=0))
        user_msg = messages[-1]["content"]
        assert "0%" in user_msg or "0 (0%)" in user_msg
