"""Tier 1 model certification tests.

Tests real API migrations between certified model pairs:
  - openai/gpt-4o → anthropic/claude-sonnet-4
  - anthropic/claude-sonnet-4 → openai/gpt-4o (reverse)
  - openai/gpt-4o → anthropic/claude-haiku-4-5

Skip automatically if required API keys are absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


def _openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _anthropic_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _both_apis_available() -> bool:
    return _openai_available() and _anthropic_available()


skip_if_no_openai = pytest.mark.skipif(
    not _openai_available(),
    reason="OPENAI_API_KEY not set",
)

skip_if_no_apis = pytest.mark.skipif(
    not _both_apis_available(),
    reason="OPENAI_API_KEY and/or ANTHROPIC_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Canonical 20-pair dataset
# ---------------------------------------------------------------------------

CANONICAL_PAIRS = [
    # JSON output
    {"prompt": "Return JSON: {\"status\": \"ok\", \"code\": 200}", "response": "{\"status\": \"ok\", \"code\": 200}"},
    {"prompt": "Return JSON: {\"items\": [1, 2, 3], \"total\": 3}", "response": "{\"items\": [1, 2, 3], \"total\": 3}"},
    {"prompt": "Return JSON: {\"name\": \"Alice\", \"age\": 30}", "response": "{\"name\": \"Alice\", \"age\": 30}"},
    {"prompt": "Return JSON: {\"error\": null, \"success\": true}", "response": "{\"error\": null, \"success\": true}"},
    {"prompt": "Return JSON: {\"count\": 0, \"results\": []}", "response": "{\"count\": 0, \"results\": []}"},
    # Text/QA
    {"prompt": "What is the capital of France?", "response": "Paris"},
    {"prompt": "What does HTTP stand for?", "response": "HyperText Transfer Protocol"},
    {"prompt": "Name the first three elements on the periodic table.", "response": "Hydrogen, Helium, and Lithium"},
    {"prompt": "What programming language was created by Guido van Rossum?", "response": "Python"},
    {"prompt": "What is the largest planet in our solar system?", "response": "Jupiter"},
    # Classification/boolean
    {"prompt": "Is Python an interpreted language? Answer yes or no.", "response": "Yes"},
    {"prompt": "Is the number 17 prime? Answer yes or no.", "response": "Yes"},
    {"prompt": "Is water a compound? Answer yes or no.", "response": "Yes"},
    {"prompt": "Is gold a metal? Answer yes or no.", "response": "Yes"},
    {"prompt": "Does Python support multiple inheritance? Answer yes or no.", "response": "Yes"},
    # Reasoning
    {"prompt": "What is 15 multiplied by 8?", "response": "120"},
    {"prompt": "If a train travels 60 mph for 2 hours, how far does it go?", "response": "120 miles"},
    {"prompt": "What is the square root of 144?", "response": "12"},
    {"prompt": "Convert 100 Celsius to Fahrenheit.", "response": "212 degrees Fahrenheit"},
    {"prompt": "How many days are in a leap year?", "response": "366"},
]


@pytest.fixture(scope="module")
def canonical_data_file(tmp_path_factory) -> Path:
    """Write CANONICAL_PAIRS to a temp JSONL file."""
    data_dir = tmp_path_factory.mktemp("tier1_data")
    path = data_dir / "canonical.jsonl"
    with path.open("w") as f:
        for pair in CANONICAL_PAIRS:
            f.write(json.dumps(pair) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tier 1 certification tests
# ---------------------------------------------------------------------------


@skip_if_no_apis
class TestGPT4oToClaudeSonnet:
    """openai/gpt-4o → anthropic/claude-sonnet-4 migration."""

    @pytest.fixture(scope="class")
    def result(self, canonical_data_file):
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=canonical_data_file,
            gepa_auto="light",
        )
        return Migrator(config).run()

    def test_migration_completes(self, result) -> None:
        assert result is not None

    def test_recommendation_is_valid(self, result) -> None:
        assert result.recommendation in ("GO", "NO_GO", "CONDITIONAL")

    def test_confidence_score_in_range(self, result) -> None:
        assert 0.0 <= result.confidence_score <= 1.0

    def test_baseline_score_in_range(self, result) -> None:
        assert 0.0 <= result.baseline_score <= 1.0

    def test_has_per_type_scores(self, result) -> None:
        assert result.per_type_scores
        assert len(result.per_type_scores) >= 1

    def test_report_generated(self, result) -> None:
        assert result.report_path
        assert Path(result.report_path).exists()


@skip_if_no_apis
class TestClaudeSonnetToGPT4o:
    """anthropic/claude-sonnet-4 → openai/gpt-4o (reverse migration)."""

    @pytest.fixture(scope="class")
    def result(self, canonical_data_file):
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        config = MigrationConfig(
            source_model="anthropic/claude-sonnet-4",
            target_model="openai/gpt-4o",
            data_path=canonical_data_file,
            gepa_auto="light",
        )
        return Migrator(config).run()

    def test_migration_completes(self, result) -> None:
        assert result is not None

    def test_recommendation_is_valid(self, result) -> None:
        assert result.recommendation in ("GO", "NO_GO", "CONDITIONAL")

    def test_confidence_score_in_range(self, result) -> None:
        assert 0.0 <= result.confidence_score <= 1.0


@skip_if_no_apis
class TestGPT4oToClaudeHaiku:
    """openai/gpt-4o → anthropic/claude-haiku-4-5 migration."""

    @pytest.fixture(scope="class")
    def result(self, canonical_data_file):
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-haiku-4-5-20251001",
            data_path=canonical_data_file,
            gepa_auto="light",
        )
        return Migrator(config).run()

    def test_migration_completes(self, result) -> None:
        assert result is not None

    def test_recommendation_is_valid(self, result) -> None:
        assert result.recommendation in ("GO", "NO_GO", "CONDITIONAL")

    def test_confidence_score_in_range(self, result) -> None:
        assert 0.0 <= result.confidence_score <= 1.0
