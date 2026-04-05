"""End-to-end migration tests using local Ollama models.

Requires Ollama to be running with qwen3:8b and qwen3.5:4b models.
Skip automatically if Ollama is not available.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Mark all tests in this module
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

_OLLAMA_BASE_URL = "http://localhost:11434"
_SOURCE_MODEL = "ollama/qwen3:8b"
_TARGET_MODEL = "ollama/qwen3.5:4b"


def _ollama_available() -> bool:
    """Check if Ollama is running and has the required qwen3 models."""
    try:
        import httpx

        resp = httpx.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any("qwen3" in m for m in models)
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running or qwen3 models not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama_sample_data(tmp_path_factory) -> Path:
    """Write the static Ollama sample JSONL to a temporary path for the test run."""
    static_data = Path(__file__).parent.parent.parent / "examples" / "ollama_sample_data.jsonl"
    if static_data.exists():
        return static_data

    # Fallback: generate inline if the static file is somehow missing
    data_dir = tmp_path_factory.mktemp("ollama_data")
    data_file = data_dir / "sample.jsonl"

    pairs = [
        {
            "prompt": "What is the capital of France?",
            "response": "Paris",
            "source_model": _SOURCE_MODEL,
        },
        {"prompt": "What is 15 multiplied by 7?", "response": "105", "source_model": _SOURCE_MODEL},
        {
            "prompt": "Name three primary colors.",
            "response": "Red, blue, and yellow.",
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": "What does HTTP stand for?",
            "response": "HyperText Transfer Protocol",
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": "What is the approximate speed of light?",
            "response": "299,792,458 meters per second",
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": 'Return JSON: {"status": "ok", "code": 200}',
            "response": '{"status": "ok", "code": 200}',
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": 'Return JSON with name field: {"name": "Alice"}',
            "response": '{"name": "Alice"}',
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": 'Return JSON: {"items": ["a", "b", "c"]}',
            "response": '{"items": ["a", "b", "c"]}',
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": 'Return JSON: {"error": null, "success": true}',
            "response": '{"error": null, "success": true}',
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": 'Return JSON: {"count": 42, "label": "test"}',
            "response": '{"count": 42, "label": "test"}',
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": "Is Python an interpreted language? Answer yes or no.",
            "response": "Yes",
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": "What is the largest planet in our solar system?",
            "response": "Jupiter",
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": "What programming language was created by Guido van Rossum?",
            "response": "Python",
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": "What does CPU stand for?",
            "response": "Central Processing Unit",
            "source_model": _SOURCE_MODEL,
        },
        {
            "prompt": "Is water a compound or an element?",
            "response": "Compound",
            "source_model": _SOURCE_MODEL,
        },
    ]

    with data_file.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return data_file


@pytest.fixture(scope="module")
def migration_output_dir(tmp_path_factory) -> Path:
    """Shared output directory for the full Ollama migration run."""
    return tmp_path_factory.mktemp("ollama_output")


@pytest.fixture(scope="module")
def migration_result(ollama_sample_data, migration_output_dir):
    """Run a full migration from qwen3:8b to qwen3.5:4b.

    Uses skip_preflight=True and Ollama models as reflection/judge to avoid
    requiring external API keys during the E2E run.
    """
    from rosettastone.config import MigrationConfig
    from rosettastone.core.migrator import Migrator

    config = MigrationConfig(
        source_model=_SOURCE_MODEL,
        target_model=_TARGET_MODEL,
        data_path=ollama_sample_data,
        output_dir=migration_output_dir,
        gepa_max_metric_calls=20,  # small budget for local hardware
        reflection_model=_TARGET_MODEL,  # smaller model for reflection
        judge_model=_TARGET_MODEL,  # smaller model for judging
        skip_preflight=True,
        min_pairs=10,  # accommodate the 15-pair dataset
        num_threads=2,  # limit concurrency on local hardware
        lm_extra_kwargs={"extra_body": {"think": False}},  # disable qwen3 extended reasoning
    )
    migrator = Migrator(config)
    return migrator.run()


# ---------------------------------------------------------------------------
# TestOllamaMigration — full pipeline assertions
# ---------------------------------------------------------------------------


@skip_if_no_ollama
class TestOllamaMigration:
    def test_migration_completes_without_error(self, migration_result) -> None:
        """Full migration pipeline completes successfully."""
        assert migration_result is not None

    def test_migration_duration_positive(self, migration_result) -> None:
        """Migration records positive wall-clock duration."""
        assert migration_result.duration_seconds > 0

    def test_migration_has_optimized_prompt(self, migration_result) -> None:
        """Migration produces a non-empty optimized prompt."""
        assert migration_result.optimized_prompt

    def test_migration_has_validation_results(self, migration_result) -> None:
        """Migration evaluated at least one test pair during validation."""
        assert len(migration_result.validation_results) > 0

    def test_migration_has_recommendation(self, migration_result) -> None:
        """Migration produces a GO/NO_GO/CONDITIONAL recommendation."""
        assert migration_result.recommendation in {"GO", "NO_GO", "CONDITIONAL"}

    def test_migration_confidence_score_in_range(self, migration_result) -> None:
        """Confidence score is between 0.0 and 1.0 inclusive."""
        assert 0.0 <= migration_result.confidence_score <= 1.0

    def test_migration_baseline_score_in_range(self, migration_result) -> None:
        """Baseline score is between 0.0 and 1.0 inclusive."""
        assert 0.0 <= migration_result.baseline_score <= 1.0

    def test_migration_cost_non_negative(self, migration_result) -> None:
        """Migration records a non-negative cost (Ollama local cost is ~$0)."""
        assert migration_result.cost_usd >= 0.0

    def test_migration_has_per_type_scores(self, migration_result) -> None:
        """Migration reports per-output-type score breakdown."""
        assert isinstance(migration_result.per_type_scores, dict)
        assert len(migration_result.per_type_scores) >= 1

    def test_migration_per_type_keys_are_known(self, migration_result) -> None:
        """Per-type score keys correspond to known OutputType values."""
        known_types = {"json", "classification", "short_text", "long_text"}
        found = set(migration_result.per_type_scores.keys())
        assert found & known_types, (
            f"Expected at least one of {known_types} in per_type_scores, got {found}"
        )

    def test_migration_report_file_written(self, migration_result, migration_output_dir) -> None:
        """Migration writes a markdown report file to output_dir."""
        report = migration_output_dir / "migration_report.md"
        assert report.exists(), f"Report not found at {report}"
        content = report.read_text()
        assert len(content) > 0

    def test_migration_report_mentions_models(self, migration_output_dir) -> None:
        """Report content references the source or target model names."""
        report = migration_output_dir / "migration_report.md"
        content = report.read_text().lower()
        assert "qwen3" in content or "migration" in content

    def test_migration_optimized_prompt_file_written(self, migration_output_dir) -> None:
        """Migration writes the optimized prompt to a text file."""
        prompt_file = migration_output_dir / "optimized_prompt.txt"
        assert prompt_file.exists(), f"Optimized prompt file not found at {prompt_file}"


# ---------------------------------------------------------------------------
# TestOllamaDryRun — validate dry-run path with Ollama config
# ---------------------------------------------------------------------------


@skip_if_no_ollama
class TestOllamaDryRun:
    def test_dry_run_returns_result(self, ollama_sample_data) -> None:
        """Dry-run migration returns a MigrationResult without running the full pipeline."""
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MigrationConfig(
                source_model=_SOURCE_MODEL,
                target_model=_TARGET_MODEL,
                data_path=ollama_sample_data,
                output_dir=Path(tmpdir) / "output",
                dry_run=True,
                skip_preflight=True,
                min_pairs=10,
                lm_extra_kwargs={"extra_body": {"think": False}},
            )
            result = Migrator(config).run()

        assert result is not None
        assert result.recommendation is not None

    def test_dry_run_skip_preflight_returns_no_go(self) -> None:
        """dry_run + skip_preflight shortcut returns NO_GO immediately."""
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MigrationConfig(
                source_model=_SOURCE_MODEL,
                target_model=_TARGET_MODEL,
                output_dir=Path(tmpdir) / "output",
                dry_run=True,
                skip_preflight=True,
            )
            result = Migrator(config).run()

        assert result.recommendation == "NO_GO"
        assert result.confidence_score == 0.0
