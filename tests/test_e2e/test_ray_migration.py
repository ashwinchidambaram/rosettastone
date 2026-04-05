"""Ray Serve E2E migration tests.

Tests GEPA optimization end-to-end against the SV Ray cluster LLM endpoint.
The cluster serves GPT-OSS-20B via an OpenAI-compatible API.

Requires SSH tunnel to be active:
  ssh -N -L 18123:localhost:18123 <host>

Skip automatically if the Ray endpoint is not reachable.

Environment variables (with defaults matching ray-cluster-worker/.env):
  RAY_CHAT_API_BASE   http://localhost:18123/v1
  RAY_CHAT_API_KEY    sv-openai-api-key
  RAY_CHAT_MODEL      openai/gpt-oss-20b
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.ray]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_RAY_API_BASE = os.environ.get("RAY_CHAT_API_BASE", "http://localhost:18123/v1")
_RAY_API_KEY = os.environ.get("RAY_CHAT_API_KEY", "sv-openai-api-key")
_RAY_MODEL = os.environ.get("RAY_CHAT_DEFAULT_MODEL", "openai/gpt-oss-20b")


def _ray_available() -> bool:
    """Check if the Ray chat endpoint is reachable."""
    try:
        import httpx

        resp = httpx.get(
            f"{_RAY_API_BASE}/models",
            timeout=3.0,
            headers={"Authorization": f"Bearer {_RAY_API_KEY}"},
        )
        return resp.status_code in (200, 401, 404)
    except Exception:
        return False


skip_if_no_ray = pytest.mark.skipif(
    not _ray_available(),
    reason=f"Ray endpoint not reachable at {_RAY_API_BASE} (SSH tunnel active?)",
)


# ---------------------------------------------------------------------------
# Sample data (short — keeps API calls minimal)
# ---------------------------------------------------------------------------

RAY_SAMPLE_PAIRS = [
    {"prompt": "What is 7 * 8?", "response": "56"},
    {"prompt": "What is the capital of Germany?", "response": "Berlin"},
    {"prompt": "Is 42 an even number? Answer yes or no.", "response": "Yes"},
    {"prompt": "What does API stand for?", "response": "Application Programming Interface"},
    {"prompt": 'Return JSON: {"x": 1, "y": 2}', "response": '{"x": 1, "y": 2}'},
    {"prompt": 'Return JSON: {"ok": true}', "response": '{"ok": true}'},
    {"prompt": "What color is the sky on a clear day?", "response": "Blue"},
    {"prompt": "How many sides does a hexagon have?", "response": "6"},
    {"prompt": "What is the square root of 81?", "response": "9"},
    {"prompt": "Name the closest star to Earth.", "response": "The Sun"},
]


@pytest.fixture(scope="module")
def ray_data_file(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp("ray_data")
    path = data_dir / "ray_sample.jsonl"
    with path.open("w") as f:
        for pair in RAY_SAMPLE_PAIRS:
            f.write(json.dumps(pair) + "\n")
    return path


# ---------------------------------------------------------------------------
# Ray E2E tests
# ---------------------------------------------------------------------------


@skip_if_no_ray
class TestRayMigration:
    """Full GEPA optimization against the SV Ray GPT-OSS-20B endpoint.

    Migrates from gpt-oss-20b → gpt-oss-20b (same model) to verify the
    pipeline works end-to-end with Ray-hosted models. Confidence scores
    should be high since source and target are identical.
    """

    @pytest.fixture(scope="class")
    def result(self, ray_data_file):
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        config = MigrationConfig(
            source_model=_RAY_MODEL,
            target_model=_RAY_MODEL,
            data_path=ray_data_file,
            gepa_auto="light",
            lm_extra_kwargs={"api_base": _RAY_API_BASE, "api_key": _RAY_API_KEY},
        )
        return Migrator(config).run()

    def test_migration_completes(self, result) -> None:
        assert result is not None

    def test_recommendation_is_valid(self, result) -> None:
        assert result.recommendation in ("GO", "NO_GO", "CONDITIONAL")

    def test_confidence_score_in_range(self, result) -> None:
        assert 0.0 <= result.confidence_score <= 1.0

    def test_has_per_type_scores(self, result) -> None:
        assert result.per_type_scores
        assert len(result.per_type_scores) >= 1

    def test_report_generated(self, result) -> None:
        assert result.report_path
        assert Path(result.report_path).exists()

    def test_same_model_confidence_high(self, result) -> None:
        """Same-model migration should yield high confidence (>=0.7)."""
        assert result.confidence_score >= 0.7


@skip_if_no_ray
class TestRayDryRun:
    """Dry-run against Ray endpoint — fast smoke test, no GEPA optimization."""

    def test_dry_run_completes(self, ray_data_file) -> None:
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        config = MigrationConfig(
            source_model=_RAY_MODEL,
            target_model=_RAY_MODEL,
            data_path=ray_data_file,
            dry_run=True,
            lm_extra_kwargs={"api_base": _RAY_API_BASE, "api_key": _RAY_API_KEY},
        )
        result = Migrator(config).run()
        assert result is not None
        assert result.recommendation is not None

    def test_preflight_passes_for_ray_model(self, ray_data_file) -> None:
        """Preflight checks pass for the Ray-served model."""
        from rosettastone.config import MigrationConfig
        from rosettastone.preflight.checks import run_all_checks

        config = MigrationConfig(
            source_model=_RAY_MODEL,
            target_model=_RAY_MODEL,
            data_path=ray_data_file,
            lm_extra_kwargs={"api_base": _RAY_API_BASE, "api_key": _RAY_API_KEY},
        )
        report = run_all_checks(config)
        # No hard blockers expected for a reachable Ray model
        assert not any(b for b in report.blockers if "not available" in b.lower())
