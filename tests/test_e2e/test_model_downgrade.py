"""E2E tests for model downgrade scenarios (C1–C2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.core.migrator import Migrator
from rosettastone.testing.redis_populator import RedisPopulator
from rosettastone.testing.scenarios import DOWNGRADE_SCENARIOS, ScenarioConfig
from rosettastone.testing.synth_data import SyntheticDataGenerator

pytestmark = pytest.mark.e2e


@pytest.fixture
def run_scenario(redis_url, clean_redis, generated_data_cache):
    """Helper fixture: generate data, populate Redis, run migration, return result."""

    def _run(scenario: ScenarioConfig):
        model = scenario.source_model
        if model not in generated_data_cache:
            gen = SyntheticDataGenerator(source_model=model)
            generated_data_cache[model] = gen.generate()
        data = generated_data_cache[model]

        populator = RedisPopulator(redis_url)
        populator.populate(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MigrationConfig(
                source_model=scenario.source_model,
                target_model=scenario.target_model,
                data_path=Path(tmpdir) / "placeholder.jsonl",
                output_dir=Path(tmpdir) / "output",
                redis_url=redis_url,
                reflection_model=scenario.reflection_model,
                judge_model=scenario.judge_model,
                gepa_auto="light",
                skip_preflight=True,
            )
            return Migrator(config).run()

    return _run


@pytest.mark.parametrize(
    "scenario",
    DOWNGRADE_SCENARIOS,
    ids=[s.name for s in DOWNGRADE_SCENARIOS],
)
class TestModelDowngrade:
    def test_downgrade_completes(self, scenario, run_scenario):
        """Downgrade migration completes without error (no min confidence required)."""
        result = run_scenario(scenario)

        # Layer 1: Pipeline completion
        assert result.duration_seconds > 0
        assert result.optimized_prompt
        assert len(result.validation_results) > 0

        # Layer 2: Structural correctness
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.recommendation in {"GO", "CONDITIONAL", "NO_GO"}

    def test_downgrade_captures_regression(self, scenario, run_scenario):
        """Downgrade per_type_scores should exist and show which types regressed."""
        result = run_scenario(scenario)

        assert isinstance(result.per_type_scores, dict)
        assert len(result.per_type_scores) > 0, "per_type_scores should have entries"

        # Recommendation should be populated (any value is valid for downgrades)
        assert result.recommendation is not None
        assert result.recommendation in scenario.expected_recommendations
