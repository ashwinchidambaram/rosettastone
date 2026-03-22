"""E2E tests for cross-provider migration scenarios (A1–A3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.core.migrator import Migrator
from rosettastone.testing.redis_populator import RedisPopulator
from rosettastone.testing.scenarios import CROSS_PROVIDER_SCENARIOS, ScenarioConfig
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
    CROSS_PROVIDER_SCENARIOS,
    ids=[s.name for s in CROSS_PROVIDER_SCENARIOS],
)
class TestCrossProviderMigration:
    def test_migration_completes(self, scenario, run_scenario):
        """Full pipeline runs without error and produces valid output."""
        result = run_scenario(scenario)

        # Layer 1: Pipeline completion
        assert result.duration_seconds > 0
        assert result.optimized_prompt
        assert len(result.validation_results) > 0

        # Layer 2: Structural correctness
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.recommendation in {"GO", "CONDITIONAL", "NO_GO"}

        # Layer 3: Directional quality
        assert result.confidence_score >= scenario.expected_min_confidence

        # Layer 3: Recommendation within expected set
        assert result.recommendation in scenario.expected_recommendations

    def test_per_type_scores_populated(self, scenario, run_scenario):
        """Verify per-output-type score breakdowns exist."""
        result = run_scenario(scenario)

        assert isinstance(result.per_type_scores, dict)
        assert len(result.per_type_scores) > 0, "per_type_scores should have entries"

        # Layer 4: Check that at least some expected output types appear
        known_types = {"json", "classification", "short_text", "long_text"}
        found_types = set(result.per_type_scores.keys())
        assert found_types & known_types, (
            f"Expected at least one of {known_types} in per_type_scores, got {found_types}"
        )
