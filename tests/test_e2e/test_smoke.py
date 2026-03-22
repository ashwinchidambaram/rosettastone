"""Smoke E2E test — cheapest scenario to validate full pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.core.migrator import Migrator
from rosettastone.testing.redis_populator import RedisPopulator
from rosettastone.testing.scenarios import SMOKE_SCENARIO
from rosettastone.testing.synth_data import SyntheticDataGenerator

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def smoke_data(generated_data_cache):
    """Generate synthetic data for the smoke scenario (cached per session)."""
    model = SMOKE_SCENARIO.source_model
    if model not in generated_data_cache:
        gen = SyntheticDataGenerator(source_model=model)
        generated_data_cache[model] = gen.generate()
    return generated_data_cache[model]


def test_smoke_full_pipeline(smoke_data, redis_url, clean_redis):
    """Full pipeline via Redis: generate → populate → migrate → assert."""
    scenario = SMOKE_SCENARIO

    # Populate Redis
    populator = RedisPopulator(redis_url)
    count = populator.populate(smoke_data)
    assert count > 0, "No data written to Redis"

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

        result = Migrator(config).run()

    # Layer 1: Pipeline completion
    assert result.duration_seconds > 0
    assert result.optimized_prompt
    assert len(result.validation_results) > 0

    # Layer 2: Structural correctness
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.recommendation in {"GO", "CONDITIONAL", "NO_GO"}

    # Layer 3: Directional quality (cross-provider, >= 0.3)
    assert result.confidence_score >= scenario.expected_min_confidence


def test_smoke_also_works_via_jsonl(smoke_data):
    """Same data via JSONL file instead of Redis — validates JSONL path works too."""
    scenario = SMOKE_SCENARIO

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "smoke_data.jsonl"
        written = RedisPopulator.write_jsonl(smoke_data, jsonl_path, scenario.source_model)
        assert written > 0

        config = MigrationConfig(
            source_model=scenario.source_model,
            target_model=scenario.target_model,
            data_path=jsonl_path,
            output_dir=Path(tmpdir) / "output",
            reflection_model=scenario.reflection_model,
            judge_model=scenario.judge_model,
            gepa_auto="light",
            skip_preflight=True,
        )

        result = Migrator(config).run()

    # Layer 1: Pipeline completion
    assert result.duration_seconds > 0
    assert result.optimized_prompt
    assert len(result.validation_results) > 0

    # Layer 2: Structural correctness
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.recommendation in {"GO", "CONDITIONAL", "NO_GO"}
