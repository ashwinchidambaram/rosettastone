"""Shared fixtures: sample prompt pairs, mock LLM responses."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rosettastone.core.types import OutputType, PromptPair


@pytest.fixture
def sample_pairs() -> list[PromptPair]:
    """A small set of prompt pairs for testing."""
    return [
        PromptPair(
            prompt="What is the capital of France?",
            response="Paris",
            source_model="openai/gpt-4o",
            output_type=OutputType.CLASSIFICATION,
        ),
        PromptPair(
            prompt="Summarize the benefits of exercise.",
            response=(
                "Regular exercise improves cardiovascular health, strengthens muscles, "
                "boosts mood through endorphin release, and helps maintain a healthy weight."
            ),
            source_model="openai/gpt-4o",
            output_type=OutputType.SHORT_TEXT,
        ),
        PromptPair(
            prompt='Return a JSON object with name and age: {"name": "Alice", "age": 30}',
            response='{"name": "Alice", "age": 30}',
            source_model="openai/gpt-4o",
            output_type=OutputType.JSON,
        ),
    ]


@pytest.fixture
def sample_jsonl_file(sample_pairs: list[PromptPair]) -> Path:
    """Create a temporary JSONL file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for pair in sample_pairs:
            line = {
                "prompt": pair.prompt,
                "response": pair.response,
                "source_model": pair.source_model,
            }
            f.write(json.dumps(line) + "\n")
        return Path(f.name)
