"""E2E test scenario definitions for real model migrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    name: str
    source_model: str
    target_model: str
    category: str  # "cross_provider", "upgrade", "downgrade"
    expected_recommendations: list[str]  # e.g. ["GO", "CONDITIONAL"]
    expected_min_confidence: float = 0.0
    reflection_model: str = "gemini/gemini-2.0-flash"
    judge_model: str = "gemini/gemini-2.0-flash"
    pair_count: int = 44


# --- Category A: Cross-Provider ---

A1_OPENAI_TO_ANTHROPIC = ScenarioConfig(
    name="openai-to-anthropic",
    source_model="openai/gpt-4o-mini",
    target_model="anthropic/claude-3-5-haiku-20241022",
    category="cross_provider",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.3,
)

A2_OPENAI_TO_GEMINI = ScenarioConfig(
    name="openai-to-gemini",
    source_model="openai/gpt-4o-mini",
    target_model="gemini/gemini-2.0-flash",
    category="cross_provider",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.3,
)

A3_GEMINI_TO_ANTHROPIC = ScenarioConfig(
    name="gemini-to-anthropic",
    source_model="gemini/gemini-2.0-flash",
    target_model="anthropic/claude-3-5-haiku-20241022",
    category="cross_provider",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.3,
)


# --- Category B: Model Upgrade ---

B1_GPT4O_MINI_TO_GPT4O = ScenarioConfig(
    name="gpt4o-mini-to-gpt4o",
    source_model="openai/gpt-4o-mini",
    target_model="openai/gpt-4o",
    category="upgrade",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.5,
)

B2_HAIKU_TO_SONNET = ScenarioConfig(
    name="haiku-to-sonnet",
    source_model="anthropic/claude-3-5-haiku-20241022",
    target_model="anthropic/claude-sonnet-4",
    category="upgrade",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.5,
)


# --- Category C: Model Downgrade ---

C1_GPT4O_TO_MINI = ScenarioConfig(
    name="gpt4o-to-mini",
    source_model="openai/gpt-4o",
    target_model="openai/gpt-4o-mini",
    category="downgrade",
    expected_recommendations=["GO", "CONDITIONAL", "NO_GO"],
    expected_min_confidence=0.0,
)

C2_SONNET_TO_HAIKU = ScenarioConfig(
    name="sonnet-to-haiku",
    source_model="anthropic/claude-sonnet-4",
    target_model="anthropic/claude-3-5-haiku-20241022",
    category="downgrade",
    expected_recommendations=["GO", "CONDITIONAL", "NO_GO"],
    expected_min_confidence=0.0,
)


# --- Grouped lists ---

CROSS_PROVIDER_SCENARIOS: list[ScenarioConfig] = [
    A1_OPENAI_TO_ANTHROPIC,
    A2_OPENAI_TO_GEMINI,
    A3_GEMINI_TO_ANTHROPIC,
]

UPGRADE_SCENARIOS: list[ScenarioConfig] = [
    B1_GPT4O_MINI_TO_GPT4O,
    B2_HAIKU_TO_SONNET,
]

DOWNGRADE_SCENARIOS: list[ScenarioConfig] = [
    C1_GPT4O_TO_MINI,
    C2_SONNET_TO_HAIKU,
]

ALL_SCENARIOS: list[ScenarioConfig] = (
    CROSS_PROVIDER_SCENARIOS + UPGRADE_SCENARIOS + DOWNGRADE_SCENARIOS
)

# Cheapest scenario for quick smoke tests (~$0.21)
SMOKE_SCENARIO: ScenarioConfig = A2_OPENAI_TO_GEMINI
