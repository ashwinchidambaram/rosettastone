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


# --- Category A: Cross-Provider (Anthropic ↔ Gemini) ---

A1_HAIKU_TO_GEMINI = ScenarioConfig(
    name="haiku-to-gemini",
    source_model="anthropic/claude-haiku-4-5-20251001",
    target_model="gemini/gemini-2.0-flash",
    category="cross_provider",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.3,
)

A2_GEMINI_TO_HAIKU = ScenarioConfig(
    name="gemini-to-haiku",
    source_model="gemini/gemini-2.0-flash",
    target_model="anthropic/claude-haiku-4-5-20251001",
    category="cross_provider",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.3,
)

A3_GEMINI_TO_SONNET = ScenarioConfig(
    name="gemini-to-sonnet",
    source_model="gemini/gemini-2.0-flash",
    target_model="anthropic/claude-sonnet-4",
    category="cross_provider",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.3,
)


# --- Category B: Model Upgrade (same provider) ---

B1_HAIKU_TO_SONNET = ScenarioConfig(
    name="haiku-to-sonnet",
    source_model="anthropic/claude-haiku-4-5-20251001",
    target_model="anthropic/claude-sonnet-4",
    category="upgrade",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.5,
)

B2_FLASH_TO_PRO = ScenarioConfig(
    name="flash-to-pro",
    source_model="gemini/gemini-2.0-flash",
    target_model="gemini/gemini-1.5-pro",
    category="upgrade",
    expected_recommendations=["GO", "CONDITIONAL"],
    expected_min_confidence=0.5,
)


# --- Category C: Model Downgrade (same provider) ---

C1_SONNET_TO_HAIKU = ScenarioConfig(
    name="sonnet-to-haiku",
    source_model="anthropic/claude-sonnet-4",
    target_model="anthropic/claude-haiku-4-5-20251001",
    category="downgrade",
    expected_recommendations=["GO", "CONDITIONAL", "NO_GO"],
    expected_min_confidence=0.0,
)

C2_PRO_TO_FLASH = ScenarioConfig(
    name="pro-to-flash",
    source_model="gemini/gemini-1.5-pro",
    target_model="gemini/gemini-2.0-flash",
    category="downgrade",
    expected_recommendations=["GO", "CONDITIONAL", "NO_GO"],
    expected_min_confidence=0.0,
)


# --- Grouped lists ---

CROSS_PROVIDER_SCENARIOS: list[ScenarioConfig] = [
    A1_HAIKU_TO_GEMINI,
    A2_GEMINI_TO_HAIKU,
    A3_GEMINI_TO_SONNET,
]

UPGRADE_SCENARIOS: list[ScenarioConfig] = [
    B1_HAIKU_TO_SONNET,
    B2_FLASH_TO_PRO,
]

DOWNGRADE_SCENARIOS: list[ScenarioConfig] = [
    C1_SONNET_TO_HAIKU,
    C2_PRO_TO_FLASH,
]

ALL_SCENARIOS: list[ScenarioConfig] = (
    CROSS_PROVIDER_SCENARIOS + UPGRADE_SCENARIOS + DOWNGRADE_SCENARIOS
)

# Cheapest scenario for quick smoke tests — gemini source (cheapest data gen) + haiku target
SMOKE_SCENARIO: ScenarioConfig = A2_GEMINI_TO_HAIKU
