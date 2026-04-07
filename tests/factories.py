"""Shared test factories for RosettaStone test suite.

Provides reusable factory functions that return valid instances of core types
with sensible defaults.  Any field can be overridden via keyword arguments
using the simple ``{**defaults, **overrides}`` pattern.
"""

from __future__ import annotations

from rosettastone.core.types import EvalResult, MigrationResult, PromptPair


def prompt_pair_factory(**overrides) -> PromptPair:
    """Return a valid PromptPair with sensible defaults."""
    defaults = {
        "prompt": "What is 2+2?",
        "response": "4",
        "source_model": "openai/gpt-4o",
    }
    return PromptPair(**{**defaults, **overrides})


def eval_result_factory(**overrides) -> EvalResult:
    """Return a valid EvalResult with sensible defaults."""
    prompt_pair = overrides.pop("prompt_pair", None) or prompt_pair_factory()
    defaults = {
        "prompt_pair": prompt_pair,
        "new_response": "4",
        "scores": {"bertscore_f1": 0.85, "embedding_sim": 0.80},
        "composite_score": 0.85,
        "is_win": True,
        "details": {"output_type": "short_text"},
    }
    return EvalResult(**{**defaults, **overrides})


def migration_result_factory(**overrides) -> MigrationResult:
    """Return a valid MigrationResult with sensible defaults.

    The ``source_model`` and ``target_model`` convenience kwargs are folded into
    ``config`` so callers don't have to construct the dict manually.
    """
    # Allow callers to pass source_model / target_model as top-level kwargs
    # and have them merged into the config dict.
    source_model = overrides.pop("source_model", "openai/gpt-4o")
    target_model = overrides.pop("target_model", "anthropic/claude-sonnet-4")

    default_config = {
        "source_model": source_model,
        "target_model": target_model,
    }

    default_validation_results = [
        eval_result_factory(
            prompt_pair=prompt_pair_factory(
                prompt="test", response="expected", source_model=source_model
            ),
            new_response="actual",
            scores={"bertscore_f1": 0.85},
            composite_score=0.85,
            is_win=True,
            details={"output_type": "short_text"},
        )
    ]

    defaults: dict = {
        "config": default_config,
        "optimized_prompt": "You are a helpful assistant.",
        "baseline_results": [],
        "validation_results": default_validation_results,
        "confidence_score": 1.0,
        "baseline_score": 0.7,
        "improvement": 0.3,
        "cost_usd": 5.0,
        "duration_seconds": 60.0,
        "warnings": [],
        "safety_warnings": [],
        "recommendation": None,
        "recommendation_reasoning": None,
        "per_type_scores": {},
    }
    return MigrationResult(**{**defaults, **overrides})
