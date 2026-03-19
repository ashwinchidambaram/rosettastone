"""Model capability detection via LiteLLM metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig


def check_capabilities(config: MigrationConfig) -> tuple[list[str], list[str]]:
    """Check that the target model supports required capabilities.

    Returns (warnings, blockers).
    """
    import litellm

    warnings: list[str] = []
    blockers: list[str] = []

    try:
        source_info = litellm.get_model_info(config.source_model)
        target_info = litellm.get_model_info(config.target_model)
    except Exception as e:
        warnings.append(f"Could not fetch model info: {e}. Skipping capability checks.")
        return warnings, blockers

    # Check feature support gaps
    feature_checks = [
        ("supports_function_calling", "function/tool calling"),
        ("supports_vision", "vision/image input"),
        ("supports_response_schema", "structured output / JSON mode"),
    ]

    for key, label in feature_checks:
        source_has = source_info.get(key, False)
        target_has = target_info.get(key, False)
        if source_has and not target_has:
            warnings.append(
                f"Source model supports {label} but target does not. "
                f"GEPA may work around this, but check results carefully."
            )

    # Check context window
    source_ctx = source_info.get("max_input_tokens", 0)
    target_ctx = target_info.get("max_input_tokens", 0)
    if target_ctx and source_ctx and target_ctx < source_ctx:
        warnings.append(
            f"Target context window ({target_ctx:,}) is smaller than source ({source_ctx:,}). "
            f"Long prompts may be truncated."
        )

    return warnings, blockers
