"""Token counting with cross-tokenizer awareness."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig


def check_token_budget(config: MigrationConfig) -> tuple[list[str], list[str]]:
    """Check token budgets across source and target models.

    Returns (warnings, blockers).
    """
    import litellm

    warnings: list[str] = []
    blockers: list[str] = []

    try:
        target_info = litellm.get_model_info(config.target_model)
        max_input = target_info.get("max_input_tokens", 0)
    except Exception:
        warnings.append("Could not determine target model context window.")
        return warnings, blockers

    if not max_input:
        return warnings, blockers

    # Load data and check token counts for a sample
    try:
        from rosettastone.config import AdapterChoice

        adapter_type = config.adapter
        if config.redis_url and adapter_type == AdapterChoice.JSONL:
            adapter_type = AdapterChoice.REDIS

        if adapter_type == AdapterChoice.JSONL:
            from rosettastone.ingest.jsonl import JSONLAdapter

            if not config.data_path:
                return warnings, blockers
            adapter = JSONLAdapter(config.data_path)
        elif adapter_type == AdapterChoice.CSV:
            from rosettastone.ingest.csv_adapter import CSVAdapter

            if not config.data_path:
                return warnings, blockers
            adapter = CSVAdapter(config.data_path)
        else:
            warnings.append(
                f"Token budget check not supported for adapter '{adapter_type}'. Skipping."
            )
            return warnings, blockers

        pairs = adapter.load()
        sample = random.sample(pairs, min(20, len(pairs)))

        for i, pair in enumerate(sample):
            prompt_text = pair.prompt if isinstance(pair.prompt, str) else str(pair.prompt)
            try:
                token_count = litellm.token_counter(model=config.target_model, text=prompt_text)
            except Exception:
                continue

            usage_pct = token_count / max_input
            if usage_pct > 1.0:
                blockers.append(
                    f"Prompt {i} uses {token_count:,} tokens, exceeding target "
                    f"context window ({max_input:,})."
                )
            elif usage_pct > config.max_context_usage:
                warnings.append(
                    f"Prompt {i} uses {usage_pct:.0%} of target context window "
                    f"({token_count:,}/{max_input:,} tokens)."
                )
    except Exception as e:
        warnings.append(f"Could not check token budgets: {e}")

    return warnings, blockers
