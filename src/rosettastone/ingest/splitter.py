"""Train/validation/test split logic with deduplication."""

from __future__ import annotations

import hashlib
import json
import random

from rosettastone.core.types import PromptPair


def _fingerprint(pair: PromptPair) -> str:
    if isinstance(pair.prompt, list):
        prompt_text = json.dumps(pair.prompt, sort_keys=True)
    else:
        prompt_text = pair.prompt
    text = prompt_text + pair.response
    return hashlib.sha256(text.encode()).hexdigest()


def deduplicate(pairs: list[PromptPair]) -> list[PromptPair]:
    seen: set[str] = set()
    unique: list[PromptPair] = []
    for pair in pairs:
        fp = _fingerprint(pair)
        if fp not in seen:
            seen.add(fp)
            unique.append(pair)
    return unique


def split_data(
    pairs: list[PromptPair],
    train_ratio: float = 0.2,
    val_ratio: float = 0.8,
    seed: int | None = None,
) -> tuple[list[PromptPair], list[PromptPair], list[PromptPair]]:
    """Split pairs into train, validation, and test sets.

    train_ratio: fraction of data for training (GEPA optimization)
    val_ratio: of the remaining data, fraction for validation vs test
    seed: optional RNG seed for reproducible splits
    """
    pairs = deduplicate(pairs)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(pairs)
    else:
        random.shuffle(pairs)

    train_end = max(1, int(len(pairs) * train_ratio))
    train = pairs[:train_end]
    remaining = pairs[train_end:]

    val_end = max(1, int(len(remaining) * val_ratio))
    val = remaining[:val_end]
    test = remaining[val_end:]

    # Ensure test set is never empty
    if not test and val:
        test = [val.pop()]

    return train, val, test
