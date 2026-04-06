"""Calibration pair collection utilities."""

from __future__ import annotations

import random
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.calibration.types import LabeledPair


def generate_synthetic_pairs(
    output_type: str,
    n_pairs: int = 100,
    seed: int | None = None,
) -> list[LabeledPair]:
    """Generate synthetic labeled pairs by degrading responses to hit target score ranges.

    Produces pairs spread across 10 score buckets (0.0–0.1, 0.1–0.2, ..., 0.9–1.0)
    for use as calibration data without requiring real Ollama/API calls.

    Args:
        output_type: One of "json", "classification", "short_text", "long_text".
        n_pairs: Total number of pairs to generate.
        seed: RNG seed for reproducibility.
    """
    from rosettastone.calibration.types import DimensionalScores, LabeledPair

    rng = random.Random(seed)
    pairs: list[LabeledPair] = []

    for i in range(n_pairs):
        # Distribute evenly across 10 score buckets
        bucket = i % 10
        bucket_min = bucket * 0.1
        bucket_max = bucket_min + 0.1
        composite = rng.uniform(bucket_min, min(bucket_max, 1.0))
        composite = round(composite, 4)

        # Build dimensional scores with small random perturbation
        noise = lambda: round(rng.gauss(0, 0.05), 4)  # noqa: E731
        scores = DimensionalScores(
            bertscore_f1=max(0.0, min(1.0, composite + noise())),
            embedding_sim=max(0.0, min(1.0, composite + noise())),
            exact_match=1.0 if composite > 0.8 else 0.0,
            llm_judge_score=max(0.0, min(1.0, composite + noise())),
            composite=composite,
        )

        # Synthetic prompt/response based on output type
        if output_type == "json":
            prompt = f'{{"task": "summarize", "id": {i}}}'
            source_resp = f'{{"result": "answer_{i}", "confidence": 0.9}}'
            target_resp = source_resp if composite > 0.7 else f'{{"result": "wrong_{i}"}}'
        elif output_type == "classification":
            prompt = f"Classify document {i}"
            source_resp = "positive" if i % 2 == 0 else "negative"
            flipped = "negative" if i % 2 == 0 else "positive"
            target_resp = source_resp if composite > 0.5 else flipped
        elif output_type == "short_text":
            prompt = f"Summarize: document {i}"
            source_resp = f"This is the correct summary for document {i}."
            target_resp = source_resp if composite > 0.6 else f"Partial summary {i}."
        else:  # long_text
            prompt = f"Write an essay about topic {i}"
            source_resp = f"A comprehensive essay about topic {i}. " * 5
            target_resp = source_resp if composite > 0.5 else f"A brief note about topic {i}."

        pairs.append(
            LabeledPair(
                pair_id=str(uuid.uuid4()),
                output_type=output_type,
                prompt=prompt,
                source_response=source_resp,
                target_response=target_resp,
                scores=scores,
            )
        )

    return pairs


def stratified_sample(
    pairs: list[LabeledPair],
    n_per_bucket: int = 10,
    seed: int | None = None,
) -> list[LabeledPair]:
    """Sample pairs stratified by composite score bucket (0.0–1.0 in 0.1 increments).

    Args:
        pairs: Input pairs to sample from.
        n_per_bucket: Maximum number of pairs per bucket.
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    buckets: dict[int, list[LabeledPair]] = {i: [] for i in range(10)}
    for pair in pairs:
        bucket_idx = min(int(pair.scores.composite * 10), 9)
        buckets[bucket_idx].append(pair)

    sampled: list[LabeledPair] = []
    for bucket_pairs in buckets.values():
        rng.shuffle(bucket_pairs)
        sampled.extend(bucket_pairs[:n_per_bucket])
    return sampled
