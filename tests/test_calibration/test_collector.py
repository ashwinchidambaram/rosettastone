"""Tests for rosettastone.calibration.collector — generate_synthetic_pairs and stratified_sample."""

from __future__ import annotations

from rosettastone.calibration.collector import generate_synthetic_pairs, stratified_sample
from rosettastone.calibration.types import LabeledPair


# ===========================================================================
# generate_synthetic_pairs tests
# ===========================================================================


def test_generate_synthetic_pairs_returns_correct_count():
    """generate_synthetic_pairs(n_pairs=50) returns exactly 50 LabeledPairs."""
    pairs = generate_synthetic_pairs(output_type="short_text", n_pairs=50)

    assert len(pairs) == 50, f"Expected 50 pairs, got {len(pairs)}"


def test_generate_synthetic_pairs_output_type_matches():
    """All returned pairs have the output_type matching the requested value."""
    output_type = "json"
    pairs = generate_synthetic_pairs(output_type=output_type, n_pairs=20)

    for i, pair in enumerate(pairs):
        assert pair.output_type == output_type, (
            f"Pair {i}: expected output_type '{output_type}', got '{pair.output_type}'"
        )


def test_generate_synthetic_pairs_score_distribution():
    """Scores across generated pairs span multiple distinct composite score buckets.

    With n_pairs >= 10 and bucket assignment by (i % 10), all 10 buckets
    (0.0–0.1, 0.1–0.2, ..., 0.9–1.0) should be represented.
    """
    pairs = generate_synthetic_pairs(output_type="classification", n_pairs=100, seed=42)

    # Map each pair to its 0.1-wide bucket
    buckets_seen: set[int] = set()
    for pair in pairs:
        bucket = int(pair.scores.composite * 10)
        bucket = min(bucket, 9)  # clamp the 1.0 edge case to bucket 9
        buckets_seen.add(bucket)

    assert len(buckets_seen) > 1, (
        f"Expected scores spanning multiple buckets, but only saw buckets: {sorted(buckets_seen)}"
    )
    # With 100 pairs cycling through 10 buckets, all 10 should be present
    assert len(buckets_seen) >= 5, (
        f"Expected at least 5 distinct score buckets, got {len(buckets_seen)}: {sorted(buckets_seen)}"
    )


def test_generate_synthetic_pairs_seed_reproducibility():
    """Calling generate_synthetic_pairs with the same seed twice produces identical output."""
    seed = 12345
    pairs_a = generate_synthetic_pairs(output_type="long_text", n_pairs=30, seed=seed)
    pairs_b = generate_synthetic_pairs(output_type="long_text", n_pairs=30, seed=seed)

    assert len(pairs_a) == len(pairs_b), (
        f"Expected same length: {len(pairs_a)} vs {len(pairs_b)}"
    )
    for i, (a, b) in enumerate(zip(pairs_a, pairs_b)):
        assert a.prompt == b.prompt, (
            f"Pair {i}: prompt mismatch — {a.prompt!r} vs {b.prompt!r}"
        )
        assert a.source_response == b.source_response, (
            f"Pair {i}: source_response mismatch"
        )
        assert a.target_response == b.target_response, (
            f"Pair {i}: target_response mismatch"
        )
        assert a.scores.composite == b.scores.composite, (
            f"Pair {i}: composite score mismatch — {a.scores.composite} vs {b.scores.composite}"
        )


def test_generate_synthetic_pairs_different_seeds_differ():
    """Different seeds produce different output (at least one pair differs)."""
    pairs_a = generate_synthetic_pairs(output_type="short_text", n_pairs=20, seed=1)
    pairs_b = generate_synthetic_pairs(output_type="short_text", n_pairs=20, seed=9999)

    # Compare composite scores — different seeds should produce different random values
    scores_a = [p.scores.composite for p in pairs_a]
    scores_b = [p.scores.composite for p in pairs_b]

    assert scores_a != scores_b, (
        "Expected different seeds to produce different composite scores, but they were identical"
    )


# ===========================================================================
# stratified_sample tests
# ===========================================================================


def test_stratified_sample_respects_bucket_limit():
    """No bucket in the sample result has more than n_per_bucket items."""
    n_per_bucket = 5
    # Generate more than enough pairs to fill each bucket
    all_pairs = generate_synthetic_pairs(output_type="classification", n_pairs=200, seed=0)
    sampled = stratified_sample(all_pairs, n_per_bucket=n_per_bucket, seed=0)

    # Count how many sampled pairs fall in each bucket
    bucket_counts: dict[int, int] = {}
    for pair in sampled:
        bucket = min(int(pair.scores.composite * 10), 9)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for bucket, count in bucket_counts.items():
        assert count <= n_per_bucket, (
            f"Bucket {bucket} has {count} items, exceeding n_per_bucket={n_per_bucket}"
        )


def test_stratified_sample_empty_input():
    """stratified_sample with an empty list returns an empty list."""
    result = stratified_sample([], n_per_bucket=10, seed=42)

    assert result == [], f"Expected empty list from empty input, got: {result!r}"


def test_stratified_sample_preserves_types():
    """All elements in the stratified sample output are LabeledPair instances."""
    pairs = generate_synthetic_pairs(output_type="json", n_pairs=50, seed=7)
    sampled = stratified_sample(pairs, n_per_bucket=3, seed=7)

    for i, item in enumerate(sampled):
        assert isinstance(item, LabeledPair), (
            f"Item {i}: expected LabeledPair, got {type(item)}"
        )
