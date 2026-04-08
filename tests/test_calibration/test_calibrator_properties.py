"""Property-based tests for ThresholdCalibrator.fit() and compute_alpha()."""

from __future__ import annotations

import pytest

from rosettastone.calibration.calibrator import ThresholdCalibrator
from rosettastone.calibration.collector import generate_synthetic_pairs
from rosettastone.calibration.types import (
    CalibrationDataset,
    DimensionalScores,
    HumanLabel,
    LabeledPair,
    ProductionSafety,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labeled_pair(
    pair_id: str,
    output_type: str,
    composite: float,
    is_safe: bool,
    reviewer_ids: list[str] | None = None,
) -> LabeledPair:
    """Build a LabeledPair with one or more reviewer labels."""
    if reviewer_ids is None:
        reviewer_ids = ["r1"]
    safety = ProductionSafety.SAFE if is_safe else ProductionSafety.UNSAFE
    labels = [HumanLabel(reviewer_id=rid, safety=safety) for rid in reviewer_ids]
    return LabeledPair(
        pair_id=pair_id,
        output_type=output_type,
        prompt="q",
        source_response="a",
        target_response="b",
        scores=DimensionalScores(composite=composite),
        labels=labels,
    )


def _build_labeled_dataset_for_type(output_type: str, n: int = 20) -> CalibrationDataset:
    """Build a CalibrationDataset with `n` labeled pairs for the given output_type.

    Alternates between safe (high composite) and unsafe (low composite) so the
    ROC calibration has clear signal.
    """
    pairs: list[LabeledPair] = []
    for i in range(n):
        is_safe = i % 2 == 0
        composite = 0.7 + (i % 10) * 0.03 if is_safe else (i % 10) * 0.03
        composite = max(0.0, min(1.0, composite))
        pairs.append(_make_labeled_pair(f"p{i}", output_type, composite, is_safe))
    return CalibrationDataset(pairs=pairs)


# ---------------------------------------------------------------------------
# test_fit_thresholds_in_unit_interval
# ---------------------------------------------------------------------------


def test_fit_thresholds_in_unit_interval() -> None:
    """All thresholds returned by fit() must lie in [0, 1].

    Uses generate_synthetic_pairs() to create a CalibrationDataset with ~20
    pairs per type, adds deterministic labels, then calls fit().
    """
    pytest.importorskip("sklearn")

    # Build labeled pairs for all four output types
    output_types = ["json", "classification", "short_text", "long_text"]
    all_pairs: list[LabeledPair] = []
    for ot in output_types:
        raw_pairs = generate_synthetic_pairs(output_type=ot, n_pairs=20, seed=42)
        for i, pair in enumerate(raw_pairs):
            is_safe = pair.scores.composite >= 0.5
            safety = ProductionSafety.SAFE if is_safe else ProductionSafety.UNSAFE
            labeled = LabeledPair(
                pair_id=pair.pair_id,
                output_type=pair.output_type,
                prompt=pair.prompt,
                source_response=pair.source_response,
                target_response=pair.target_response,
                scores=pair.scores,
                labels=[HumanLabel(reviewer_id="r1", safety=safety)],
            )
            all_pairs.append(labeled)

    dataset = CalibrationDataset(pairs=all_pairs)
    calibrator = ThresholdCalibrator()
    thresholds = calibrator.fit(dataset)

    for output_type, threshold in thresholds.items():
        assert 0.0 <= threshold <= 1.0, (
            f"Threshold for {output_type!r} is {threshold}, not in [0, 1]"
        )


# ---------------------------------------------------------------------------
# test_fit_returns_expected_output_types
# ---------------------------------------------------------------------------


def test_fit_returns_expected_output_types() -> None:
    """fit() must return a dict whose keys cover the four standard output types."""
    pytest.importorskip("sklearn")

    dataset = _build_labeled_dataset_for_type("classification", n=20)
    calibrator = ThresholdCalibrator()
    thresholds = calibrator.fit(dataset)

    # fit() always iterates over the four canonical types
    expected_keys = {"json", "classification", "short_text", "long_text"}
    assert set(thresholds.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(thresholds.keys())}"
    )
    # Each value should be a float
    for ot, thresh in thresholds.items():
        assert isinstance(thresh, float), (
            f"Threshold for {ot!r} should be float, got {type(thresh)}"
        )


# ---------------------------------------------------------------------------
# test_compute_alpha_in_valid_range
# ---------------------------------------------------------------------------


def test_compute_alpha_in_valid_range() -> None:
    """compute_alpha() must return a float in [-1, 1] with real labeled data."""
    pytest.importorskip("krippendorff")

    # Build pairs with two reviewers who sometimes disagree
    pairs: list[LabeledPair] = []
    for i in range(20):
        composite = 0.7 + (i % 10) * 0.03 if i % 2 == 0 else (i % 10) * 0.03
        composite = max(0.0, min(1.0, composite))
        is_safe = composite >= 0.5
        safety_r1 = ProductionSafety.SAFE if is_safe else ProductionSafety.UNSAFE
        # Reviewer 2 disagrees on ~1 in 4 pairs
        safety_r2 = ProductionSafety.UNSAFE if (i % 4 == 0 and is_safe) else safety_r1
        pairs.append(
            LabeledPair(
                pair_id=f"p{i}",
                output_type="classification",
                prompt="q",
                source_response="a",
                target_response="b",
                scores=DimensionalScores(composite=composite),
                labels=[
                    HumanLabel(reviewer_id="r1", safety=safety_r1),
                    HumanLabel(reviewer_id="r2", safety=safety_r2),
                ],
            )
        )

    dataset = CalibrationDataset(pairs=pairs)
    calibrator = ThresholdCalibrator()
    alpha = calibrator.compute_alpha(dataset)

    assert isinstance(alpha, float), f"Expected float, got {type(alpha)}"
    assert -1.0 <= alpha <= 1.0, f"Alpha={alpha} is outside [-1, 1]"


# ---------------------------------------------------------------------------
# test_compute_alpha_perfect_agreement
# ---------------------------------------------------------------------------


def test_compute_alpha_perfect_agreement() -> None:
    """All reviewers agree on every pair → alpha should be close to 1.0."""
    pytest.importorskip("krippendorff")

    pairs: list[LabeledPair] = []
    for i in range(20):
        is_safe = i % 2 == 0
        safety = ProductionSafety.SAFE if is_safe else ProductionSafety.UNSAFE
        composite = 0.9 if is_safe else 0.1
        pairs.append(
            LabeledPair(
                pair_id=f"p{i}",
                output_type="classification",
                prompt="q",
                source_response="a",
                target_response="b",
                scores=DimensionalScores(composite=composite),
                labels=[
                    HumanLabel(reviewer_id="r1", safety=safety),
                    HumanLabel(reviewer_id="r2", safety=safety),
                    HumanLabel(reviewer_id="r3", safety=safety),
                ],
            )
        )

    dataset = CalibrationDataset(pairs=pairs)
    calibrator = ThresholdCalibrator()
    alpha = calibrator.compute_alpha(dataset)

    assert abs(alpha - 1.0) < 1e-6, f"Expected alpha≈1.0 for perfect agreement, got {alpha}"
