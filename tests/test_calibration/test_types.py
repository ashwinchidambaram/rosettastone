"""Tests for calibration dataset types."""
from __future__ import annotations

from rosettastone.calibration.types import (
    CalibrationDataset,
    DimensionalScores,
    HumanLabel,
    LabeledPair,
    ProductionSafety,
)


class TestLabeledPair:
    def test_majority_label_safe(self):
        pair = LabeledPair(
            pair_id="p1",
            output_type="classification",
            prompt="q",
            source_response="a",
            target_response="a",
            scores=DimensionalScores(composite=0.9),
            labels=[
                HumanLabel(reviewer_id="r1", safety=ProductionSafety.SAFE),
                HumanLabel(reviewer_id="r2", safety=ProductionSafety.SAFE),
                HumanLabel(reviewer_id="r3", safety=ProductionSafety.UNSAFE),
            ],
        )
        assert pair.majority_label == ProductionSafety.SAFE

    def test_majority_label_none_when_no_labels(self):
        pair = LabeledPair(
            pair_id="p2",
            output_type="json",
            prompt="q",
            source_response="{}",
            target_response="{}",
            scores=DimensionalScores(),
        )
        assert pair.majority_label is None

    def test_is_safe_majority_true(self):
        pair = LabeledPair(
            pair_id="p3",
            output_type="short_text",
            prompt="q",
            source_response="a",
            target_response="a",
            scores=DimensionalScores(composite=0.85),
            labels=[
                HumanLabel(reviewer_id="r1", safety=ProductionSafety.SAFE),
                HumanLabel(reviewer_id="r2", safety=ProductionSafety.SAFE),
            ],
        )
        assert pair.is_safe_majority is True

    def test_round_trip_json(self):
        pair = LabeledPair(
            pair_id="p4",
            output_type="long_text",
            prompt="write essay",
            source_response="essay here",
            target_response="essay there",
            scores=DimensionalScores(composite=0.75, bertscore_f1=0.8),
            labels=[HumanLabel(reviewer_id="r1", safety=ProductionSafety.BORDERLINE)],
        )
        serialized = pair.model_dump_json()
        restored = LabeledPair.model_validate_json(serialized)
        assert restored.pair_id == pair.pair_id
        assert restored.scores.composite == pair.scores.composite
        assert restored.labels[0].safety == ProductionSafety.BORDERLINE


class TestCalibrationDataset:
    def test_by_output_type(self):
        pairs = [
            LabeledPair(pair_id=f"p{i}", output_type=ot, prompt="q",
                        source_response="a", target_response="b",
                        scores=DimensionalScores())
            for i, ot in enumerate(["json", "classification", "json", "long_text"])
        ]
        dataset = CalibrationDataset(pairs=pairs)
        assert len(dataset.by_output_type("json")) == 2
        assert len(dataset.by_output_type("classification")) == 1
        assert len(dataset.by_output_type("short_text")) == 0

    def test_labeled_pairs_filters_unlabeled(self):
        labeled = LabeledPair(
            pair_id="p1", output_type="json", prompt="q",
            source_response="a", target_response="b",
            scores=DimensionalScores(),
            labels=[HumanLabel(reviewer_id="r1", safety=ProductionSafety.SAFE)],
        )
        unlabeled = LabeledPair(
            pair_id="p2", output_type="json", prompt="q",
            source_response="a", target_response="b",
            scores=DimensionalScores(),
        )
        dataset = CalibrationDataset(pairs=[labeled, unlabeled])
        assert len(dataset.labeled_pairs()) == 1
        assert dataset.labeled_pairs()[0].pair_id == "p1"
