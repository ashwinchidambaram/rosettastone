"""Tests for ThresholdCalibrator — no real API calls."""
from __future__ import annotations

import pytest

from rosettastone.calibration.types import (
    CalibrationDataset,
    DimensionalScores,
    HumanLabel,
    LabeledPair,
    ProductionSafety,
)


def _make_labeled_pair(
    pair_id: str,
    output_type: str,
    composite: float,
    is_safe: bool,
) -> LabeledPair:
    return LabeledPair(
        pair_id=pair_id,
        output_type=output_type,
        prompt="q",
        source_response="a",
        target_response="b",
        scores=DimensionalScores(composite=composite),
        labels=[
            HumanLabel(
                reviewer_id="r1",
                safety=ProductionSafety.SAFE if is_safe else ProductionSafety.UNSAFE,
            )
        ],
    )


class TestThresholdCalibrator:
    @pytest.fixture
    def synthetic_dataset(self):
        """Create a synthetic dataset with known separation for testing."""
        pairs = []
        # 50 safe pairs with scores 0.7–1.0, 50 unsafe with scores 0.0–0.6
        for i in range(50):
            pairs.append(_make_labeled_pair(f"safe_{i}", "classification", 0.7 + i * 0.006, True))
        for i in range(50):
            pairs.append(_make_labeled_pair(f"unsafe_{i}", "classification", i * 0.012, False))
        return CalibrationDataset(pairs=pairs)

    def test_fit_returns_thresholds_for_all_types(self, synthetic_dataset):
        """fit() returns a threshold for every output type."""
        pytest.importorskip("sklearn")
        from rosettastone.calibration.calibrator import ThresholdCalibrator

        calibrator = ThresholdCalibrator()
        thresholds = calibrator.fit(synthetic_dataset)
        assert set(thresholds.keys()) == {"json", "classification", "short_text", "long_text"}
        for thresh in thresholds.values():
            assert 0.0 <= thresh <= 1.0

    def test_fit_uses_default_for_sparse_types(self, synthetic_dataset):
        """Types with < 5 labeled pairs fall back to DEFAULT_THRESHOLDS."""
        pytest.importorskip("sklearn")
        from rosettastone.calibration.calibrator import ThresholdCalibrator
        from rosettastone.decision.recommendation import DEFAULT_THRESHOLDS

        calibrator = ThresholdCalibrator()
        thresholds = calibrator.fit(synthetic_dataset)
        # json, short_text, long_text have no pairs → fallback to defaults
        assert thresholds["json"] == DEFAULT_THRESHOLDS["json"]
        assert thresholds["short_text"] == DEFAULT_THRESHOLDS["short_text"]
        assert thresholds["long_text"] == DEFAULT_THRESHOLDS["long_text"]

    def test_compute_alpha_perfect_agreement(self):
        """Alpha = 1.0 when all reviewers agree."""
        pytest.importorskip("krippendorff")
        from rosettastone.calibration.calibrator import ThresholdCalibrator

        pairs = [
            LabeledPair(
                pair_id=f"p{i}",
                output_type="classification",
                prompt="q",
                source_response="a",
                target_response="b",
                scores=DimensionalScores(composite=0.9 if i % 2 == 0 else 0.3),
                labels=[
                    HumanLabel(
                        reviewer_id="r1",
                        safety=ProductionSafety.SAFE if i % 2 == 0 else ProductionSafety.UNSAFE,
                    ),
                    HumanLabel(
                        reviewer_id="r2",
                        safety=ProductionSafety.SAFE if i % 2 == 0 else ProductionSafety.UNSAFE,
                    ),
                ],
            )
            for i in range(10)
        ]
        dataset = CalibrationDataset(pairs=pairs)
        calibrator = ThresholdCalibrator()
        alpha = calibrator.compute_alpha(dataset)
        assert abs(alpha - 1.0) < 1e-6

    def test_report_output_contains_thresholds(self):
        """report() returns a string mentioning all output types."""
        from rosettastone.calibration.calibrator import ThresholdCalibrator

        calibrator = ThresholdCalibrator()
        thresholds = {"json": 0.95, "classification": 0.88, "short_text": 0.79, "long_text": 0.74}
        dataset = CalibrationDataset()
        report = calibrator.report(dataset, thresholds)
        assert "json" in report
        assert "classification" in report
        assert "0.8800" in report  # classification threshold formatted

    def test_fit_raises_import_error_without_sklearn(self, monkeypatch):
        """fit() raises ImportError with helpful message when sklearn is missing."""
        import sys

        from rosettastone.calibration.calibrator import ThresholdCalibrator

        calibrator = ThresholdCalibrator()
        monkeypatch.setitem(sys.modules, "sklearn", None)
        monkeypatch.setitem(sys.modules, "sklearn.metrics", None)
        with pytest.raises(ImportError, match="scikit-learn"):
            calibrator.fit(CalibrationDataset())
