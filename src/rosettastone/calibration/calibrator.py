"""Threshold calibrator — ROC-based calibration against human labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.calibration.types import CalibrationDataset

# Target false positive rates per output type
# (higher = more permissive threshold = lower numeric threshold)
FPR_TARGETS: dict[str, float] = {
    "json": 0.02,  # 2% FPR — strict
    "classification": 0.05,  # 5% FPR
    "short_text": 0.08,  # 8% FPR
    "long_text": 0.10,  # 10% FPR — most permissive
}


class ThresholdCalibrator:
    """Calibrates win-rate thresholds using ROC analysis on human-labeled data."""

    def fit(
        self,
        dataset: CalibrationDataset,
        fpr_targets: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute calibrated thresholds for each output type.

        For each output type, builds an ROC curve from human-labeled pairs and
        finds the composite score threshold that achieves the target FPR.

        Args:
            dataset: Labeled calibration dataset.
            fpr_targets: Override FPR targets per output type. Defaults to FPR_TARGETS.

        Returns:
            Dict mapping output_type → calibrated threshold (0.0–1.0).
        """
        try:
            from sklearn.metrics import roc_curve
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for calibration. "
                "Install with: uv pip install 'rosettastone[calibration]'"
            ) from exc

        effective_fpr_targets = {**FPR_TARGETS, **(fpr_targets or {})}
        thresholds: dict[str, float] = {}

        for output_type in ["json", "classification", "short_text", "long_text"]:
            pairs = [
                p
                for p in dataset.labeled_pairs()
                if p.output_type == output_type and p.is_safe_majority is not None
            ]

            if len(pairs) < 5:
                # Not enough data — fall back to default threshold
                from rosettastone.decision.recommendation import DEFAULT_THRESHOLDS

                thresholds[output_type] = DEFAULT_THRESHOLDS.get(output_type, 0.80)
                continue

            y_true = [1 if p.is_safe_majority else 0 for p in pairs]
            y_scores = [p.scores.composite for p in pairs]

            # sklearn roc_curve: fpr, tpr, thresholds
            fpr_arr, _tpr_arr, thresh_arr = roc_curve(y_true, y_scores)

            target_fpr = effective_fpr_targets.get(output_type, 0.05)

            # sklearn roc_curve returns fpr/tpr of length n+1 (prepends a (0,0) point)
            # and thresholds of length n — skip fpr[0] to align indices correctly.
            # Thresholds are sorted descending (high→low), fpr ascending (low→high).
            # Iterate forward and keep updating calibrated on each match so that
            # the last match is the lowest threshold still satisfying fpr <= target
            # (i.e. the least conservative threshold within the FPR budget).
            calibrated = 0.5  # fallback
            for fpr_val, thresh_val in zip(fpr_arr[1:], thresh_arr):
                if fpr_val <= target_fpr:
                    calibrated = float(thresh_val)

            thresholds[output_type] = round(max(0.0, min(1.0, calibrated)), 4)

        return thresholds

    def compute_alpha(self, dataset: CalibrationDataset) -> float:
        """Compute Krippendorff's alpha inter-rater reliability.

        Returns a float in [-1, 1]. Values >= 0.80 indicate acceptable reliability.
        Warns if alpha < 0.80.

        Args:
            dataset: Labeled dataset with multiple reviewers per pair.
        """
        import logging

        try:
            import krippendorff
        except ImportError as exc:
            raise ImportError(
                "krippendorff is required for reliability computation. "
                "Install with: uv pip install 'rosettastone[calibration]'"
            ) from exc

        logger = logging.getLogger(__name__)

        # Build a reliability matrix: rows = reviewers, cols = pairs
        # Value: 0=UNSAFE, 0.5=BORDERLINE, 1=SAFE (numeric for interval alpha)
        label_map = {"safe": 1.0, "borderline": 0.5, "unsafe": 0.0}
        labeled = dataset.labeled_pairs()

        if not labeled:
            return 0.0

        # Gather all reviewer IDs
        reviewer_ids: list[str] = sorted(
            {label.reviewer_id for pair in labeled for label in pair.labels}
        )
        if len(reviewer_ids) < 2:
            logger.warning(
                "Krippendorff alpha requires at least 2 reviewers; got %d", len(reviewer_ids)
            )
            return 1.0  # Perfect agreement by definition with 1 reviewer

        # Build matrix: rows=reviewers, cols=pairs; None for missing
        reviewer_idx = {r: i for i, r in enumerate(reviewer_ids)}
        matrix: list[list[float | None]] = [[None] * len(labeled) for _ in reviewer_ids]
        for col, pair in enumerate(labeled):
            for label in pair.labels:
                row = reviewer_idx[label.reviewer_id]
                matrix[row][col] = label_map.get(label.safety.value, 0.5)

        alpha = float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"))

        if alpha < 0.80:
            logger.warning(
                "Krippendorff alpha=%.3f < 0.80 — inter-rater reliability is low. "
                "Consider additional reviewer training or clearer labeling guidelines.",
                alpha,
            )
        return alpha

    def report(self, dataset: CalibrationDataset, thresholds: dict[str, float]) -> str:
        """Generate a text report of calibration results."""
        lines = ["# Threshold Calibration Report", ""]
        lines.append(f"Total pairs: {len(dataset.pairs)}")
        lines.append(f"Labeled pairs: {len(dataset.labeled_pairs())}")
        lines.append("")
        lines.append("## Calibrated Thresholds")
        lines.append("")

        from rosettastone.decision.recommendation import DEFAULT_THRESHOLDS

        for ot, thresh in thresholds.items():
            default = DEFAULT_THRESHOLDS.get(ot, 0.80)
            delta = thresh - default
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"- **{ot}**: {thresh:.4f} (default={default:.2f}, delta={sign}{delta:.4f})"
            )

        lines.append("")
        lines.append("## Per-Type Sample Counts")
        for ot in ["json", "classification", "short_text", "long_text"]:
            labeled_for_type = [
                p
                for p in dataset.labeled_pairs()
                if p.output_type == ot and p.is_safe_majority is not None
            ]
            lines.append(f"- {ot}: {len(labeled_for_type)} labeled pairs")

        return "\n".join(lines)
