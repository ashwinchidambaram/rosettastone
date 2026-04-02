"""Calibration dataset types."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProductionSafety(StrEnum):
    """Human-assigned production safety label."""
    SAFE = "safe"           # OK to deploy: behavior is equivalent
    UNSAFE = "unsafe"       # Do NOT deploy: behavior is meaningfully different
    BORDERLINE = "borderline"  # Reviewer uncertain


class DimensionalScores(BaseModel):
    """Per-metric scores for a single evaluation."""
    bertscore_f1: float = 0.0
    embedding_sim: float = 0.0
    exact_match: float = 0.0
    llm_judge_score: float = 0.0
    composite: float = 0.0


class HumanLabel(BaseModel):
    """A human reviewer's label for a prompt/response pair."""
    reviewer_id: str
    safety: ProductionSafety
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = ""


class LabeledPair(BaseModel):
    """A prompt/response pair with automated scores and human labels."""
    pair_id: str
    output_type: str  # "json", "classification", "short_text", "long_text"
    prompt: str
    source_response: str
    target_response: str
    scores: DimensionalScores
    labels: list[HumanLabel] = []
    metadata: dict[str, Any] = {}

    @property
    def majority_label(self) -> ProductionSafety | None:
        """Return the majority-vote label, or None if no labels."""
        if not self.labels:
            return None
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label.safety.value] = counts.get(label.safety.value, 0) + 1
        winner = max(counts, key=lambda k: counts[k])
        return ProductionSafety(winner)

    @property
    def is_safe_majority(self) -> bool | None:
        """True if majority label is SAFE, False if UNSAFE, None if no labels or borderline."""
        ml = self.majority_label
        if ml is None or ml == ProductionSafety.BORDERLINE:
            return None
        return ml == ProductionSafety.SAFE


class CalibrationDataset(BaseModel):
    """Collection of labeled pairs for threshold calibration."""
    version: str = "1.0"
    pairs: list[LabeledPair] = []
    metadata: dict[str, Any] = {}

    def by_output_type(self, output_type: str) -> list[LabeledPair]:
        return [p for p in self.pairs if p.output_type == output_type]

    def labeled_pairs(self) -> list[LabeledPair]:
        """Return only pairs that have at least one label."""
        return [p for p in self.pairs if p.labels]
