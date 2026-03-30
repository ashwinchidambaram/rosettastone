"""Data types for prompt clustering."""

from __future__ import annotations

from dataclasses import dataclass, field

from rosettastone.core.types import PromptPair


@dataclass
class PromptCluster:
    """A single cluster of semantically similar prompt pairs."""

    cluster_id: int
    label: str  # auto-generated from common terms
    pairs: list[PromptPair] = field(default_factory=list)
    centroid: list[float] | None = None


@dataclass
class ClusterResult:
    """Result of clustering a set of prompt pairs."""

    clusters: list[PromptCluster]
    noise_pairs: list[PromptPair] = field(default_factory=list)
    n_clusters: int = 0
    silhouette_score: float | None = None
