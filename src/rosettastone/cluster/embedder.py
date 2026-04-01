"""Prompt clustering by semantic similarity."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from rosettastone.cluster.types import ClusterResult, PromptCluster
from rosettastone.core.types import PromptPair

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy imports — these may not be installed.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

try:
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import silhouette_score
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore[assignment,misc]
    HDBSCAN = None  # type: ignore[assignment,misc]
    silhouette_score = None  # type: ignore[assignment,misc]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[assignment,misc]

# Stop words to exclude from auto-labels
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "below",
        "between",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
    }
)

_MAX_LABEL_LENGTH = 50


class PromptClusterer:
    """Cluster prompt pairs by semantic similarity."""

    def __init__(
        self,
        n_clusters: int | None = None,
        method: Literal["kmeans", "hdbscan"] = "kmeans",
        min_cluster_size: int = 5,
    ) -> None:
        self._n_clusters = n_clusters
        self._method = method
        self._min_cluster_size = min_cluster_size

    def cluster(self, pairs: list[PromptPair]) -> ClusterResult:
        """Cluster prompt pairs by semantic similarity."""
        if not pairs:
            return ClusterResult(clusters=[], n_clusters=0)

        embeddings = self._embed_prompts(pairs)
        labels = self._fit_clusters(embeddings)
        return self._build_result(pairs, labels, embeddings)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_prompts(self, pairs: list[PromptPair]) -> np.ndarray:
        """Embed prompts using sentence-transformers or TF-IDF fallback."""
        texts = [self._extract_prompt_text(p) for p in pairs]
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence_transformers not installed")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return cast(np.ndarray[Any, Any], model.encode(texts))
        except ImportError:
            logger.info("sentence_transformers unavailable, using TF-IDF")
            return self._tfidf_embed(texts)

    def _tfidf_embed(self, texts: list[str]) -> np.ndarray:
        """TF-IDF fallback when sentence-transformers unavailable."""
        if TfidfVectorizer is None:
            msg = "sklearn is required for TF-IDF fallback. Install with: pip install scikit-learn"
            raise ImportError(msg)
        vectorizer = TfidfVectorizer(max_features=512)
        return cast(np.ndarray[Any, Any], vectorizer.fit_transform(texts).toarray())

    def _extract_prompt_text(self, pair: PromptPair) -> str:
        """Extract plain text from prompt (handles str and list[dict])."""
        if isinstance(pair.prompt, str):
            return pair.prompt
        # Join content from message dicts
        parts = []
        for msg in pair.prompt:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg["content"]))
        return " ".join(parts) or ""

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _fit_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply clustering algorithm."""
        if self._method == "hdbscan":
            if HDBSCAN is None:
                msg = "sklearn is required for HDBSCAN clustering."
                raise ImportError(msg)
            clusterer = HDBSCAN(min_cluster_size=self._min_cluster_size)
            return cast(np.ndarray[Any, Any], clusterer.fit_predict(embeddings))

        if KMeans is None:
            msg = "sklearn is required for KMeans clustering."
            raise ImportError(msg)
        n = self._n_clusters or min(5, len(embeddings))
        # Cap n_clusters to number of samples
        n = min(n, len(embeddings))
        return cast(np.ndarray[Any, Any], KMeans(n_clusters=n, n_init="auto").fit_predict(embeddings))

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(
        self,
        pairs: list[PromptPair],
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> ClusterResult:
        """Assemble ClusterResult from pairs and their cluster labels."""
        unique_labels = sorted(set(labels))
        noise_pairs: list[PromptPair] = []
        clusters: list[PromptCluster] = []

        cluster_id_counter = 0
        for label_val in unique_labels:
            indices = [i for i, lbl in enumerate(labels) if lbl == label_val]

            if label_val == -1:
                # Noise / outliers (HDBSCAN)
                noise_pairs.extend(pairs[i] for i in indices)
                continue

            cluster_pairs = [pairs[i] for i in indices]
            cluster_embeddings = embeddings[indices]
            centroid = cluster_embeddings.mean(axis=0).tolist()

            clusters.append(
                PromptCluster(
                    cluster_id=cluster_id_counter,
                    label=self._auto_label(cluster_pairs),
                    pairs=cluster_pairs,
                    centroid=centroid,
                )
            )
            cluster_id_counter += 1

        # Silhouette score requires 2+ clusters and more samples than clusters
        sil_score: float | None = None
        n_real_clusters = len(clusters)
        if n_real_clusters >= 2 and silhouette_score is not None:
            # Filter out noise labels for silhouette calculation
            mask = labels != -1
            if mask.sum() > n_real_clusters:
                sil_score = float(silhouette_score(embeddings[mask], labels[mask]))

        return ClusterResult(
            clusters=clusters,
            noise_pairs=noise_pairs,
            n_clusters=n_real_clusters,
            silhouette_score=sil_score,
        )

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def _auto_label(self, pairs: list[PromptPair]) -> str:
        """Generate cluster label from most common terms."""
        all_words: list[str] = []
        for pair in pairs:
            text = self._extract_prompt_text(pair)
            words = text.lower().split()
            all_words.extend(w for w in words if w not in _STOP_WORDS and len(w) > 1)

        if not all_words:
            return "cluster"

        counter = Counter(all_words)
        top_words = [word for word, _ in counter.most_common(5)]
        label = " ".join(top_words)

        if len(label) > _MAX_LABEL_LENGTH:
            label = label[:_MAX_LABEL_LENGTH].rsplit(" ", 1)[0]

        return label
