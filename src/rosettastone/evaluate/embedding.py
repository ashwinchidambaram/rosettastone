"""Embedding cosine similarity via sentence-transformers."""

from __future__ import annotations

from typing import Any

from rosettastone.evaluate.base import Evaluator


def compute_embedding_sim(expected: str, actual: str) -> float:
    """Compute cosine similarity between embeddings.

    Raises ImportError if sentence-transformers is not installed.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([expected, actual])
    # Cosine similarity
    import numpy as np

    a, b = embeddings[0], embeddings[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EmbeddingEvaluator(Evaluator):
    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        return {"embedding_sim": compute_embedding_sim(expected, actual)}
