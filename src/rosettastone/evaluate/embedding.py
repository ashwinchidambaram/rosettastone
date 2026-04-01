"""Embedding cosine similarity via sentence-transformers."""

from __future__ import annotations

import functools
from typing import Any

from rosettastone.evaluate.base import Evaluator


@functools.lru_cache(maxsize=4)
def _get_sentence_transformer(model_name: str) -> Any:
    """Load and cache a SentenceTransformer model. Cached per model name."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def compute_embedding_sim(expected: str, actual: str) -> float:
    """Compute cosine similarity between embeddings.

    Raises ImportError if sentence-transformers is not installed.
    """
    model = _get_sentence_transformer("all-MiniLM-L6-v2")
    embeddings = model.encode([expected, actual])
    # Cosine similarity
    import numpy as np

    a, b = embeddings[0], embeddings[1]
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm_product)


class EmbeddingEvaluator(Evaluator):
    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        return {"embedding_sim": compute_embedding_sim(expected, actual)}
