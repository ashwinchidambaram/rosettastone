"""Composite evaluator: combines metrics by output type, computes pairwise win rate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import litellm

from rosettastone.core.types import EvalResult, OutputType, PromptPair
from rosettastone.evaluate.exact_match import ExactMatchEvaluator
from rosettastone.evaluate.json_validator import JSONEvaluator
from rosettastone.evaluate.types import detect_output_type

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig


WIN_THRESHOLD = 0.8


class CompositeEvaluator:
    def __init__(self, config: MigrationConfig) -> None:
        self.config = config

    def evaluate(
        self,
        test_set: list[PromptPair],
        optimized_prompt: str | None = None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []

        for pair in test_set:
            # Generate response from target model
            prompt_content = pair.prompt if isinstance(pair.prompt, str) else pair.prompt
            messages = (
                [{"role": "user", "content": prompt_content}]
                if isinstance(prompt_content, str)
                else prompt_content
            )

            if optimized_prompt:
                messages = [{"role": "system", "content": optimized_prompt}] + messages

            response = litellm.completion(
                model=self.config.target_model,
                messages=messages,
            )
            new_response = response.choices[0].message.content or ""

            # Score
            scores = self._score(pair.response, new_response, pair.output_type)
            composite = self._composite_score(scores)

            results.append(
                EvalResult(
                    prompt_pair=pair,
                    new_response=new_response,
                    scores=scores,
                    composite_score=composite,
                    is_win=composite >= WIN_THRESHOLD,
                )
            )

        return results

    def _score(
        self, expected: str, actual: str, output_type: OutputType | None
    ) -> dict[str, float]:
        if output_type is None:
            output_type = detect_output_type(expected)

        scores: dict[str, float] = {}

        if output_type == OutputType.JSON:
            scores.update(JSONEvaluator().score(expected, actual))
        elif output_type == OutputType.CLASSIFICATION:
            scores.update(ExactMatchEvaluator().score(expected, actual))
        else:
            # Free text — try BERTScore, fall back to embedding, then string sim
            try:
                from rosettastone.evaluate.bertscore import BERTScoreEvaluator

                scores.update(BERTScoreEvaluator().score(expected, actual))
            except ImportError:
                try:
                    from rosettastone.evaluate.embedding import EmbeddingEvaluator

                    scores.update(EmbeddingEvaluator().score(expected, actual))
                except ImportError:
                    scores.update(ExactMatchEvaluator().score(expected, actual))

        return scores

    def _composite_score(self, scores: dict[str, float]) -> float:
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)
