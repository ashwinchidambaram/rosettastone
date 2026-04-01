"""Composite evaluator: combines metrics by output type, computes pairwise win rate."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import litellm

from rosettastone.core.types import EvalResult, OutputType, PromptPair
from rosettastone.evaluate.exact_match import ExactMatchEvaluator
from rosettastone.evaluate.json_validator import JSONEvaluator
from rosettastone.evaluate.types import detect_output_type
from rosettastone.utils.logging import get_logger

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig

logger = get_logger("evaluate.composite")


# Default thresholds per output type (used when config.win_thresholds missing a key)
DEFAULT_WIN_THRESHOLDS: dict[str, float] = {
    "json": 0.95,
    "classification": 0.90,
    "short_text": 0.80,
    "long_text": 0.75,
}

# Routing table: which evaluators run for which output type
EVALUATOR_ROUTING: dict[OutputType, list[str]] = {
    OutputType.JSON: ["json_validator", "json_structural"],
    OutputType.CLASSIFICATION: ["exact_match"],
    OutputType.SHORT_TEXT: ["semantic"],
    OutputType.LONG_TEXT: ["semantic", "llm_judge"],
}

# Weights for composite score by metric category
METRIC_WEIGHTS: dict[str, float] = {
    # JSON metrics
    "json_valid": 0.0,  # gating — handled separately
    "json_field_match": 0.4,
    "json_structural_sim": 0.4,
    "json_schema_match": 0.2,
    # Classification
    "exact_match": 0.7,
    "string_similarity": 0.3,
    # Semantic / text
    "bertscore_f1": 0.5,
    "embedding_sim": 0.5,
    # LLM judge
    "llm_judge_score": 0.3,
}


class CompositeEvaluator:
    def __init__(
        self,
        config: MigrationConfig,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.on_progress = on_progress

    def evaluate(
        self,
        test_set: list[PromptPair],
        optimized_prompt: str | None = None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []

        for i, pair in enumerate(test_set):
            # Build messages from prompt
            if isinstance(pair.prompt, str):
                messages: list[dict[str, str]] = [{"role": "user", "content": pair.prompt}]
            else:
                messages = pair.prompt  # type: ignore[assignment,unused-ignore]

            if optimized_prompt:
                messages = [{"role": "system", "content": optimized_prompt}] + messages

            # Generate response from target model
            try:
                response = litellm.completion(
                    model=self.config.target_model,
                    messages=messages,
                )
                if not response.choices:
                    logger.warning("Pair %d: empty choices in response, skipping", i)
                    continue
                new_response = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(
                    "Pair %d: litellm.completion failed (%s), skipping",
                    i,
                    type(e).__name__,  # log only the exception type, not the message
                )
                continue

            # Detect output type
            output_type = pair.output_type or detect_output_type(pair.response)

            # Score with prompt kwarg for evaluators that accept it
            scores = self._score(pair.response, new_response, output_type, prompt=pair.prompt)
            composite = self._composite_score(scores, output_type)

            # Get threshold for this output type
            threshold = self._get_threshold(output_type)

            results.append(
                EvalResult(
                    prompt_pair=pair,
                    new_response=new_response,
                    scores=scores,
                    composite_score=composite,
                    is_win=composite >= threshold,
                    details={
                        "output_type": output_type.value,
                        "evaluators_used": list(scores.keys()),
                        "threshold": threshold,
                    },
                )
            )

            if self.on_progress:
                self.on_progress(i + 1, len(test_set))

        return results

    def _get_threshold(self, output_type: OutputType) -> float:
        thresholds = getattr(self.config, "win_thresholds", DEFAULT_WIN_THRESHOLDS)
        return cast(
            float,
            thresholds.get(output_type.value, DEFAULT_WIN_THRESHOLDS.get(output_type.value, 0.8)),
        )

    def _score(
        self,
        expected: str,
        actual: str,
        output_type: OutputType,
        prompt: str | list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        prompt_str = prompt if isinstance(prompt, str) else None

        if output_type == OutputType.JSON:
            scores.update(JSONEvaluator(config=self.config).score(expected, actual))

            # JSON structural evaluator (Phase 2 addition — lazy import)
            try:
                from rosettastone.evaluate.json_structural import JSONStructuralEvaluator

                scores.update(JSONStructuralEvaluator(config=self.config).score(expected, actual))
            except ImportError:
                pass

        elif output_type == OutputType.CLASSIFICATION:
            scores.update(ExactMatchEvaluator(config=self.config).score(expected, actual))
        else:
            # Free text — try BERTScore, fall back to embedding, then string sim
            scores.update(self._score_semantic(expected, actual))

        # LLM judge for long text (and optionally all types) — Phase 2
        if output_type == OutputType.LONG_TEXT and not getattr(self.config, "local_only", False):
            try:
                from rosettastone.evaluate.llm_judge import LLMJudgeEvaluator

                judge_scores = LLMJudgeEvaluator(config=self.config).score(
                    expected, actual, prompt=prompt_str
                )
                scores.update(judge_scores)
            except ImportError:
                pass

        return scores

    def _score_semantic(self, expected: str, actual: str) -> dict[str, float]:
        local_only = getattr(self.config, "local_only", False)

        try:
            from rosettastone.evaluate.bertscore import BERTScoreEvaluator

            return BERTScoreEvaluator(config=self.config).score(expected, actual)
        except ImportError:
            if local_only:
                logger.warning(
                    "BERTScore not available in local_only mode; falling back to string similarity"
                )
            try:
                from rosettastone.evaluate.embedding import EmbeddingEvaluator

                return EmbeddingEvaluator(config=self.config).score(expected, actual)
            except ImportError:
                return ExactMatchEvaluator(config=self.config).score(expected, actual)

    def _composite_score(self, scores: dict[str, float], output_type: OutputType) -> float:
        if not scores:
            return 0.0

        # Gated scoring: if JSON and json_valid == 0, composite is 0.0
        if output_type == OutputType.JSON and scores.get("json_valid", 1.0) == 0.0:
            return 0.0

        # Weighted composite
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, value in scores.items():
            # Skip gating metrics from weighted average
            if metric == "json_valid":
                continue
            weight = METRIC_WEIGHTS.get(metric, 1.0)
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        return weighted_sum / total_weight
