"""GEPA optimizer wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import dspy

from rosettastone.optimize.base import Optimizer
from rosettastone.optimize.dspy_program import MigrationProgram
from rosettastone.optimize.metric import IterationTracker, build_migration_metric

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair


class InstructionExtractionError(Exception):
    """Raised when optimized instructions cannot be extracted from a compiled DSPy program."""


class GEPAOptimizer(Optimizer):
    def optimize(
        self,
        train_set: list[PromptPair],
        val_set: list[PromptPair],
        config: MigrationConfig,
        on_iteration: Callable[[int, int, float], None] | None = None,
    ) -> str:
        # Configure LMs
        target_lm = dspy.LM(config.target_model)
        reflection_lm = dspy.LM(config.reflection_model, temperature=1.0, max_tokens=16000)

        # Build DSPy program
        program = MigrationProgram()

        # Build metric (with feedback map if training data has feedback)
        metric = build_migration_metric(config, train_set=train_set)

        # Convert to DSPy Examples
        trainset = [
            dspy.Example(prompt=p.prompt, expected_response=p.response).with_inputs("prompt")
            for p in train_set
        ]

        # Wrap metric with iteration tracker if a callback was provided
        if on_iteration is not None and len(trainset) > 0:
            # GEPA "light"=25, "medium"=50, "heavy"=100 iterations by default.
            # We derive total_iterations from the auto setting as a best-effort estimate;
            # the tracker fires on every trainset_size calls regardless of the exact count.
            _auto_iterations = {"light": 25, "medium": 50, "heavy": 100}
            total_iterations = _auto_iterations.get(config.gepa_auto, 25)
            tracker = IterationTracker(
                trainset_size=len(trainset),
                total_iterations=total_iterations,
                callback=on_iteration,
            )
            metric = tracker.wrap(metric)

        # Run GEPA
        with dspy.context(lm=target_lm):
            optimizer = dspy.GEPA(
                metric=metric,
                auto=config.gepa_auto,
                reflection_lm=reflection_lm,
                num_threads=config.num_threads,
            )
            compiled = optimizer.compile(program, trainset=trainset)

        from rosettastone.optimize.utils import extract_optimized_instructions

        return extract_optimized_instructions(compiled)
