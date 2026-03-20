"""MIPROv2 optimizer wrapper — fallback optimizer for RosettaStone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from rosettastone.optimize.base import Optimizer
from rosettastone.optimize.dspy_program import MigrationProgram
from rosettastone.optimize.metric import build_migration_metric
from rosettastone.optimize.utils import extract_optimized_instructions

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair


class MIPROv2Optimizer(Optimizer):
    """DSPy MIPROv2-based optimizer.

    Runs in zero-shot mode (max_bootstrapped_demos=0, max_labeled_demos=0) so no
    production prompt/response content appears in demo slots — avoiding PII leakage.

    When config.mipro_auto is None the optimizer falls back to the "light" auto preset.
    Callers should prefer GEPA when reflection is available; use MIPROv2 as a fallback.
    """

    def optimize(
        self,
        train_set: list[PromptPair],
        val_set: list[PromptPair],
        config: MigrationConfig,
    ) -> str:
        # Configure target LM
        target_lm = dspy.LM(config.target_model)

        # Build DSPy program
        program = MigrationProgram()

        # Build metric
        metric = build_migration_metric(config)

        # Convert to DSPy Examples — zero-shot: no demos from production data (PII safety)
        trainset = [
            dspy.Example(prompt=p.prompt, expected_response=p.response).with_inputs("prompt")
            for p in train_set
        ]

        # Run MIPROv2 in zero-shot mode
        auto_preset = config.mipro_auto if config.mipro_auto is not None else "light"
        with dspy.context(lm=target_lm):
            optimizer = dspy.MIPROv2(
                metric=metric,
                auto=auto_preset,
                max_bootstrapped_demos=0,
                max_labeled_demos=0,
                num_threads=config.num_threads,
            )
            compiled = optimizer.compile(program, trainset=trainset)

        return extract_optimized_instructions(compiled)
