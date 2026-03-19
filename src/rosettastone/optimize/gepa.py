"""GEPA optimizer wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from rosettastone.optimize.base import Optimizer
from rosettastone.optimize.dspy_program import MigrationProgram
from rosettastone.optimize.metric import build_migration_metric

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair


class GEPAOptimizer(Optimizer):
    def optimize(
        self,
        train_set: list[PromptPair],
        val_set: list[PromptPair],
        config: MigrationConfig,
    ) -> str:
        # Configure LMs
        target_lm = dspy.LM(config.target_model)
        reflection_lm = dspy.LM(
            config.reflection_model, temperature=1.0, max_tokens=16000
        )

        # Build DSPy program
        program = MigrationProgram()

        # Build metric
        metric = build_migration_metric(config)

        # Convert to DSPy Examples
        trainset = [
            dspy.Example(
                prompt=p.prompt, expected_response=p.response
            ).with_inputs("prompt")
            for p in train_set
        ]

        # Run GEPA
        with dspy.context(lm=target_lm):
            optimizer = dspy.GEPA(
                metric=metric,
                auto=config.gepa_auto,
                reflection_lm=reflection_lm,
                num_threads=config.num_threads,
            )
            compiled = optimizer.compile(program, trainset=trainset)

        return _extract_optimized_instructions(compiled)


def _extract_optimized_instructions(compiled: dspy.Module) -> str:
    """Extract the optimized prompt instructions from a compiled DSPy program."""
    # GEPA stores optimized instructions in the predict module's signature
    if hasattr(compiled, "predict") and hasattr(compiled.predict, "signature"):
        sig = compiled.predict.signature
        if hasattr(sig, "instructions"):
            return str(sig.instructions)

    # Fallback: inspect the compiled program's state
    for name, module in compiled.named_predictors():
        if hasattr(module, "signature") and hasattr(module.signature, "instructions"):
            return str(module.signature.instructions)

    return "Could not extract optimized instructions from compiled program."
