"""Shared utilities for optimizers."""

from __future__ import annotations

import dspy


class InstructionExtractionError(Exception):
    """Raised when optimized instructions cannot be extracted from a compiled DSPy program."""


def extract_optimized_instructions(compiled: dspy.Module) -> str:
    """Extract the optimized prompt instructions from a compiled DSPy program.

    Used by both GEPA and MIPROv2 optimizers.
    """
    # GEPA/MIPROv2 store optimized instructions in the predict module's signature
    if hasattr(compiled, "predict") and hasattr(compiled.predict, "signature"):
        sig = compiled.predict.signature
        if hasattr(sig, "instructions"):
            return str(sig.instructions)

    # Fallback: inspect the compiled program's state via named_predictors
    for _name, module in compiled.named_predictors():
        if hasattr(module, "signature") and hasattr(module.signature, "instructions"):
            return str(module.signature.instructions)

    raise InstructionExtractionError(
        "Could not extract optimized instructions from compiled program. "
        "This may indicate a DSPy version incompatibility. "
        "Check that dspy>=3.1 is installed."
    )
