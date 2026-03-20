"""Shared utilities for optimizer implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy


class InstructionExtractionError(Exception):
    """Raised when optimized instructions cannot be extracted from a compiled DSPy program."""


def extract_optimized_instructions(compiled: dspy.Module) -> str:  # type: ignore[name-defined]
    """Extract the optimized prompt instructions from a compiled DSPy program.

    Checks the primary path (compiled.predict.signature.instructions) first,
    then falls back to named_predictors() iteration.
    """
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
        "Check that dspy>=2.6 is installed."
    )
