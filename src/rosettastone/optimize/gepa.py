"""GEPA optimizer wrapper."""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import dspy

from rosettastone.optimize.base import Optimizer
from rosettastone.optimize.dspy_program import MigrationProgram
from rosettastone.optimize.metric import IterationTracker, build_migration_metric
from rosettastone.optimize.utils import InstructionExtractionError, extract_optimized_instructions

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair

logger = logging.getLogger(__name__)

# Re-export for backward compatibility — callers import InstructionExtractionError from gepa.py
__all__ = ["GEPAOptimizer", "GEPATimeoutWithResult", "InstructionExtractionError"]


class GEPATimeoutWithResult(Exception):  # noqa: N818
    """Raised when GEPA times out but has a usable intermediate result."""

    def __init__(self, instructions: str, message: str) -> None:
        self.instructions = instructions
        self.message = message
        super().__init__(message)


class GEPAOptimizer(Optimizer):
    def optimize(
        self,
        train_set: list[PromptPair],
        val_set: list[PromptPair],
        config: MigrationConfig,
        on_iteration: Callable[[int, int, float], None] | None = None,
    ) -> str:
        # Configure LMs — merge any provider-specific extra kwargs from config
        extra_kwargs: dict[str, object] = (
            dict(config.lm_extra_kwargs) if config.lm_extra_kwargs else {}
        )
        target_lm = dspy.LM(config.target_model, **extra_kwargs)
        # Filter out keys that conflict with reflection_lm's explicit temperature/max_tokens args
        _conflict_keys = ("temperature", "max_tokens")
        reflection_extra = {k: v for k, v in extra_kwargs.items() if k not in _conflict_keys}
        reflection_lm = dspy.LM(
            config.reflection_model, temperature=1.0, max_tokens=16000, **reflection_extra
        )

        # Build DSPy program
        program = MigrationProgram()

        # Build metric (with feedback map if training data has feedback)
        metric = build_migration_metric(config, train_set=train_set)

        # Convert to DSPy Examples
        trainset = [
            dspy.Example(prompt=p.prompt, expected_response=p.response).with_inputs("prompt")
            for p in train_set
        ]

        # best_intermediate accumulates the most recent extractable instructions after each
        # iteration fires. Using a list so the thread's closure can append to it safely.
        best_intermediate: list[str] = []

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
            base_metric = tracker.wrap(metric)
        else:
            base_metric = metric

        def _iteration_capturing_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            """Wraps the base metric and captures program state after each call."""
            result = base_metric(gold, pred, trace, pred_name, pred_trace)
            # After each metric call, try to snapshot the current program state.
            # Extraction failure must not interrupt the metric callback.
            try:
                snapshot = extract_optimized_instructions(program)
                best_intermediate.append(snapshot)
            except Exception as e:
                logger.debug("Snapshot extraction failed: %s", type(e).__name__)
            return result

        # Run GEPA — use explicit max_metric_calls if provided, otherwise use auto preset
        gepa_max_metric_calls = getattr(config, "gepa_max_metric_calls", None)
        gepa_kwargs: dict[str, object]
        if gepa_max_metric_calls is not None:
            gepa_kwargs = {
                "metric": _iteration_capturing_metric,
                "max_metric_calls": gepa_max_metric_calls,
                "reflection_lm": reflection_lm,
                "num_threads": config.num_threads,
            }
        else:
            gepa_kwargs = {
                "metric": _iteration_capturing_metric,
                "auto": config.gepa_auto,
                "reflection_lm": reflection_lm,
                "num_threads": config.num_threads,
            }

        timeout: int | None = getattr(config, "gepa_timeout_seconds", None)

        def _run_gepa() -> dspy.Module:
            with dspy.context(lm=target_lm):
                optimizer = dspy.GEPA(**gepa_kwargs)
                return optimizer.compile(program, trainset=trainset)

        if timeout is not None:
            # Safety-net path: wrap in executor with hard timeout.
            # Used when the caller explicitly sets gepa_timeout_seconds (e.g. CI pipelines).
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_gepa)
                try:
                    compiled = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    if best_intermediate:
                        logger.warning(
                            "GEPA timed out after %ds — using best intermediate result.", timeout
                        )
                        # best_intermediate is appended inside _iteration_capturing_metric which
                        # runs in the GEPA thread. The GIL protects the list.append() itself, but
                        # reads of program state inside extract_optimized_instructions are
                        # best-effort snapshots (not a guaranteed consistent state).
                        instructions = best_intermediate[-1]
                        raise GEPATimeoutWithResult(
                            instructions=instructions,
                            message=(
                                f"GEPA timed out after {timeout}s — using best intermediate "
                                f"result. Consider increasing gepa_timeout_seconds or reducing "
                                f"gepa_auto complexity."
                            ),
                        )
                    raise TimeoutError(
                        f"GEPA timed out after {timeout}s with no intermediate result"
                    )
        else:
            # Default path: run GEPA directly — no external timeout.
            # GEPA terminates naturally when it exhausts its max_metric_calls budget.
            compiled = _run_gepa()

        return extract_optimized_instructions(compiled)
