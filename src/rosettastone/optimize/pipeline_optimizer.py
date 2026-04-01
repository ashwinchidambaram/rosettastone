"""Dynamic multi-module DSPy pipeline builder and optimizer.

Builds a dspy.Module from a PipelineConfig by dynamically constructing
Signatures for each pipeline module and executing them in topological order.
GEPA optimizes all modules end-to-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from rosettastone.optimize.pipeline_config import PipelineConfig, validate_dag

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair


def _build_signature(name: str, input_fields: list[str], output_fields: list[str]) -> type:
    """Dynamically build a dspy.Signature class for a pipeline module.

    Creates a Signature with the specified input/output fields, suitable for
    use with dspy.ChainOfThought or dspy.Predict.
    """
    fields: dict[str, Any] = {"__doc__": f"Signature for pipeline module '{name}'."}
    for field in input_fields:
        fields[field] = dspy.InputField(desc=f"Input: {field}")
    for field in output_fields:
        fields[field] = dspy.OutputField(desc=f"Output: {field}")

    return type(f"{name}Signature", (dspy.Signature,), fields)  # type: ignore[return-value]


class PipelineProgram(dspy.Module):  # type: ignore[misc]
    """A multi-module DSPy program built dynamically from a PipelineConfig.

    Each module in the pipeline config becomes a dspy.ChainOfThought predictor.
    forward() executes modules in topological order, passing outputs from
    upstream modules as inputs to downstream modules.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.execution_order = validate_dag(config)
        self.module_configs = {m.name: m for m in config.modules}

        # Build a ChainOfThought predictor for each module
        self.predictors: dict[str, dspy.ChainOfThought] = {}  # type: ignore[type-arg]
        for module_cfg in config.modules:
            sig = _build_signature(
                module_cfg.name, module_cfg.input_fields, module_cfg.output_fields
            )
            predictor = dspy.ChainOfThought(sig)
            self.predictors[module_cfg.name] = predictor
            # Register as a named sub-module so GEPA can optimize it
            setattr(self, f"predict_{module_cfg.name}", predictor)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """Execute the pipeline in topological order.

        Args:
            **kwargs: Initial inputs for the pipeline (raw pipeline inputs
                      that aren't produced by any module).

        Returns:
            dspy.Prediction with all accumulated outputs from all modules.
        """
        # Accumulate all field values as we execute modules
        context: dict[str, Any] = dict(kwargs)

        for module_name in self.execution_order:
            module_cfg = self.module_configs[module_name]
            predictor = self.predictors[module_name]

            # Gather inputs for this module from the accumulated context
            module_inputs = {}
            for field in module_cfg.input_fields:
                if field in context:
                    module_inputs[field] = context[field]
                else:
                    module_inputs[field] = ""  # Missing input defaults to empty

            # Execute the predictor
            result = predictor(**module_inputs)

            # Collect outputs into the shared context
            for field in module_cfg.output_fields:
                if hasattr(result, field):
                    context[field] = getattr(result, field)

        return dspy.Prediction(**context)


def optimize_pipeline(
    config: PipelineConfig,
    train_set: list[PromptPair],
    migration_config: MigrationConfig,
) -> dict[str, str]:
    """Optimize a multi-module pipeline using GEPA.

    Returns a dict mapping module_name -> optimized instructions for each module.
    """
    target_lm = dspy.LM(config.target_model)
    reflection_lm = dspy.LM(config.reflection_model, temperature=1.0, max_tokens=16000)

    program = PipelineProgram(config)

    # Build metric
    from rosettastone.optimize.metric import build_migration_metric

    metric = build_migration_metric(migration_config, train_set=train_set)

    # Convert training data to DSPy Examples
    # For pipeline mode, the first module's input field is the prompt
    first_module = config.modules[0]
    primary_input = first_module.input_fields[0] if first_module.input_fields else "prompt"

    trainset = [
        dspy.Example(**{primary_input: p.prompt, "expected_response": p.response}).with_inputs(
            primary_input
        )
        for p in train_set
    ]

    # Run GEPA optimization on the full pipeline
    with dspy.context(lm=target_lm):
        optimizer = dspy.GEPA(
            metric=metric,
            auto=config.gepa_auto,
            reflection_lm=reflection_lm,
            num_threads=migration_config.num_threads,
        )
        compiled = optimizer.compile(program, trainset=trainset)

    # Extract per-module optimized instructions
    from rosettastone.optimize.utils import extract_optimized_instructions

    results: dict[str, str] = {}
    for module_name in config.modules:
        predictor_attr = f"predict_{module_name.name}"
        if hasattr(compiled, predictor_attr):
            predictor = getattr(compiled, predictor_attr)
            try:
                # Extract instructions from the compiled predictor
                if hasattr(predictor, "extended_signature"):
                    instructions = predictor.extended_signature.instructions
                    results[module_name.name] = str(instructions)
                else:
                    results[module_name.name] = extract_optimized_instructions(compiled)
            except Exception:
                results[module_name.name] = ""
        else:
            results[module_name.name] = ""

    return results
