"""Teacher/Student bootstrapper for multi-step pipeline migration.

The Teacher phase runs the source model on training data to generate demonstrations.
The Student phase uses GEPA to optimize the target model using those demonstrations
as the metric target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rosettastone.optimize.base import Optimizer
from rosettastone.optimize.pipeline_optimizer import optimize_pipeline

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair
    from rosettastone.optimize.pipeline_config import PipelineConfig


class TeacherStudentOptimizer(Optimizer):
    """Optimizer that uses a Teacher/Student bootstrapping strategy for pipelines.

    The Teacher phase calls the source LLM on each training example to generate
    demonstrations. The Student phase then runs GEPA to optimize the target model's
    instructions using those teacher demonstrations as the metric target.

    This class is pipeline-only. Use ``pipeline_optimize()`` instead of ``optimize()``.
    """

    def optimize(
        self,
        train_set: list[PromptPair],
        val_set: list[PromptPair],
        config: MigrationConfig,
    ) -> str:
        """Not supported — use ``pipeline_optimize()`` for pipeline migrations."""
        raise NotImplementedError(
            "TeacherStudentOptimizer does not support single-prompt optimization. "
            "Use pipeline_optimize() for pipeline migrations."
        )

    def generate_teacher_demos(
        self,
        train_set: list[PromptPair],
        source_model: str,
        pipeline_config: PipelineConfig,
    ) -> list[PromptPair]:
        """Run the source LLM on each training prompt to generate demonstrations.

        Uses the first input field of the first pipeline module as the prompt field.
        Never logs prompt content (PII risk).

        Args:
            train_set: Training prompt/response pairs from production data.
            source_model: LiteLLM model string for the source (teacher) model.
            pipeline_config: Pipeline configuration defining module structure.

        Returns:
            List of PromptPair with the teacher model's responses.
        """
        import dspy

        from rosettastone.core.types import PromptPair

        if not train_set:
            return []

        # Determine the primary prompt field from the first module's first input field
        first_module = pipeline_config.modules[0]
        primary_input = first_module.input_fields[0] if first_module.input_fields else "prompt"  # noqa: F841

        # Instantiate the teacher (source) LM
        teacher_lm = dspy.LM(source_model)

        demos: list[PromptPair] = []
        for pair in train_set:
            # Build a minimal chat message from the prompt
            prompt_text = pair.prompt if isinstance(pair.prompt, str) else str(pair.prompt)
            messages = [{"role": "user", "content": prompt_text}]

            # Call the teacher LM — DSPy handles retries/rate limiting
            raw_response = teacher_lm(messages=messages)

            # dspy.LM returns a list of completion strings
            if isinstance(raw_response, list):
                teacher_response = raw_response[0] if raw_response else ""
            else:
                teacher_response = str(raw_response)

            demos.append(
                PromptPair(
                    prompt=pair.prompt,
                    response=teacher_response,
                    source_model=source_model,
                    metadata={"teacher_bootstrapped": True},
                )
            )

        return demos

    def pipeline_optimize(
        self,
        pipeline_config: PipelineConfig,
        train_set: list[PromptPair],
        migration_config: MigrationConfig,
    ) -> dict[str, str]:
        """Optimize a multi-module pipeline using Teacher/Student bootstrapping.

        1. Teacher phase: call the source model on each training example to
           generate demonstrations.
        2. Student phase: run ``optimize_pipeline`` (GEPA) using the teacher
           demonstrations as training signal.

        Args:
            pipeline_config: Pipeline configuration with module definitions.
            train_set: Production prompt/response pairs to learn from.
            migration_config: Migration settings (models, GEPA config, etc.).

        Returns:
            A dict mapping module_name -> optimized instruction string.
        """
        # Teacher phase: generate demonstrations from the source model
        teacher_demos = self.generate_teacher_demos(
            train_set, migration_config.source_model, pipeline_config
        )

        # Student phase: GEPA optimizes target model using teacher demos as signal
        return optimize_pipeline(pipeline_config, teacher_demos, migration_config)
