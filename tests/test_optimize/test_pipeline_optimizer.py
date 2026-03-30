"""Tests for pipeline_optimizer.py — PipelineProgram and optimize_pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy

from rosettastone.optimize.pipeline_config import PipelineConfig, PipelineModuleConfig
from rosettastone.optimize.pipeline_optimizer import PipelineProgram, _build_signature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_config() -> PipelineConfig:
    """Two-module linear pipeline: step1 → step2."""
    return PipelineConfig(
        name="test_pipeline",
        modules=[
            PipelineModuleConfig(
                name="step1",
                prompt_template="You are helpful.",
                input_fields=["user_query"],
                output_fields=["step1_result"],
                depends_on=[],
            ),
            PipelineModuleConfig(
                name="step2",
                prompt_template="Refine the result.",
                input_fields=["step1_result"],
                output_fields=["final_answer"],
                depends_on=["step1"],
            ),
        ],
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-haiku-4-5",
    )


def _make_mock_migration_config() -> MagicMock:
    cfg = MagicMock()
    cfg.source_model = "openai/gpt-4o"
    cfg.target_model = "anthropic/claude-haiku-4-5"
    cfg.num_threads = 1
    return cfg


# ---------------------------------------------------------------------------
# _build_signature tests
# ---------------------------------------------------------------------------


class TestBuildSignature:
    def test_build_signature_creates_fields(self) -> None:
        """Returned type has the specified input and output fields."""
        sig_cls = _build_signature("test_mod", ["input_a", "input_b"], ["output_x"])

        # It should be a subclass of dspy.Signature
        assert issubclass(sig_cls, dspy.Signature)

        # Field names should be present
        field_names = set(sig_cls.fields.keys())
        assert "input_a" in field_names
        assert "input_b" in field_names
        assert "output_x" in field_names

    def test_build_signature_docstring(self) -> None:
        """Signature docstring contains the module name."""
        sig_cls = _build_signature("my_module", ["x"], ["y"])
        assert "my_module" in sig_cls.__doc__

    def test_build_signature_empty_fields(self) -> None:
        """Signature with no inputs/outputs can still be created."""
        sig_cls = _build_signature("empty_mod", [], [])
        assert issubclass(sig_cls, dspy.Signature)


# ---------------------------------------------------------------------------
# PipelineProgram tests
# ---------------------------------------------------------------------------


class TestPipelineProgram:
    def test_pipeline_program_init(self) -> None:
        """PipelineProgram creates predictors for each module."""
        config = _make_linear_config()
        with patch("dspy.ChainOfThought") as mock_cot:
            mock_cot.return_value = MagicMock()
            program = PipelineProgram(config)

        assert len(program.predictors) == 2
        assert "step1" in program.predictors
        assert "step2" in program.predictors

    def test_pipeline_program_sets_predict_attrs(self) -> None:
        """Each module has a predict_{name} attribute set on the program."""
        config = _make_linear_config()
        with patch("dspy.ChainOfThought") as mock_cot:
            mock_cot.return_value = MagicMock()
            program = PipelineProgram(config)

        assert hasattr(program, "predict_step1")
        assert hasattr(program, "predict_step2")

    def test_pipeline_program_forward_returns_prediction(self) -> None:
        """program.forward() returns a dspy.Prediction with accumulated context."""
        config = _make_linear_config()

        # Mock step1 predictor: returns step1_result
        mock_step1_result = MagicMock()
        mock_step1_result.step1_result = "intermediate output"

        # Mock step2 predictor: returns final_answer
        mock_step2_result = MagicMock()
        mock_step2_result.final_answer = "final output"

        with patch("dspy.ChainOfThought") as mock_cot:
            step1_predictor = MagicMock(return_value=mock_step1_result)
            step2_predictor = MagicMock(return_value=mock_step2_result)

            # First call builds step1 signature, second builds step2
            mock_cot.side_effect = [step1_predictor, step2_predictor]

            program = PipelineProgram(config)
            result = program.forward(user_query="What is AI?")

        assert isinstance(result, dspy.Prediction)

    def test_pipeline_program_forward_missing_input_defaults_empty(self) -> None:
        """Missing input field defaults to '' rather than raising."""
        config = PipelineConfig(
            name="single_step",
            modules=[
                PipelineModuleConfig(
                    name="mod",
                    prompt_template="Template",
                    input_fields=["missing_field"],
                    output_fields=["out"],
                )
            ],
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-haiku-4-5",
        )

        captured_inputs: list[dict] = []

        def fake_predictor(**kwargs):  # type: ignore[no-untyped-def]
            captured_inputs.append(kwargs)
            result = MagicMock()
            result.out = "ok"
            return result

        with patch("dspy.ChainOfThought") as mock_cot:
            mock_cot.return_value = fake_predictor
            program = PipelineProgram(config)
            # Forward with NO matching field
            program.forward(some_other_field="value")

        assert len(captured_inputs) == 1
        assert captured_inputs[0]["missing_field"] == ""

    def test_pipeline_program_execution_order(self) -> None:
        """Modules execute in topological order (step1 before step2)."""
        config = _make_linear_config()
        call_order: list[str] = []

        step1_result = MagicMock()
        step1_result.step1_result = "from_step1"

        step2_result = MagicMock()
        step2_result.final_answer = "from_step2"

        def make_predictor(name: str, mock_result: MagicMock) -> MagicMock:
            def pred(**kwargs):  # type: ignore[no-untyped-def]
                call_order.append(name)
                return mock_result

            return pred  # type: ignore[return-value]

        with patch("dspy.ChainOfThought") as mock_cot:
            mock_cot.side_effect = [
                make_predictor("step1", step1_result),
                make_predictor("step2", step2_result),
            ]
            program = PipelineProgram(config)
            program.forward(user_query="hello")

        assert call_order == ["step1", "step2"]


# ---------------------------------------------------------------------------
# optimize_pipeline tests
# ---------------------------------------------------------------------------


class TestOptimizePipeline:
    def test_optimize_pipeline_calls_gepa(self) -> None:
        """optimize_pipeline calls GEPA.compile on the program."""
        from rosettastone.optimize.pipeline_optimizer import optimize_pipeline

        config = _make_linear_config()
        migration_config = _make_mock_migration_config()

        mock_compiled = MagicMock()
        # Give the compiled program the expected predictor attributes
        mock_compiled.predict_step1 = MagicMock()
        mock_compiled.predict_step1.extended_signature = MagicMock(instructions="step1 instr")
        mock_compiled.predict_step2 = MagicMock()
        mock_compiled.predict_step2.extended_signature = MagicMock(instructions="step2 instr")

        mock_gepa_instance = MagicMock()
        mock_gepa_instance.compile.return_value = mock_compiled

        with (
            patch("dspy.LM", return_value=MagicMock()),
            patch("dspy.GEPA", return_value=mock_gepa_instance),
            patch("dspy.context"),
            patch("rosettastone.optimize.metric.build_migration_metric", return_value=MagicMock()),
        ):
            optimize_pipeline(config, [], migration_config)

        mock_gepa_instance.compile.assert_called_once()

    def test_optimize_pipeline_returns_dict(self) -> None:
        """optimize_pipeline returns a dict keyed by module names."""
        from rosettastone.optimize.pipeline_optimizer import optimize_pipeline

        config = _make_linear_config()
        migration_config = _make_mock_migration_config()

        mock_compiled = MagicMock()
        mock_compiled.predict_step1 = MagicMock()
        mock_compiled.predict_step1.extended_signature = MagicMock(
            instructions="optimized step1"
        )
        mock_compiled.predict_step2 = MagicMock()
        mock_compiled.predict_step2.extended_signature = MagicMock(
            instructions="optimized step2"
        )

        mock_gepa_instance = MagicMock()
        mock_gepa_instance.compile.return_value = mock_compiled

        with (
            patch("dspy.LM", return_value=MagicMock()),
            patch("dspy.GEPA", return_value=mock_gepa_instance),
            patch("dspy.context"),
            patch("rosettastone.optimize.metric.build_migration_metric", return_value=MagicMock()),
        ):
            result = optimize_pipeline(config, [], migration_config)

        assert isinstance(result, dict)
        assert "step1" in result
        assert "step2" in result
        for val in result.values():
            assert isinstance(val, str)
