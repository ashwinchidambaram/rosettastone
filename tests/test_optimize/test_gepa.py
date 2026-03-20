"""Tests for GEPAOptimizer and _extract_optimized_instructions in optimize/gepa.py.

dspy.GEPA and dspy.LM are mocked entirely — no API calls, no model downloads,
no money spent. The tests verify the optimizer's wiring and instruction extraction
logic, not DSPy's internal behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.core.types import OutputType, PromptPair
from rosettastone.optimize.gepa import (
    GEPAOptimizer,
    InstructionExtractionError,
    _extract_optimized_instructions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path) -> MigrationConfig:
    data_file = tmp_path / "data.jsonl"
    data_file.touch()
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_file,
    )


def _make_pairs(n: int = 2) -> list[PromptPair]:
    return [
        PromptPair(
            prompt=f"prompt {i}",
            response=f"response {i}",
            source_model="openai/gpt-4o",
            output_type=OutputType.SHORT_TEXT,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _extract_optimized_instructions — primary path
# ---------------------------------------------------------------------------


class TestExtractOptimizedInstructionsPrimaryPath:
    """Instructions should be read from compiled.predict.signature.instructions."""

    def test_extracts_from_predict_signature_instructions(self) -> None:
        """Primary path: compiled.predict.signature.instructions returns the string directly."""
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Optimized: be concise"

        result = _extract_optimized_instructions(mock_compiled)

        assert result == "Optimized: be concise", (
            f"Expected 'Optimized: be concise', got {result!r}"
        )

    def test_result_is_string(self) -> None:
        """Return value must always be a string — callers pass it as prompt instructions."""
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Some instructions"

        result = _extract_optimized_instructions(mock_compiled)

        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

    def test_coerces_non_string_instructions_to_string(self) -> None:
        """If instructions is not a str (e.g. a DSPy object), str() coerces it."""
        mock_compiled = MagicMock()
        # Simulate a non-string instructions value that has a useful __str__
        mock_instructions = MagicMock()
        mock_instructions.__str__ = lambda self: "Coerced instructions"
        mock_compiled.predict.signature.instructions = mock_instructions

        result = _extract_optimized_instructions(mock_compiled)

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _extract_optimized_instructions — fallback path
# ---------------------------------------------------------------------------


class TestExtractOptimizedInstructionsFallback:
    """When predict.signature is absent, named_predictors() is the fallback."""

    def test_fallback_via_named_predictors(self) -> None:
        """named_predictors() fallback: first predictor with .signature.instructions wins."""
        mock_compiled = MagicMock(spec=["named_predictors"])
        predictor = MagicMock()
        predictor.signature.instructions = "Fallback instructions"
        mock_compiled.named_predictors.return_value = [("predict", predictor)]

        result = _extract_optimized_instructions(mock_compiled)

        assert result == "Fallback instructions", f"Expected fallback instructions, got {result!r}"

    def test_fallback_uses_first_matching_predictor(self) -> None:
        """When multiple predictors exist, the first with instructions is used."""
        mock_compiled = MagicMock(spec=["named_predictors"])
        predictor_a = MagicMock()
        predictor_a.signature.instructions = "First predictor instructions"
        predictor_b = MagicMock()
        predictor_b.signature.instructions = "Second predictor instructions"
        mock_compiled.named_predictors.return_value = [
            ("first", predictor_a),
            ("second", predictor_b),
        ]

        result = _extract_optimized_instructions(mock_compiled)

        assert result == "First predictor instructions"


# ---------------------------------------------------------------------------
# _extract_optimized_instructions — error path
# ---------------------------------------------------------------------------


class TestExtractOptimizedInstructionsError:
    """InstructionExtractionError must be raised when no instructions can be found."""

    def test_raises_when_no_predictors(self) -> None:
        """Empty named_predictors() and no predict.signature → InstructionExtractionError."""
        mock_compiled = MagicMock(spec=["named_predictors"])
        mock_compiled.named_predictors.return_value = []

        with pytest.raises(InstructionExtractionError):
            _extract_optimized_instructions(mock_compiled)

    def test_raises_with_descriptive_message(self) -> None:
        """Error message must be non-empty to help diagnose compiled program issues."""
        mock_compiled = MagicMock(spec=["named_predictors"])
        mock_compiled.named_predictors.return_value = []

        with pytest.raises(InstructionExtractionError, match=r".+"):
            _extract_optimized_instructions(mock_compiled)

    def test_raises_when_predictors_have_no_instructions(self) -> None:
        """Predictor without .signature.instructions attribute must not satisfy the fallback."""
        mock_compiled = MagicMock(spec=["named_predictors"])
        # Predictor has signature but not instructions
        predictor = MagicMock(spec=["some_other_attr"])
        mock_compiled.named_predictors.return_value = [("predict", predictor)]

        with pytest.raises(InstructionExtractionError):
            _extract_optimized_instructions(mock_compiled)

    def test_instruction_extraction_error_is_exception(self) -> None:
        """InstructionExtractionError must be a proper Exception subclass."""
        assert issubclass(InstructionExtractionError, Exception), (
            "InstructionExtractionError must inherit from Exception"
        )


# ---------------------------------------------------------------------------
# GEPAOptimizer.optimize() — wiring tests
# ---------------------------------------------------------------------------


class TestGEPAOptimizerOptimize:
    """GEPAOptimizer must wire dspy.LM, dspy.GEPA, and dspy.context correctly."""

    def test_returns_string(self, tmp_path) -> None:
        """optimize() must return a str — it is stored as the optimized prompt."""
        config = _make_config(tmp_path)
        train = _make_pairs(3)
        val = _make_pairs(2)

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Optimized instructions"

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_instance = MagicMock()
            mock_gepa_cls.return_value = mock_gepa_instance
            mock_gepa_instance.compile.return_value = mock_compiled

            optimizer = GEPAOptimizer()
            result = optimizer.optimize(train, val, config)

        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

    def test_uses_target_model_for_target_lm(self, tmp_path) -> None:
        """dspy.LM must be called with config.target_model as the first argument."""
        config = _make_config(tmp_path)
        train = _make_pairs(2)
        val = _make_pairs(1)

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.gepa.dspy.LM") as mock_lm,
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.return_value = mock_compiled

            GEPAOptimizer().optimize(train, val, config)

        # First call to dspy.LM should be with target_model
        first_call_args = mock_lm.call_args_list[0]
        assert first_call_args[0][0] == config.target_model, (
            f"Expected dspy.LM called with '{config.target_model}', got: {first_call_args}"
        )

    def test_uses_reflection_model_for_reflection_lm(self, tmp_path) -> None:
        """dspy.LM must be called with config.reflection_model for the reflection LM."""
        config = _make_config(tmp_path)
        train = _make_pairs(2)
        val = _make_pairs(1)

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.gepa.dspy.LM") as mock_lm,
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.return_value = mock_compiled

            GEPAOptimizer().optimize(train, val, config)

        # Second call to dspy.LM should be with reflection_model
        assert mock_lm.call_count == 2, (
            f"Expected dspy.LM called twice (target + reflection), got {mock_lm.call_count}"
        )
        second_call_args = mock_lm.call_args_list[1]
        assert second_call_args[0][0] == config.reflection_model, (
            f"Expected reflection LM with '{config.reflection_model}', got: {second_call_args}"
        )

    def test_gepa_constructed_with_config_params(self, tmp_path) -> None:
        """GEPA must be initialized with the metric, auto mode, and num_threads from config."""
        config = _make_config(tmp_path)
        config_with_custom = config.model_copy(update={"gepa_auto": "medium", "num_threads": 8})
        train = _make_pairs(2)
        val = _make_pairs(1)

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.return_value = mock_compiled

            GEPAOptimizer().optimize(train, val, config_with_custom)

        gepa_kwargs = mock_gepa_cls.call_args[1]
        assert gepa_kwargs.get("auto") == "medium", (
            f"GEPA not initialized with auto='medium', got: {gepa_kwargs}"
        )
        assert gepa_kwargs.get("num_threads") == 8, (
            f"GEPA not initialized with num_threads=8, got: {gepa_kwargs}"
        )

    def test_trainset_uses_expected_response_not_response(self, tmp_path) -> None:
        """DSPy Examples for trainset must use 'expected_response' field — metric reads gold.expected_response."""
        config = _make_config(tmp_path)
        train = [
            PromptPair(
                prompt="What is 2+2?",
                response="4",
                source_model="openai/gpt-4o",
                output_type=OutputType.CLASSIFICATION,
            )
        ]
        val: list[PromptPair] = []

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        captured_trainset: list = []

        def capture_compile(program, trainset):
            captured_trainset.extend(trainset)
            return mock_compiled

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.side_effect = capture_compile

            GEPAOptimizer().optimize(train, val, config)

        assert len(captured_trainset) == 1
        example = captured_trainset[0]
        # metric.py accesses gold.expected_response — the field must exist on the example
        assert hasattr(example, "expected_response"), (
            "DSPy Example must have 'expected_response' field, metric accesses gold.expected_response"
        )
        assert example.expected_response == "4", (
            f"expected_response should be '4' (the PromptPair.response), got {example.expected_response!r}"
        )

    def test_trainset_prompt_field_is_input(self, tmp_path) -> None:
        """Each DSPy Example in trainset must have 'prompt' as an input field via .with_inputs()."""
        config = _make_config(tmp_path)
        train = _make_pairs(1)
        val: list[PromptPair] = []

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        captured_trainset: list = []

        def capture_compile(program, trainset):
            captured_trainset.extend(trainset)
            return mock_compiled

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.side_effect = capture_compile

            GEPAOptimizer().optimize(train, val, config)

        example = captured_trainset[0]
        assert "prompt" in example.inputs(), (
            f"'prompt' must be an input field on the DSPy Example, inputs: {example.inputs()}"
        )

    def test_returns_extracted_instructions_from_compiled(self, tmp_path) -> None:
        """optimize() must return what _extract_optimized_instructions finds in compiled program."""
        config = _make_config(tmp_path)
        train = _make_pairs(2)
        val = _make_pairs(1)

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Be concise and accurate."

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.return_value = mock_compiled

            result = GEPAOptimizer().optimize(train, val, config)

        assert result == "Be concise and accurate.", (
            f"optimize() should return compiled instructions, got: {result!r}"
        )

    def test_optimizer_implements_optimizer_abc(self) -> None:
        """GEPAOptimizer must implement the Optimizer ABC — callers depend on this contract."""
        from rosettastone.optimize.base import Optimizer

        assert issubclass(GEPAOptimizer, Optimizer), (
            "GEPAOptimizer must be a subclass of Optimizer ABC"
        )
