"""Tests for MIPROv2Optimizer in optimize/mipro.py.

dspy.MIPROv2 and dspy.LM are mocked entirely — no API calls, no model downloads,
no money spent. The tests verify the optimizer's wiring, zero-shot config, and
instruction extraction, not DSPy's internal behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rosettastone.config import MigrationConfig
from rosettastone.core.types import OutputType, PromptPair
from rosettastone.optimize.mipro import MIPROv2Optimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, **overrides) -> MigrationConfig:
    data_file = tmp_path / "data.jsonl"
    data_file.touch()
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_file,
        **overrides,
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


def _patched_optimize(config, train=None, val=None, instructions="Optimized instructions"):
    """Run MIPROv2Optimizer.optimize() with all dspy internals mocked."""
    if train is None:
        train = _make_pairs(3)
    if val is None:
        val = _make_pairs(2)

    mock_compiled = MagicMock()
    mock_compiled.predict.signature.instructions = instructions

    with (
        patch("rosettastone.optimize.mipro.dspy.LM"),
        patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
        patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
    ):
        mock_ctx.return_value.__enter__ = lambda s: s
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_mipro_cls.return_value.compile.return_value = mock_compiled

        optimizer = MIPROv2Optimizer()
        result = optimizer.optimize(train, val, config)

    return result, mock_mipro_cls


# ---------------------------------------------------------------------------
# MIPROv2Optimizer — ABC conformance
# ---------------------------------------------------------------------------


class TestMIPROv2OptimizerABC:
    """MIPROv2Optimizer must implement the Optimizer ABC contract."""

    def test_implements_optimizer_abc(self) -> None:
        """MIPROv2Optimizer must be a subclass of Optimizer ABC."""
        from rosettastone.optimize.base import Optimizer

        assert issubclass(MIPROv2Optimizer, Optimizer), (
            "MIPROv2Optimizer must be a subclass of Optimizer ABC"
        )

    def test_optimize_method_exists(self) -> None:
        """optimize() method must be present and callable."""
        assert callable(MIPROv2Optimizer().optimize)


# ---------------------------------------------------------------------------
# MIPROv2Optimizer.optimize() — return type
# ---------------------------------------------------------------------------


class TestMIPROv2OptimizerReturnType:
    """optimize() must always return a str."""

    def test_returns_string(self, tmp_path) -> None:
        """optimize() must return a str — it is stored as the optimized prompt."""
        config = _make_config(tmp_path)
        result, _ = _patched_optimize(config)

        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

    def test_returns_extracted_instructions(self, tmp_path) -> None:
        """optimize() must return what extract_optimized_instructions finds in compiled program."""
        config = _make_config(tmp_path)
        result, _ = _patched_optimize(config, instructions="Be concise and accurate.")

        assert result == "Be concise and accurate.", (
            f"optimize() should return compiled instructions, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# MIPROv2Optimizer — zero-shot config
# ---------------------------------------------------------------------------


class TestMIPROv2ZeroShotConfig:
    """MIPROv2 must always use zero-shot mode (no demos from production data)."""

    def test_zero_shot_max_bootstrapped_demos(self, tmp_path) -> None:
        """max_bootstrapped_demos must be 0 — no production data in demos (PII safety)."""
        config = _make_config(tmp_path)
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.return_value = mock_compiled

            MIPROv2Optimizer().optimize(_make_pairs(2), _make_pairs(1), config)

        kwargs = mock_mipro_cls.call_args[1]
        assert kwargs.get("max_bootstrapped_demos") == 0, (
            f"max_bootstrapped_demos must be 0, got: {kwargs.get('max_bootstrapped_demos')}"
        )

    def test_zero_shot_max_labeled_demos(self, tmp_path) -> None:
        """max_labeled_demos must be 0 — no production data in demos (PII safety)."""
        config = _make_config(tmp_path)
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.return_value = mock_compiled

            MIPROv2Optimizer().optimize(_make_pairs(2), _make_pairs(1), config)

        kwargs = mock_mipro_cls.call_args[1]
        assert kwargs.get("max_labeled_demos") == 0, (
            f"max_labeled_demos must be 0, got: {kwargs.get('max_labeled_demos')}"
        )


# ---------------------------------------------------------------------------
# MIPROv2Optimizer — config wiring
# ---------------------------------------------------------------------------


class TestMIPROv2ConfigWiring:
    """MIPROv2 must forward config values to the DSPy optimizer correctly."""

    def test_uses_target_model_for_lm(self, tmp_path) -> None:
        """dspy.LM must be called with config.target_model."""
        config = _make_config(tmp_path)
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.mipro.dspy.LM") as mock_lm,
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.return_value = mock_compiled

            MIPROv2Optimizer().optimize(_make_pairs(2), _make_pairs(1), config)

        first_call_args = mock_lm.call_args_list[0]
        assert first_call_args[0][0] == config.target_model, (
            f"Expected dspy.LM called with '{config.target_model}', got: {first_call_args}"
        )

    def test_uses_mipro_auto_from_config(self, tmp_path) -> None:
        """MIPROv2 must be initialized with config.mipro_auto when it is set."""
        config = _make_config(tmp_path, mipro_auto="heavy")
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.return_value = mock_compiled

            MIPROv2Optimizer().optimize(_make_pairs(2), _make_pairs(1), config)

        kwargs = mock_mipro_cls.call_args[1]
        assert kwargs.get("auto") == "heavy", f"Expected auto='heavy', got: {kwargs.get('auto')}"

    def test_defaults_to_light_when_mipro_auto_is_none(self, tmp_path) -> None:
        """When config.mipro_auto is None, MIPROv2 must default to 'light' auto preset."""
        config = _make_config(tmp_path, mipro_auto=None)
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.return_value = mock_compiled

            MIPROv2Optimizer().optimize(_make_pairs(2), _make_pairs(1), config)

        kwargs = mock_mipro_cls.call_args[1]
        assert kwargs.get("auto") == "light", (
            f"Expected auto='light' fallback when mipro_auto=None, got: {kwargs.get('auto')}"
        )

    def test_uses_num_threads_from_config(self, tmp_path) -> None:
        """MIPROv2 must be initialized with config.num_threads."""
        config = _make_config(tmp_path, num_threads=8)
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.return_value = mock_compiled

            MIPROv2Optimizer().optimize(_make_pairs(2), _make_pairs(1), config)

        kwargs = mock_mipro_cls.call_args[1]
        assert kwargs.get("num_threads") == 8, (
            f"Expected num_threads=8, got: {kwargs.get('num_threads')}"
        )


# ---------------------------------------------------------------------------
# MIPROv2Optimizer — trainset construction
# ---------------------------------------------------------------------------


class TestMIPROv2TrainsetConstruction:
    """DSPy Examples must be constructed correctly from PromptPair objects."""

    def test_trainset_uses_expected_response_field(self, tmp_path) -> None:
        """DSPy Examples must use 'expected_response' — metric reads gold.expected_response."""
        config = _make_config(tmp_path)
        train = [
            PromptPair(
                prompt="What is 2+2?",
                response="4",
                source_model="openai/gpt-4o",
                output_type=OutputType.CLASSIFICATION,
            )
        ]

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"
        captured_trainset: list = []

        def capture_compile(program, trainset):
            captured_trainset.extend(trainset)
            return mock_compiled

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.side_effect = capture_compile

            MIPROv2Optimizer().optimize(train, [], config)

        assert len(captured_trainset) == 1
        example = captured_trainset[0]
        assert hasattr(example, "expected_response"), (
            "DSPy Example must have 'expected_response' field, metric accesses gold.expected_response"
        )
        assert example.expected_response == "4", (
            f"expected_response should be '4', got {example.expected_response!r}"
        )

    def test_trainset_prompt_field_is_input(self, tmp_path) -> None:
        """Each DSPy Example in trainset must have 'prompt' as an input field."""
        config = _make_config(tmp_path)
        train = _make_pairs(1)

        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"
        captured_trainset: list = []

        def capture_compile(program, trainset):
            captured_trainset.extend(trainset)
            return mock_compiled

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.side_effect = capture_compile

            MIPROv2Optimizer().optimize(train, [], config)

        example = captured_trainset[0]
        assert "prompt" in example.inputs(), (
            f"'prompt' must be an input field on the DSPy Example, inputs: {example.inputs()}"
        )

    def test_empty_train_set_passes_empty_trainset(self, tmp_path) -> None:
        """An empty train_set must pass an empty list to compile — no crash."""
        config = _make_config(tmp_path)
        mock_compiled = MagicMock()
        mock_compiled.predict.signature.instructions = "Instructions"
        captured_trainset: list = []

        def capture_compile(program, trainset):
            captured_trainset.extend(trainset)
            return mock_compiled

        with (
            patch("rosettastone.optimize.mipro.dspy.LM"),
            patch("rosettastone.optimize.mipro.dspy.MIPROv2") as mock_mipro_cls,
            patch("rosettastone.optimize.mipro.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mipro_cls.return_value.compile.side_effect = capture_compile

            result = MIPROv2Optimizer().optimize([], [], config)

        assert captured_trainset == [], f"Expected empty trainset, got: {captured_trainset}"
        assert isinstance(result, str)
