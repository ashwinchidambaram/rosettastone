"""Contract tests verifying DSPy API surface we depend on.

These tests make NO API calls and use NO mocks. They verify that the DSPy
library version installed exposes the classes, methods, and attributes that
our optimizer code depends on. If any of these fail after a DSPy upgrade,
it means our optimizer code needs to be updated.
"""

import dspy
import pytest


class TestDSPyAPISurface:
    """Verify that DSPy exposes the classes and attributes our optimizer code uses."""

    def test_dspy_lm_class_exists(self) -> None:
        """dspy.LM must exist and be callable (used in gepa.py and mipro.py)."""
        assert hasattr(dspy, "LM"), "dspy.LM not found — optimizer code depends on it"
        assert callable(dspy.LM), "dspy.LM must be callable to construct LM instances"

    def test_dspy_gepa_class_exists(self) -> None:
        """dspy.GEPA must exist (used in gepa.py as primary optimizer).

        If this fails after a DSPy upgrade, check whether GEPA was renamed or
        moved to a submodule (e.g., dspy.optimizers.GEPA or dspy.teleprompt.GEPA).
        """
        try:
            assert hasattr(dspy, "GEPA"), (
                "dspy.GEPA not found. Check if GEPA was renamed or moved in the new DSPy "
                "version. Look for it under dspy.optimizers or dspy.teleprompt."
            )
            assert callable(dspy.GEPA), "dspy.GEPA must be callable to instantiate the optimizer"
        except (AssertionError, AttributeError) as exc:
            # Provide actionable guidance on where to look
            hints = []
            for alt in ("dspy.optimizers.GEPA", "dspy.teleprompt.GEPA"):
                module_path, _, attr = alt.rpartition(".")
                try:
                    import importlib

                    mod = importlib.import_module(module_path)
                    if hasattr(mod, attr):
                        hints.append(f"  Found at {alt}")
                except ImportError:
                    pass
            hint_msg = "\n".join(hints) if hints else "  Not found in common alternative locations."
            pytest.fail(f"{exc}\nAlternative locations checked:\n{hint_msg}")

    def test_dspy_miprov2_class_exists(self) -> None:
        """dspy.MIPROv2 must exist (used in mipro.py as fallback optimizer)."""
        assert hasattr(dspy, "MIPROv2"), (
            "dspy.MIPROv2 not found — mipro.py depends on it. "
            "Check dspy.optimizers or dspy.teleprompt for relocated class."
        )
        assert callable(dspy.MIPROv2), "dspy.MIPROv2 must be callable"

    def test_dspy_prediction_accepts_score_and_feedback(self) -> None:
        """dspy.Prediction must accept score and feedback kwargs.

        metric.py returns dspy.Prediction(score=..., feedback=...) for GEPA's
        reflective optimization loop. Both attributes must be accessible.
        """
        pred = dspy.Prediction(score=0.5, feedback="ok")
        assert hasattr(pred, "score"), "Prediction must expose .score attribute"
        assert hasattr(pred, "feedback"), "Prediction must expose .feedback attribute"
        assert pred.score == 0.5, f"Expected score=0.5, got {pred.score}"
        assert pred.feedback == "ok", f"Expected feedback='ok', got {pred.feedback}"

    def test_dspy_example_accepts_kwargs(self) -> None:
        """dspy.Example must accept arbitrary keyword arguments.

        gepa.py and mipro.py construct Examples like:
            dspy.Example(prompt="...", expected_response="...").with_inputs("prompt")
        """
        ex = dspy.Example(prompt="hello", expected_response="world")
        assert hasattr(ex, "prompt"), "Example must expose .prompt attribute"
        assert hasattr(ex, "expected_response"), "Example must expose .expected_response attribute"
        assert ex.prompt == "hello"
        assert ex.expected_response == "world"

    def test_dspy_example_with_inputs(self) -> None:
        """dspy.Example must support .with_inputs() for marking input fields.

        Both gepa.py and mipro.py call .with_inputs("prompt") on constructed examples.
        """
        ex = dspy.Example(prompt="hello", expected_response="world").with_inputs("prompt")
        assert ex.prompt == "hello", "with_inputs should preserve attribute access"

    def test_dspy_module_supports_forward_override(self) -> None:
        """dspy.Module must be subclassable with a forward() method.

        dspy_program.py defines MigrationProgram(dspy.Module) with a forward() method.
        dspy.Module follows the PyTorch convention: __call__ dispatches to forward(),
        but forward() is not defined on the base class itself.
        """
        assert callable(dspy.Module), "dspy.Module must be callable (instantiable)"
        assert hasattr(dspy.Module, "__call__"), (
            "dspy.Module must have __call__ to dispatch to forward()"
        )

        # Verify that a subclass can define forward() — this is the actual contract
        class _TestModule(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(result="ok")

        mod = _TestModule()
        assert hasattr(mod, "forward"), "Subclass must be able to define forward()"

    def test_dspy_signature_supports_fields(self) -> None:
        """dspy.Signature, dspy.InputField, and dspy.OutputField must exist.

        dspy_program.py defines MigrationSignature as a dspy.Signature subclass
        with InputField and OutputField annotations.
        """
        assert hasattr(dspy, "Signature"), "dspy.Signature not found"
        assert hasattr(dspy, "InputField"), "dspy.InputField not found"
        assert hasattr(dspy, "OutputField"), "dspy.OutputField not found"

        # Verify InputField and OutputField are callable (used as field descriptors)
        assert callable(dspy.InputField), "dspy.InputField must be callable"
        assert callable(dspy.OutputField), "dspy.OutputField must be callable"

    def test_dspy_chain_of_thought_exists(self) -> None:
        """dspy.ChainOfThought must exist (used in dspy_program.py)."""
        assert hasattr(dspy, "ChainOfThought"), (
            "dspy.ChainOfThought not found — MigrationProgram uses it as its predict module"
        )
        assert callable(dspy.ChainOfThought), "dspy.ChainOfThought must be callable"

    def test_dspy_context_manager_exists(self) -> None:
        """dspy.context must exist and be callable (used for LM context in gepa.py/mipro.py).

        Both optimizer wrappers use `with dspy.context(lm=target_lm):` to set the
        active language model for the optimization run.
        """
        assert hasattr(dspy, "context"), (
            "dspy.context not found — gepa.py and mipro.py use `with dspy.context(lm=...):`"
        )
        assert callable(dspy.context), "dspy.context must be callable"

    def test_migration_program_instantiates(self) -> None:
        """MigrationProgram from rosettastone.optimize.dspy_program must instantiate.

        This validates that our Signature/Module/ChainOfThought integration works
        end-to-end without requiring any API calls.
        """
        from rosettastone.optimize.dspy_program import MigrationProgram

        program = MigrationProgram()
        assert isinstance(program, dspy.Module), "MigrationProgram must be a dspy.Module subclass"
        assert hasattr(program, "forward"), "MigrationProgram must have a forward method"
        assert hasattr(program, "predict"), "MigrationProgram must have a predict attribute"
