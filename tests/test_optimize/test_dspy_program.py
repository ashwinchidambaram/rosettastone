"""Tests for the DSPy program structure (MigrationSignature, MigrationProgram).

These are pure structure tests — no mocking needed. They prove that the
program is wired correctly before any optimization or inference runs.

DSPy specifics:
- Field type (input vs output) is stored in field.json_schema_extra['__dspy_field_type']
- ChainOfThought wraps an inner Predict; the signature lives on cot.predict.signature
"""

import dspy
import pytest

from rosettastone.optimize.dspy_program import MigrationProgram, MigrationSignature


class TestMigrationSignature:
    """MigrationSignature must have the exact field contract DSPy and GEPA expect."""

    def test_has_prompt_input_field(self) -> None:
        """Signature must declare `prompt` as an InputField so GEPA injects it correctly."""
        fields = MigrationSignature.model_fields
        assert "prompt" in fields, (
            "MigrationSignature missing 'prompt' InputField — "
            "GEPA will fail to inject the user prompt"
        )

    def test_has_response_output_field(self) -> None:
        """Signature must declare `response` as an OutputField to receive model output."""
        fields = MigrationSignature.model_fields
        assert "response" in fields, (
            "MigrationSignature missing 'response' OutputField — "
            "metric accesses pred.response which would raise AttributeError"
        )

    def test_prompt_is_input_field(self) -> None:
        """prompt must be a DSPy InputField — wrong type breaks GEPA injection."""
        field_info = MigrationSignature.model_fields["prompt"]
        extra = field_info.json_schema_extra or {}
        field_type = extra.get("__dspy_field_type")
        assert field_type == "input", (
            f"Expected prompt to be DSPy InputField (__dspy_field_type='input'), "
            f"got json_schema_extra={extra!r}"
        )

    def test_response_is_output_field(self) -> None:
        """response must be a DSPy OutputField — wrong type means GEPA can't optimize it."""
        field_info = MigrationSignature.model_fields["response"]
        extra = field_info.json_schema_extra or {}
        field_type = extra.get("__dspy_field_type")
        assert field_type == "output", (
            f"Expected response to be DSPy OutputField (__dspy_field_type='output'), "
            f"got json_schema_extra={extra!r}"
        )

    def test_exactly_two_fields(self) -> None:
        """Signature should have exactly 2 fields — extra fields would confuse GEPA's signature."""
        fields = MigrationSignature.model_fields
        assert len(fields) == 2, (
            f"Expected exactly 2 fields (prompt, response), found {len(fields)}: {list(fields)}"
        )


class TestMigrationProgram:
    """MigrationProgram must be wired as ChainOfThought so GEPA can inject instructions."""

    def test_has_predict_attribute(self) -> None:
        """MigrationProgram must expose a .predict attribute — GEPA reads compiled.predict."""
        program = MigrationProgram()
        assert hasattr(program, "predict"), (
            "MigrationProgram missing 'predict' attribute — "
            "_extract_optimized_instructions accesses compiled.predict.signature.instructions"
        )

    def test_predict_is_chain_of_thought(self) -> None:
        """predict must be ChainOfThought — this is the DSPy module GEPA can optimize."""
        program = MigrationProgram()
        assert isinstance(program.predict, dspy.ChainOfThought), (
            f"Expected predict to be dspy.ChainOfThought, got {type(program.predict).__name__}"
        )

    def test_predict_uses_migration_signature_fields(self) -> None:
        """ChainOfThought must be bound to MigrationSignature fields (prompt, response)."""
        program = MigrationProgram()
        # ChainOfThought stores its inner Predict on .predict; signature lives on .predict.signature
        inner_sig = program.predict.predict.signature
        assert "prompt" in inner_sig.model_fields, (
            "ChainOfThought inner signature missing 'prompt' field"
        )
        assert "response" in inner_sig.model_fields, (
            "ChainOfThought inner signature missing 'response' field"
        )

    def test_predict_inner_signature_has_input_output_types(self) -> None:
        """Inner Predict signature must preserve InputField/OutputField designations."""
        program = MigrationProgram()
        inner_sig = program.predict.predict.signature
        prompt_extra = inner_sig.model_fields["prompt"].json_schema_extra or {}
        response_extra = inner_sig.model_fields["response"].json_schema_extra or {}
        assert prompt_extra.get("__dspy_field_type") == "input", (
            f"Inner signature 'prompt' field must be input, got: {prompt_extra!r}"
        )
        assert response_extra.get("__dspy_field_type") == "output", (
            f"Inner signature 'response' field must be output, got: {response_extra!r}"
        )

    def test_program_is_dspy_module(self) -> None:
        """MigrationProgram must extend dspy.Module so DSPy can compile it."""
        program = MigrationProgram()
        assert isinstance(program, dspy.Module), (
            f"MigrationProgram must be a dspy.Module subclass, got {type(program).__name__}"
        )
