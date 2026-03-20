"""Tests for the JSONLAdapter ingest module."""

from __future__ import annotations

import json

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.ingest.jsonl import JSONLAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL to path."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_record(
    prompt="Hello",
    response="World",
    source_model="openai/gpt-4o",
    **extra,
) -> dict:
    return {"prompt": prompt, "response": response, "source_model": source_model, **extra}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_single_valid_pair(tmp_path):
    """This test proves that a single valid JSONL line is loaded as one PromptPair."""
    jsonl_file = tmp_path / "data.jsonl"
    _write_jsonl(jsonl_file, [_make_record(prompt="Say hi", response="Hi!")])

    adapter = JSONLAdapter(jsonl_file)
    result = adapter.load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert isinstance(result[0], PromptPair), (
        f"Expected PromptPair instance, got: {type(result[0])}"
    )
    assert result[0].prompt == "Say hi", f"Expected prompt 'Say hi', got: {result[0].prompt!r}"
    assert result[0].response == "Hi!", f"Expected response 'Hi!', got: {result[0].response!r}"
    assert result[0].source_model == "openai/gpt-4o", (
        f"Expected source_model 'openai/gpt-4o', got: {result[0].source_model!r}"
    )


def test_load_multiple_valid_pairs(tmp_path):
    """This test proves that multiple valid JSONL lines each produce a PromptPair."""
    records = [
        _make_record(prompt="Q1", response="A1"),
        _make_record(prompt="Q2", response="A2"),
        _make_record(prompt="Q3", response="A3"),
    ]
    jsonl_file = tmp_path / "data.jsonl"
    _write_jsonl(jsonl_file, records)

    result = JSONLAdapter(jsonl_file).load()

    assert len(result) == 3, f"Expected 3 pairs, got {len(result)}"
    prompts = [p.prompt for p in result]
    assert "Q1" in prompts and "Q2" in prompts and "Q3" in prompts, (
        f"Expected prompts Q1/Q2/Q3 in result, got: {prompts}"
    )


def test_empty_lines_are_skipped(tmp_path):
    """This test proves that blank lines in JSONL do not produce errors or extra pairs."""
    jsonl_file = tmp_path / "data.jsonl"
    with open(jsonl_file, "w") as f:
        f.write(json.dumps(_make_record(prompt="Q1", response="A1")) + "\n")
        f.write("\n")  # blank line
        f.write("   \n")  # whitespace-only line
        f.write(json.dumps(_make_record(prompt="Q2", response="A2")) + "\n")

    result = JSONLAdapter(jsonl_file).load()

    assert len(result) == 2, (
        f"Expected 2 pairs (blank lines skipped), got {len(result)}"
    )


def test_list_prompt_messages_format_loaded_correctly(tmp_path):
    """This test proves that a list-format prompt (messages) is preserved in the PromptPair."""
    messages = [{"role": "user", "content": "What is 2+2?"}]
    record = {"prompt": messages, "response": "4", "source_model": "openai/gpt-4o"}
    jsonl_file = tmp_path / "data.jsonl"
    _write_jsonl(jsonl_file, [record])

    result = JSONLAdapter(jsonl_file).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert isinstance(result[0].prompt, list), (
        f"Expected prompt to be a list, got: {type(result[0].prompt)}"
    )
    assert result[0].prompt == messages, (
        f"Expected messages format preserved, got: {result[0].prompt!r}"
    )


def test_dict_response_normalized_to_string(tmp_path):
    """This test proves that a dict response is coerced to a string by the schema validator."""
    record = {
        "prompt": "Say hello",
        "response": {"content": "Hello!"},
        "source_model": "openai/gpt-4o",
    }
    jsonl_file = tmp_path / "data.jsonl"
    _write_jsonl(jsonl_file, [record])

    result = JSONLAdapter(jsonl_file).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert isinstance(result[0].response, str), (
        f"Expected response to be string, got: {type(result[0].response)}"
    )
    assert result[0].response == "Hello!", (
        f"Expected 'Hello!' (normalized from content key), got: {result[0].response!r}"
    )


def test_optional_metadata_and_feedback_preserved(tmp_path):
    """This test proves that optional metadata and feedback fields are loaded when present."""
    record = _make_record(
        prompt="Test",
        response="Response",
        metadata={"session": "xyz"},
        feedback="positive",
    )
    jsonl_file = tmp_path / "data.jsonl"
    _write_jsonl(jsonl_file, [record])

    result = JSONLAdapter(jsonl_file).load()

    assert result[0].metadata == {"session": "xyz"}, (
        f"Expected metadata preserved, got: {result[0].metadata!r}"
    )
    assert result[0].feedback == "positive", (
        f"Expected feedback='positive', got: {result[0].feedback!r}"
    )


def test_file_with_only_blank_lines_returns_empty_list(tmp_path):
    """This test proves that a file with only blank lines yields an empty list, not an error."""
    jsonl_file = tmp_path / "empty.jsonl"
    with open(jsonl_file, "w") as f:
        f.write("\n\n\n")

    result = JSONLAdapter(jsonl_file).load()

    assert result == [], (
        f"Expected empty list from blank-only file, got: {result!r}"
    )


def test_truly_empty_file_returns_empty_list(tmp_path):
    """This test proves that a completely empty file yields an empty list."""
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_text("")

    result = JSONLAdapter(jsonl_file).load()

    assert result == [], f"Expected empty list from empty file, got: {result!r}"


def test_accepts_path_as_string(tmp_path):
    """This test proves that a str path (not Path object) is accepted by JSONLAdapter."""
    jsonl_file = tmp_path / "data.jsonl"
    _write_jsonl(jsonl_file, [_make_record()])

    # Pass as string, not Path
    result = JSONLAdapter(str(jsonl_file)).load()

    assert len(result) == 1, f"Expected 1 pair when path passed as str, got {len(result)}"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_malformed_json_line_raises_value_error_with_line_number(tmp_path):
    """This test proves that invalid JSON on a line raises ValueError citing that line number."""
    jsonl_file = tmp_path / "bad.jsonl"
    with open(jsonl_file, "w") as f:
        f.write(json.dumps(_make_record(prompt="Q1", response="A1")) + "\n")
        f.write("this is not json\n")  # line 2
        f.write(json.dumps(_make_record(prompt="Q3", response="A3")) + "\n")

    with pytest.raises(ValueError, match="line 2") as exc_info:
        JSONLAdapter(jsonl_file).load()

    # Confirm it's descriptive enough to diagnose the issue
    assert "2" in str(exc_info.value), (
        f"Expected line number 2 in error message, got: {str(exc_info.value)!r}"
    )


def test_malformed_json_on_first_line_raises_value_error(tmp_path):
    """This test proves that line 1 malformed JSON is caught and cited correctly."""
    jsonl_file = tmp_path / "bad.jsonl"
    with open(jsonl_file, "w") as f:
        f.write("{not valid json\n")

    with pytest.raises(ValueError, match="line 1"):
        JSONLAdapter(jsonl_file).load()


def test_missing_required_field_raises_value_error_with_line_number(tmp_path):
    """This test proves that a line missing 'prompt' raises ValueError with the line number."""
    jsonl_file = tmp_path / "bad.jsonl"
    with open(jsonl_file, "w") as f:
        f.write(json.dumps(_make_record(prompt="Q1", response="A1")) + "\n")
        # Line 2 is missing 'prompt'
        f.write(json.dumps({"response": "A2", "source_model": "openai/gpt-4o"}) + "\n")

    with pytest.raises(ValueError, match="line 2"):
        JSONLAdapter(jsonl_file).load()


def test_missing_response_field_raises_value_error(tmp_path):
    """This test proves that a line missing 'response' raises ValueError."""
    jsonl_file = tmp_path / "bad.jsonl"
    with open(jsonl_file, "w") as f:
        # Missing 'response'
        f.write(json.dumps({"prompt": "Q1", "source_model": "openai/gpt-4o"}) + "\n")

    with pytest.raises(ValueError, match="line 1"):
        JSONLAdapter(jsonl_file).load()


def test_missing_source_model_raises_value_error(tmp_path):
    """This test proves that a line missing 'source_model' raises ValueError."""
    jsonl_file = tmp_path / "bad.jsonl"
    with open(jsonl_file, "w") as f:
        f.write(json.dumps({"prompt": "Q1", "response": "A1"}) + "\n")

    with pytest.raises(ValueError, match="line 1"):
        JSONLAdapter(jsonl_file).load()


def test_error_stops_processing_at_bad_line(tmp_path):
    """This test proves that a bad line raises immediately rather than being silently skipped."""
    jsonl_file = tmp_path / "mixed.jsonl"
    with open(jsonl_file, "w") as f:
        f.write(json.dumps(_make_record(prompt="Q1", response="A1")) + "\n")
        f.write("INVALID\n")

    # Must raise — bad lines are not silently skipped
    with pytest.raises(ValueError):
        JSONLAdapter(jsonl_file).load()


def test_partial_json_object_raises_value_error(tmp_path):
    """This test proves that truncated JSON (e.g., half-written object) raises ValueError."""
    jsonl_file = tmp_path / "truncated.jsonl"
    with open(jsonl_file, "w") as f:
        f.write('{"prompt": "Hello", "response":\n')  # truncated

    with pytest.raises(ValueError):
        JSONLAdapter(jsonl_file).load()
