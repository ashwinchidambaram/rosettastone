"""Tests for BraintrustAdapter."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.ingest.braintrust_adapter import BraintrustAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_NAME = "my-project"
_SOURCE_MODEL = "openai/gpt-4o"
_API_KEY = "test-api-key"


def _make_log_entry(
    input_val=None,
    output_val="Answer",
    model: str | None = "gpt-4o",
    expected: str | None = None,
    tags: list[str] | None = None,
    timestamp: str | None = None,
    scores: dict | None = None,
    entry_id: str | None = "entry-123",
) -> dict:
    """Build a minimal Braintrust log entry dict."""
    entry: dict = {}
    if input_val is not None:
        entry["input"] = input_val
    if output_val is not None:
        entry["output"] = output_val
    metadata: dict = {}
    if model is not None:
        metadata["model"] = model
    if tags is not None:
        metadata["tags"] = tags
    if timestamp is not None:
        metadata["timestamp"] = timestamp
    if scores is not None:
        metadata["scores"] = scores
    entry["metadata"] = metadata
    if expected is not None:
        entry["expected"] = expected
    if entry_id is not None:
        entry["id"] = entry_id
    return entry


def _make_mock_client(entries: list[dict]) -> MagicMock:
    """Build a mock Braintrust client that returns the given log entries."""
    mock_logs = MagicMock()
    mock_logs.list.return_value = entries

    mock_project = MagicMock()
    mock_project.logs = mock_logs

    mock_client = MagicMock()
    mock_client.projects.retrieve.return_value = mock_project

    return mock_client


# ---------------------------------------------------------------------------
# Load — happy path (7 tests)
# ---------------------------------------------------------------------------


def test_load_returns_prompt_pairs_for_valid_entries():
    """This test proves that valid log entries are loaded as PromptPairs."""
    entries = [
        _make_log_entry(input_val="What is 2+2?", output_val="4", entry_id="e1"),
        _make_log_entry(input_val="Capital of France?", output_val="Paris", entry_id="e2"),
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"
    assert all(isinstance(p, PromptPair) for p in result), (
        "Expected all results to be PromptPair instances"
    )


def test_load_extracts_input_as_prompt():
    """This test proves that the 'input' field is extracted as the prompt."""
    entries = [_make_log_entry(input_val="Hello world", output_val="Hi there")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1
    assert result[0].prompt == "Hello world", f"Unexpected prompt: {result[0].prompt!r}"


def test_load_extracts_output_as_response():
    """This test proves that the 'output' field is extracted as the response."""
    entries = [_make_log_entry(input_val="Prompt", output_val="The response text")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].response == "The response text", f"Unexpected response: {result[0].response!r}"


def test_load_model_from_metadata():
    """This test proves that source_model is read from entry metadata.model."""
    entries = [_make_log_entry(input_val="Prompt", output_val="Resp", model="gpt-4-turbo")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].source_model == "gpt-4-turbo", (
        f"Expected source_model from metadata, got: {result[0].source_model!r}"
    )


def test_load_metadata_preserved():
    """This test proves that extra metadata fields are preserved on the PromptPair."""
    entries = [
        _make_log_entry(
            input_val="Prompt",
            output_val="Resp",
            tags=["production"],
            timestamp="2024-01-01T00:00:00Z",
            entry_id="abc-123",
        )
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].metadata.get("entry_id") == "abc-123", (
        f"Expected entry_id in metadata, got: {result[0].metadata!r}"
    )


def test_load_fallback_model_when_no_metadata_model():
    """This test proves that source_model falls back to adapter's source_model when metadata has no model."""
    entries = [_make_log_entry(input_val="Prompt", output_val="Resp", model=None)]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].source_model == _SOURCE_MODEL, (
        f"Expected fallback source_model={_SOURCE_MODEL!r}, got: {result[0].source_model!r}"
    )


def test_load_uses_project_name():
    """This test proves that the project_name is used to retrieve the project."""
    entries = [_make_log_entry(input_val="Q", output_val="A")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        adapter.load()

    mock_client.projects.retrieve.assert_called_once_with(_PROJECT_NAME)


def test_load_source_model_on_all_pairs():
    """This test proves that all PromptPairs have source_model set."""
    entries = [
        _make_log_entry(input_val=f"Q{i}", output_val=f"A{i}", model=None, entry_id=f"e{i}")
        for i in range(5)
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert all(p.source_model == _SOURCE_MODEL for p in result), (
        "Expected all PromptPairs to have the fallback source_model"
    )


# ---------------------------------------------------------------------------
# Edge cases (8 tests)
# ---------------------------------------------------------------------------


def test_load_empty_project_returns_empty_list():
    """This test proves that an empty log returns an empty list (not an error)."""
    mock_client = _make_mock_client([])

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result == [], f"Expected empty list for empty project, got: {result!r}"


def test_load_skips_entry_missing_input():
    """This test proves that entries without an 'input' field are silently skipped."""
    entries = [
        {"output": "Some output", "metadata": {}},
        _make_log_entry(input_val="Valid", output_val="Resp", entry_id="e2"),
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1, f"Expected 1 result (entry missing input skipped), got {len(result)}"
    assert result[0].prompt == "Valid"


def test_load_skips_entry_missing_output():
    """This test proves that entries without an 'output' field are silently skipped."""
    entries = [
        {"input": "Some input", "metadata": {}},
        _make_log_entry(input_val="Valid", output_val="Resp", entry_id="e2"),
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1, f"Expected 1 result (entry missing output skipped), got {len(result)}"


def test_load_mixed_valid_and_invalid_entries():
    """This test proves that mixed entries yield only valid PromptPairs."""
    entries = [
        _make_log_entry(input_val="Good1", output_val="Resp1", entry_id="e1"),
        {"output": "No input here", "metadata": {}},
        _make_log_entry(input_val="Good2", output_val="Resp2", entry_id="e3"),
        {"input": "No output here", "metadata": {}},
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 2, f"Expected 2 valid pairs, got {len(result)}"


def test_load_logs_skipped_entries_structurally(caplog):
    """This test proves that skipped entries produce a structural warning log (no content logged)."""
    entries = [
        {"output": "Bad entry — no input", "metadata": {}},
        _make_log_entry(input_val="Good", output_val="Resp", entry_id="e2"),
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        with caplog.at_level(logging.WARNING, logger="rosettastone.ingest.braintrust_adapter"):
            adapter.load()

    assert any("skip" in record.message.lower() for record in caplog.records), (
        "Expected a warning log mentioning 'skip' for unparseable entries"
    )


def test_load_dict_output_normalized_to_string():
    """This test proves that a dict output is JSON-serialized to a string response."""
    dict_output = {"key": "value", "nested": {"a": 1}}
    entries = [_make_log_entry(input_val="Prompt", output_val=dict_output)]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1
    assert isinstance(result[0].response, str), (
        f"Expected response to be str after dict normalization, got: {type(result[0].response)}"
    )
    parsed = json.loads(result[0].response)
    assert parsed == dict_output, f"Unexpected parsed response: {parsed!r}"


def test_load_list_messages_input_preserved():
    """This test proves that a list-of-messages input is preserved as-is on the PromptPair."""
    list_input = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    entries = [_make_log_entry(input_val=list_input, output_val="Hi there")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1
    assert result[0].prompt == list_input, (
        f"Expected list input preserved, got: {result[0].prompt!r}"
    )


def test_load_string_input_preserved():
    """This test proves that a plain string input is preserved as-is."""
    entries = [_make_log_entry(input_val="Plain string prompt", output_val="Resp")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].prompt == "Plain string prompt"


# ---------------------------------------------------------------------------
# ImportError handling (3 tests)
# ---------------------------------------------------------------------------


def test_make_client_raises_import_error_with_clear_message():
    """This test proves that a missing 'braintrust' package raises ImportError."""
    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY)

    with patch.dict("sys.modules", {"braintrust": None}):
        with pytest.raises(ImportError):
            adapter._make_client()


def test_make_client_import_error_includes_install_suggestion():
    """This test proves that the ImportError message includes pip install guidance."""
    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY)

    with patch.dict("sys.modules", {"braintrust": None}):
        with pytest.raises(ImportError, match="pip install braintrust"):
            adapter._make_client()


def test_adapter_constructable_without_braintrust_installed():
    """This test proves that BraintrustAdapter can be instantiated without braintrust installed."""
    with patch.dict("sys.modules", {"braintrust": None}):
        # Should NOT raise — lazy import means construction is always safe
        adapter = BraintrustAdapter(_PROJECT_NAME)
        assert adapter is not None


# ---------------------------------------------------------------------------
# Field mapping (7 tests)
# ---------------------------------------------------------------------------


def test_chat_format_messages_as_prompt():
    """This test proves that a chat-format messages list is mapped to prompt correctly."""
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What time is it?"},
    ]
    entries = [_make_log_entry(input_val=messages, output_val="12:00 PM")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].prompt == messages


def test_text_format_string_as_prompt():
    """This test proves that a plain text string input maps directly to prompt."""
    entries = [_make_log_entry(input_val="Summarize this article.", output_val="Summary here.")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].prompt == "Summarize this article."


def test_expected_field_maps_to_feedback():
    """This test proves that the 'expected' field in an entry maps to PromptPair.feedback."""
    entries = [
        _make_log_entry(input_val="Prompt", output_val="Actual", expected="Expected ideal response")
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].feedback == "Expected ideal response", (
        f"Expected feedback from 'expected' field, got: {result[0].feedback!r}"
    )


def test_tags_in_metadata():
    """This test proves that tags from entry metadata are preserved in PromptPair.metadata."""
    entries = [
        _make_log_entry(
            input_val="Prompt", output_val="Resp", tags=["production", "v2"], entry_id="e1"
        )
    ]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].metadata.get("tags") == ["production", "v2"], (
        f"Expected tags preserved in metadata, got: {result[0].metadata!r}"
    )


def test_timestamp_in_metadata():
    """This test proves that timestamp from entry metadata is preserved in PromptPair.metadata."""
    ts = "2024-06-15T12:34:56Z"
    entries = [_make_log_entry(input_val="Prompt", output_val="Resp", timestamp=ts, entry_id="e1")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].metadata.get("timestamp") == ts, (
        f"Expected timestamp in metadata, got: {result[0].metadata!r}"
    )


def test_scores_in_metadata():
    """This test proves that scores from entry metadata are preserved in PromptPair.metadata."""
    scores = {"accuracy": 0.95, "helpfulness": 0.88}
    entries = [_make_log_entry(input_val="Prompt", output_val="Resp", scores=scores, entry_id="e1")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].metadata.get("scores") == scores, (
        f"Expected scores in metadata, got: {result[0].metadata!r}"
    )


def test_entry_id_in_metadata():
    """This test proves that the entry 'id' field is stored in PromptPair.metadata as entry_id."""
    entries = [_make_log_entry(input_val="Prompt", output_val="Resp", entry_id="unique-id-999")]
    mock_client = _make_mock_client(entries)

    adapter = BraintrustAdapter(_PROJECT_NAME, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].metadata.get("entry_id") == "unique-id-999", (
        f"Expected entry_id in metadata, got: {result[0].metadata!r}"
    )
