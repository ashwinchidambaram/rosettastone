"""Tests for LangSmithAdapter."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.ingest.langsmith_adapter import LangSmithAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT = "my-langsmith-project"
_SOURCE_MODEL = "openai/gpt-4o"
_API_KEY = "ls__test_api_key"

# Sentinel: use _MISSING to explicitly set a run attribute to None (distinct
# from "use the default").
_MISSING = object()


# ---------------------------------------------------------------------------
# Helpers — mock run factory
# ---------------------------------------------------------------------------


def _make_run(
    *,
    inputs: Any = _MISSING,
    outputs: Any = _MISSING,
    extra: dict[str, Any] | None = None,
    execution_order: int = 1,
) -> MagicMock:
    """Return a MagicMock that looks like a LangSmith Run object.

    Pass ``inputs=None`` or ``outputs=None`` explicitly to simulate a run with
    no inputs/outputs (the default provides sensible values).
    """
    run = MagicMock()
    run.inputs = (
        {"messages": [{"role": "user", "content": "Hello"}]} if inputs is _MISSING else inputs
    )
    run.outputs = {"output": "Hi there"} if outputs is _MISSING else outputs
    run.extra = extra if extra is not None else {}
    run.execution_order = execution_order
    return run


def _make_chat_run(prompt: str = "Hello", response: str = "Hi") -> MagicMock:
    """Run with messages-style prompt and output-style response."""
    return _make_run(
        inputs={"messages": [{"role": "user", "content": prompt}]},
        outputs={"output": response},
    )


def _make_mock_client(runs: list[MagicMock]) -> MagicMock:
    """Return a mock LangSmith client whose list_runs returns ``runs``."""
    client = MagicMock()
    client.list_runs.return_value = iter(runs)
    return client


# ---------------------------------------------------------------------------
# Load tests (8)
# ---------------------------------------------------------------------------


def test_load_returns_prompt_pairs_for_valid_runs():
    """Valid runs with messages+output are returned as PromptPairs."""
    runs = [
        _make_chat_run("What is 2+2?", "4"),
        _make_chat_run("Capital of France?", "Paris"),
    ]
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client(runs)):
        result = adapter.load()

    assert len(result) == 2
    assert all(isinstance(p, PromptPair) for p in result)


def test_load_extracts_prompt_from_messages_key():
    """Prompt is extracted from run.inputs['messages'] as a list of dicts."""
    messages = [{"role": "user", "content": "Hello from messages key"}]
    run = _make_run(inputs={"messages": messages}, outputs={"output": "Response"})
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert len(result) == 1
    assert result[0].prompt == messages


def test_load_extracts_prompt_from_input_key():
    """Prompt is extracted from run.inputs['input'] when 'messages' is absent."""
    run = _make_run(inputs={"input": "String prompt via input key"}, outputs={"output": "OK"})
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert len(result) == 1
    assert result[0].prompt == "String prompt via input key"


def test_load_extracts_response_from_output_key():
    """Response is extracted from run.outputs['output']."""
    run = _make_run(
        inputs={"messages": [{"role": "user", "content": "Q"}]},
        outputs={"output": "Answer from output key"},
    )
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert result[0].response == "Answer from output key"


def test_load_extracts_response_from_generations_key():
    """Response is extracted from run.outputs['generations'][0][0]['text'] when 'output' is absent."""
    run = _make_run(
        inputs={"messages": [{"role": "user", "content": "Q"}]},
        outputs={"generations": [[{"text": "Generations response"}]]},
    )
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert result[0].response == "Generations response"


def test_load_extracts_model_from_run_metadata():
    """source_model on PromptPair comes from run.extra['metadata']['ls_model_name'] when present."""
    run = _make_run(
        inputs={"messages": [{"role": "user", "content": "Q"}]},
        outputs={"output": "A"},
        extra={"metadata": {"ls_model_name": "openai/gpt-4-turbo"}},
    )
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert result[0].source_model == "openai/gpt-4-turbo"


def test_load_falls_back_to_source_model_when_no_metadata_model():
    """When ls_model_name is absent, PromptPair.source_model falls back to adapter source_model."""
    run = _make_run(
        inputs={"messages": [{"role": "user", "content": "Q"}]},
        outputs={"output": "A"},
        extra={},
    )
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert result[0].source_model == _SOURCE_MODEL


def test_load_preserves_run_metadata_in_prompt_pair():
    """run.extra contents (minus model name) are stored in PromptPair.metadata."""
    extra = {"metadata": {"ls_model_name": "gpt-4o", "session_id": "abc-123", "tags": ["prod"]}}
    run = _make_run(
        inputs={"messages": [{"role": "user", "content": "Q"}]},
        outputs={"output": "A"},
        extra=extra,
    )
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert "extra" in result[0].metadata
    assert result[0].metadata["extra"] == extra


# ---------------------------------------------------------------------------
# Filtering tests (5)
# ---------------------------------------------------------------------------


def test_load_passes_date_range_to_list_runs():
    """start_date and end_date are forwarded to client.list_runs as keyword arguments."""
    run = _make_chat_run()
    mock_client = _make_mock_client([run])
    adapter = LangSmithAdapter(
        _PROJECT,
        api_key=_API_KEY,
        start_date="2024-01-01",
        end_date="2024-12-31",
        source_model=_SOURCE_MODEL,
    )
    with patch.object(adapter, "_make_client", return_value=mock_client):
        adapter.load()

    call_kwargs = mock_client.list_runs.call_args.kwargs
    assert call_kwargs.get("start_time") == "2024-01-01"
    assert call_kwargs.get("end_time") == "2024-12-31"


def test_load_omits_date_filters_when_none():
    """When start_date and end_date are None, no time filters are passed to list_runs."""
    run = _make_chat_run()
    mock_client = _make_mock_client([run])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        adapter.load()

    call_kwargs = mock_client.list_runs.call_args.kwargs
    assert "start_time" not in call_kwargs
    assert "end_time" not in call_kwargs


def test_load_passes_project_name_to_list_runs():
    """project_name is forwarded to client.list_runs."""
    run = _make_chat_run()
    mock_client = _make_mock_client([run])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        adapter.load()

    call_kwargs = mock_client.list_runs.call_args.kwargs
    assert call_kwargs.get("project_name") == _PROJECT


def test_load_requests_only_top_level_runs():
    """execution_order=1 is always passed to client.list_runs to skip sub-runs."""
    run = _make_chat_run()
    mock_client = _make_mock_client([run])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        adapter.load()

    call_kwargs = mock_client.list_runs.call_args.kwargs
    assert call_kwargs.get("execution_order") == 1


def test_load_skips_runs_with_no_outputs():
    """Runs where outputs is None or empty dict are skipped."""
    run_no_output = _make_run(
        inputs={"messages": [{"role": "user", "content": "Q"}]},
        outputs=None,
    )
    run_empty_output = _make_run(
        inputs={"messages": [{"role": "user", "content": "Q"}]},
        outputs={},
    )
    good_run = _make_chat_run("Valid Q", "Valid A")
    mock_client = _make_mock_client([run_no_output, run_empty_output, good_run])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1
    assert result[0].response == "Valid A"


# ---------------------------------------------------------------------------
# Edge cases (8)
# ---------------------------------------------------------------------------


def test_load_empty_project_returns_empty_list():
    """An empty run list yields an empty PromptPair list (no error)."""
    mock_client = _make_mock_client([])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result == []


def test_load_skips_run_with_missing_inputs():
    """A run with inputs=None is skipped without raising an exception."""
    bad_run = _make_run(inputs=None, outputs={"output": "A"})
    good_run = _make_chat_run("Good Q", "Good A")
    mock_client = _make_mock_client([bad_run, good_run])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1
    assert result[0].response == "Good A"


def test_load_skips_run_with_empty_inputs():
    """A run with inputs={} (no recognised key) is skipped without raising."""
    bad_run = _make_run(inputs={}, outputs={"output": "A"})
    good_run = _make_chat_run()
    mock_client = _make_mock_client([bad_run, good_run])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1


def test_load_handles_mix_of_valid_and_invalid_runs():
    """Valid and invalid runs are processed independently; only valid ones are returned."""
    runs = [
        _make_chat_run("Valid 1", "Resp 1"),
        _make_run(inputs=None, outputs={"output": "Orphan"}),
        _make_chat_run("Valid 2", "Resp 2"),
        _make_run(inputs={}, outputs={}),
    ]
    mock_client = _make_mock_client(runs)
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 2
    responses = {p.response for p in result}
    assert responses == {"Resp 1", "Resp 2"}


def test_load_logs_warning_for_skipped_runs(caplog):
    """A structural warning (no content) is emitted when runs are skipped."""
    bad_run = _make_run(inputs=None, outputs={"output": "A"})
    good_run = _make_chat_run()
    mock_client = _make_mock_client([bad_run, good_run])
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        with caplog.at_level(logging.WARNING, logger="rosettastone.ingest.langsmith_adapter"):
            adapter.load()

    assert any("skipped" in record.message.lower() for record in caplog.records)


def test_load_handles_pagination_across_multiple_pages():
    """list_runs may return many runs — all of them are processed (iterator exhausted)."""
    # Simulate 50 runs coming back from the iterator
    runs = [_make_chat_run(f"Q{i}", f"A{i}") for i in range(50)]
    mock_client = _make_mock_client(runs)
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 50


def test_load_uses_env_var_api_key_when_not_provided(monkeypatch):
    """LANGCHAIN_API_KEY env var is used when api_key=None is passed to the constructor."""
    monkeypatch.setenv("LANGCHAIN_API_KEY", "env_key_value")
    run = _make_chat_run()
    mock_client = _make_mock_client([run])
    adapter = LangSmithAdapter(_PROJECT, source_model=_SOURCE_MODEL)  # no api_key
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1
    assert adapter._api_key == "env_key_value"


def test_load_sets_source_model_on_all_pairs():
    """All returned PromptPairs have the adapter's source_model when no run-level model found."""
    runs = [_make_chat_run(f"Q{i}", f"A{i}") for i in range(3)]
    mock_client = _make_mock_client(runs)
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert all(p.source_model == _SOURCE_MODEL for p in result)


# ---------------------------------------------------------------------------
# ImportError tests (3)
# ---------------------------------------------------------------------------


def test_import_error_has_clear_message():
    """Missing 'langsmith' package raises ImportError with a descriptive message."""
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.dict("sys.modules", {"langsmith": None}):
        with pytest.raises(ImportError, match="langsmith"):
            adapter._make_client()


def test_import_error_includes_install_suggestion():
    """The ImportError message tells the user how to install the package."""
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.dict("sys.modules", {"langsmith": None}):
        with pytest.raises(ImportError, match="pip install langsmith"):
            adapter._make_client()


def test_constructor_works_without_langsmith_installed():
    """LangSmithAdapter can be constructed even if langsmith is not installed."""
    with patch.dict("sys.modules", {"langsmith": None}):
        # Should not raise at construction time — lazy import only
        adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    assert adapter is not None


# ---------------------------------------------------------------------------
# Prompt format tests (4)
# ---------------------------------------------------------------------------


def test_prompt_from_chat_messages_is_list_of_dicts():
    """When inputs contains 'messages', the prompt is stored as a list[dict] (not a string)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Tell me a joke"},
    ]
    run = _make_run(inputs={"messages": messages}, outputs={"output": "Why did the chicken..."})
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert isinstance(result[0].prompt, list)
    assert result[0].prompt == messages


def test_prompt_from_string_prompt_key():
    """When inputs contains 'prompt' (string), the prompt is stored as a string."""
    run = _make_run(
        inputs={"prompt": "A plain string prompt"},
        outputs={"output": "Response"},
    )
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert isinstance(result[0].prompt, str)
    assert result[0].prompt == "A plain string prompt"


def test_prompt_from_multi_role_messages():
    """Multi-turn conversation messages (user + assistant + user) are preserved as-is."""
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "A programming language."},
        {"role": "user", "content": "Can you elaborate?"},
    ]
    run = _make_run(inputs={"messages": messages}, outputs={"output": "Sure, Python is..."})
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert result[0].prompt == messages
    assert len(result[0].prompt) == 3  # type: ignore[arg-type]


def test_response_dict_is_coerced_to_string():
    """When run.outputs['output'] is a dict, it is coerced to a string for PromptPair.response."""
    dict_response = {"text": "Hello world", "stop_reason": "end_turn"}
    run = _make_run(
        inputs={"messages": [{"role": "user", "content": "Hi"}]},
        outputs={"output": dict_response},
    )
    adapter = LangSmithAdapter(_PROJECT, api_key=_API_KEY, source_model=_SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=_make_mock_client([run])):
        result = adapter.load()

    assert isinstance(result[0].response, str)
    assert len(result[0].response) > 0
