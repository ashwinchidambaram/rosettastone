"""Tests for the OTelAdapter ingest module."""

from __future__ import annotations

import json
import logging

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.ingest.otel_adapter import OTelAdapter

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

_SOURCE_MODEL = "openai/gpt-4o"


def _make_span(
    name: str = "chat gpt-4o",
    trace_id: str = "abc123",
    attributes: list[dict] | None = None,
    events: list[dict] | None = None,
) -> dict:
    """Build a minimal OTLP span dict."""
    span: dict = {"name": name, "traceId": trace_id}
    if attributes is not None:
        span["attributes"] = attributes
    if events is not None:
        span["events"] = events
    return span


def _attr(key: str, value: str) -> dict:
    """Build an OTLP attribute entry with a stringValue."""
    return {"key": key, "value": {"stringValue": value}}


def _make_otlp(spans: list[dict]) -> dict:
    """Wrap spans in a minimal OTLP resourceSpans envelope."""
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": spans,
                    }
                ]
            }
        ]
    }


def _write_json(path, data: dict) -> None:
    """Write a dict as JSON to path."""
    with open(path, "w") as f:
        json.dump(data, f)


def _gen_ai_span(
    prompt: str = "What is 2+2?",
    completion: str = "4",
    model: str = "gpt-4o",
    name: str = "chat gpt-4o",
    trace_id: str = "abc123",
) -> dict:
    """Build a complete gen_ai span using span attributes."""
    return _make_span(
        name=name,
        trace_id=trace_id,
        attributes=[
            _attr("gen_ai.request.model", model),
            _attr("gen_ai.prompt", prompt),
            _attr("gen_ai.completion", completion),
        ],
    )


def _gen_ai_span_events(
    prompt: str = "What is 2+2?",
    completion: str = "4",
    model: str = "gpt-4o",
    name: str = "chat gpt-4o",
    trace_id: str = "abc123",
) -> dict:
    """Build a gen_ai span using span events (no inline attributes for prompt/completion)."""
    return _make_span(
        name=name,
        trace_id=trace_id,
        attributes=[
            _attr("gen_ai.request.model", model),
        ],
        events=[
            {
                "name": "gen_ai.content.prompt",
                "attributes": [_attr("gen_ai.prompt", prompt)],
            },
            {
                "name": "gen_ai.content.completion",
                "attributes": [_attr("gen_ai.completion", completion)],
            },
        ],
    )


# ===========================================================================
# Load tests (8)
# ===========================================================================


def test_load_single_file_returns_prompt_pairs(tmp_path):
    """This test proves that a single OTLP JSON file with one gen_ai span loads one PromptPair."""
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp([_gen_ai_span()]))

    adapter = OTelAdapter(otlp_file, _SOURCE_MODEL)
    result = adapter.load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert isinstance(result[0], PromptPair), (
        f"Expected PromptPair instance, got: {type(result[0])}"
    )


def test_load_multiple_spans_in_one_file(tmp_path):
    """This test proves that multiple gen_ai spans in a single file each produce a PromptPair."""
    spans = [
        _gen_ai_span(prompt="Q1", completion="A1", trace_id="t1"),
        _gen_ai_span(prompt="Q2", completion="A2", trace_id="t2"),
        _gen_ai_span(prompt="Q3", completion="A3", trace_id="t3"),
    ]
    otlp_file = tmp_path / "multi.json"
    _write_json(otlp_file, _make_otlp(spans))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 3, f"Expected 3 pairs, got {len(result)}"


def test_load_directory_of_files(tmp_path):
    """This test proves that a directory path loads spans from all .json files in it."""
    _write_json(tmp_path / "a.json", _make_otlp([_gen_ai_span(prompt="Q1", completion="A1")]))
    _write_json(tmp_path / "b.json", _make_otlp([_gen_ai_span(prompt="Q2", completion="A2")]))

    result = OTelAdapter(tmp_path, _SOURCE_MODEL).load()

    assert len(result) == 2, f"Expected 2 pairs from 2 files, got {len(result)}"


def test_load_prompt_from_attributes(tmp_path):
    """This test proves that gen_ai.prompt attribute on the span sets the PromptPair prompt."""
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp([_gen_ai_span(prompt="What is the capital of France?")]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result[0].prompt == "What is the capital of France?", (
        f"Expected prompt from attribute, got: {result[0].prompt!r}"
    )


def test_load_response_from_attributes(tmp_path):
    """This test proves that gen_ai.completion attribute on the span sets the PromptPair response."""
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp([_gen_ai_span(completion="Paris")]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result[0].response == "Paris", (
        f"Expected response from attribute, got: {result[0].response!r}"
    )


def test_load_model_from_span_attribute(tmp_path):
    """This test proves that gen_ai.request.model attribute is used as source_model on the pair."""
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp([_gen_ai_span(model="gpt-4-turbo")]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result[0].source_model == "gpt-4-turbo", (
        f"Expected model from span attribute, got: {result[0].source_model!r}"
    )


def test_load_fallback_model_when_attribute_missing(tmp_path):
    """This test proves that constructor source_model is used when span lacks gen_ai.request.model."""
    span = _make_span(
        attributes=[
            _attr("gen_ai.prompt", "Hello"),
            _attr("gen_ai.completion", "World"),
        ]
    )
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, "fallback/model").load()

    assert result[0].source_model == "fallback/model", (
        f"Expected fallback source_model, got: {result[0].source_model!r}"
    )


def test_load_span_metadata_in_pair(tmp_path):
    """This test proves that span name and traceId are stored in the PromptPair metadata."""
    span = _gen_ai_span(name="chat gpt-4o", trace_id="deadbeef")
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    meta = result[0].metadata
    assert meta.get("span_name") == "chat gpt-4o", f"Expected span_name in metadata, got: {meta!r}"
    assert meta.get("trace_id") == "deadbeef", f"Expected trace_id in metadata, got: {meta!r}"


# ===========================================================================
# Span parsing tests (8)
# ===========================================================================


def test_full_otlp_structure_traversal(tmp_path):
    """This test proves that nested resourceSpans > scopeSpans > spans is fully traversed."""
    # Two resourceSpans, each with one scopeSpans, each with one span
    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {"spans": [_gen_ai_span(prompt="Q1", completion="A1", trace_id="t1")]}
                ]
            },
            {
                "scopeSpans": [
                    {"spans": [_gen_ai_span(prompt="Q2", completion="A2", trace_id="t2")]}
                ]
            },
        ]
    }
    otlp_file = tmp_path / "multi_resource.json"
    _write_json(otlp_file, data)

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 2, f"Expected 2 pairs from 2 resourceSpans, got {len(result)}"


def test_events_used_as_fallback_for_prompt_completion(tmp_path):
    """This test proves that span events provide prompt/completion when inline attributes are absent."""
    span = _gen_ai_span_events(prompt="Event-based prompt", completion="Event-based answer")
    otlp_file = tmp_path / "events.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 1, f"Expected 1 pair from event-based span, got {len(result)}"
    assert result[0].prompt == "Event-based prompt", (
        f"Expected prompt from event, got: {result[0].prompt!r}"
    )
    assert result[0].response == "Event-based answer", (
        f"Expected response from event, got: {result[0].response!r}"
    )


def test_non_gen_ai_spans_are_silently_ignored(tmp_path):
    """This test proves that spans without any gen_ai attributes or events are skipped."""
    non_gen_ai = _make_span(
        name="http GET /api/data",
        attributes=[
            _attr("http.method", "GET"),
            _attr("http.url", "https://example.com/api"),
        ],
    )
    gen_ai = _gen_ai_span(prompt="Valid prompt", completion="Valid response")
    otlp_file = tmp_path / "mixed.json"
    _write_json(otlp_file, _make_otlp([non_gen_ai, gen_ai]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 1, (
        f"Expected only 1 gen_ai pair (non-gen_ai span ignored), got {len(result)}"
    )


def test_gen_ai_span_missing_both_attributes_and_events_skipped(tmp_path):
    """This test proves spans with gen_ai.request.model but no prompt/completion are skipped."""
    span = _make_span(
        name="chat gpt-4o",
        attributes=[
            _attr("gen_ai.request.model", "gpt-4o"),
            # No gen_ai.prompt or gen_ai.completion
        ],
    )
    otlp_file = tmp_path / "no_content.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result == [], f"Expected empty list when span has no prompt/completion, got: {result!r}"


def test_attribute_format_takes_precedence_over_events(tmp_path):
    """This test proves inline attributes are preferred over events when both are present."""
    span = _make_span(
        name="chat gpt-4o",
        attributes=[
            _attr("gen_ai.request.model", "gpt-4o"),
            _attr("gen_ai.prompt", "Inline prompt"),
            _attr("gen_ai.completion", "Inline completion"),
        ],
        events=[
            {
                "name": "gen_ai.content.prompt",
                "attributes": [_attr("gen_ai.prompt", "Event prompt")],
            },
            {
                "name": "gen_ai.content.completion",
                "attributes": [_attr("gen_ai.completion", "Event completion")],
            },
        ],
    )
    otlp_file = tmp_path / "both.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result[0].prompt == "Inline prompt", (
        f"Expected inline attribute prompt to win, got: {result[0].prompt!r}"
    )
    assert result[0].response == "Inline completion", (
        f"Expected inline attribute completion to win, got: {result[0].response!r}"
    )


def test_chat_messages_json_string_prompt(tmp_path):
    """This test proves that a JSON-encoded messages list in gen_ai.prompt is parsed to list."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    span = _make_span(
        name="chat gpt-4o",
        attributes=[
            _attr("gen_ai.request.model", "gpt-4o"),
            _attr("gen_ai.prompt", json.dumps(messages)),
            _attr("gen_ai.completion", "4"),
        ],
    )
    otlp_file = tmp_path / "messages.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert isinstance(result[0].prompt, list), (
        f"Expected prompt to be parsed as list of messages, got: {type(result[0].prompt)}"
    )
    assert result[0].prompt == messages, f"Expected messages list, got: {result[0].prompt!r}"


def test_string_prompt_stays_as_string(tmp_path):
    """This test proves that a plain string gen_ai.prompt stays as a string PromptPair.prompt."""
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp([_gen_ai_span(prompt="Plain string prompt")]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert isinstance(result[0].prompt, str), (
        f"Expected prompt to remain a string, got: {type(result[0].prompt)}"
    )
    assert result[0].prompt == "Plain string prompt", (
        f"Expected plain string prompt, got: {result[0].prompt!r}"
    )


def test_multiple_resource_spans_all_loaded(tmp_path):
    """This test proves that all resourceSpans entries in one file contribute pairs."""
    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            _gen_ai_span(prompt="Q1", completion="A1", trace_id="t1"),
                            _gen_ai_span(prompt="Q2", completion="A2", trace_id="t2"),
                        ]
                    }
                ]
            },
            {
                "scopeSpans": [
                    {"spans": [_gen_ai_span(prompt="Q3", completion="A3", trace_id="t3")]}
                ]
            },
        ]
    }
    otlp_file = tmp_path / "multi.json"
    _write_json(otlp_file, data)

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 3, f"Expected 3 pairs across multiple resourceSpans, got {len(result)}"


# ===========================================================================
# Edge case tests (8)
# ===========================================================================


def test_empty_json_object_file_returns_empty_list(tmp_path):
    """This test proves that a file containing only '{}' (valid JSON, no spans) returns []."""
    otlp_file = tmp_path / "empty.json"
    _write_json(otlp_file, {})

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result == [], f"Expected empty list from empty OTLP envelope, got: {result!r}"


def test_malformed_json_file_is_skipped_not_raised(tmp_path):
    """This test proves that a malformed JSON file is skipped and other files still load."""
    _write_json(tmp_path / "good.json", _make_otlp([_gen_ai_span()]))
    (tmp_path / "bad.json").write_text("this is not json")

    result = OTelAdapter(tmp_path, _SOURCE_MODEL).load()

    # Good file still loaded; bad file skipped
    assert len(result) == 1, f"Expected 1 pair (malformed file skipped), got {len(result)}"


def test_valid_otlp_with_no_gen_ai_spans_returns_empty(tmp_path):
    """This test proves that a valid OTLP file with zero gen_ai spans returns []."""
    data = _make_otlp(
        [
            _make_span(
                name="db.query",
                attributes=[_attr("db.system", "postgresql"), _attr("db.statement", "SELECT 1")],
            ),
            _make_span(
                name="http.request",
                attributes=[_attr("http.method", "POST")],
            ),
        ]
    )
    otlp_file = tmp_path / "infra.json"
    _write_json(otlp_file, data)

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result == [], f"Expected empty list when no gen_ai spans present, got: {result!r}"


def test_span_with_prompt_but_missing_completion_is_skipped(tmp_path):
    """This test proves that a span with gen_ai.prompt but no gen_ai.completion is skipped."""
    span = _make_span(
        name="incomplete span",
        attributes=[
            _attr("gen_ai.request.model", "gpt-4o"),
            _attr("gen_ai.prompt", "This has no completion"),
        ],
    )
    otlp_file = tmp_path / "no_completion.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result == [], f"Expected empty list when span missing completion, got: {result!r}"


def test_span_with_completion_but_missing_prompt_is_skipped(tmp_path):
    """This test proves that a span with gen_ai.completion but no gen_ai.prompt is skipped."""
    span = _make_span(
        name="incomplete span",
        attributes=[
            _attr("gen_ai.request.model", "gpt-4o"),
            _attr("gen_ai.completion", "This has no prompt"),
        ],
    )
    otlp_file = tmp_path / "no_prompt.json"
    _write_json(otlp_file, _make_otlp([span]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert result == [], f"Expected empty list when span missing prompt, got: {result!r}"


def test_mixed_valid_and_invalid_spans_loads_only_valid(tmp_path):
    """This test proves that valid spans load and incomplete spans are skipped in the same file."""
    spans = [
        _gen_ai_span(prompt="Valid Q", completion="Valid A", trace_id="valid"),
        _make_span(
            name="no completion",
            attributes=[
                _attr("gen_ai.request.model", "gpt-4o"),
                _attr("gen_ai.prompt", "No answer here"),
            ],
        ),
        _make_span(
            name="http span",
            attributes=[_attr("http.method", "GET")],
        ),
    ]
    otlp_file = tmp_path / "mixed.json"
    _write_json(otlp_file, _make_otlp(spans))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 1, f"Expected only 1 valid pair from mixed spans, got {len(result)}"
    assert result[0].prompt == "Valid Q", (
        f"Expected the valid span's prompt, got: {result[0].prompt!r}"
    )


def test_structural_logging_for_skipped_spans(tmp_path, caplog):
    """This test proves a warning is logged structurally when spans are skipped."""
    spans = [
        _gen_ai_span(prompt="Valid", completion="OK"),
        _make_span(
            name="incomplete",
            attributes=[
                _attr("gen_ai.request.model", "gpt-4o"),
                _attr("gen_ai.prompt", "No completion"),
            ],
        ),
    ]
    otlp_file = tmp_path / "trace.json"
    _write_json(otlp_file, _make_otlp(spans))

    with caplog.at_level(logging.WARNING, logger="rosettastone.ingest.otel_adapter"):
        OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert any("skipped" in record.message.lower() for record in caplog.records), (
        "Expected a warning log mentioning 'skipped' for incomplete spans"
    )


def test_large_file_with_many_spans(tmp_path):
    """This test proves that a file with 200 gen_ai spans loads all 200 pairs."""
    spans = [
        _gen_ai_span(prompt=f"Question {i}", completion=f"Answer {i}", trace_id=f"t{i}")
        for i in range(200)
    ]
    otlp_file = tmp_path / "large.json"
    _write_json(otlp_file, _make_otlp(spans))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 200, f"Expected 200 pairs from large file, got {len(result)}"


# ===========================================================================
# File discovery tests (6)
# ===========================================================================


def test_single_json_file_path_accepted(tmp_path):
    """This test proves that a Path pointing directly to a .json file is accepted."""
    otlp_file = tmp_path / "single.json"
    _write_json(otlp_file, _make_otlp([_gen_ai_span()]))

    result = OTelAdapter(otlp_file, _SOURCE_MODEL).load()

    assert len(result) == 1, f"Expected 1 pair from single file path, got {len(result)}"


def test_directory_with_multiple_json_files(tmp_path):
    """This test proves that all .json files in a directory are read."""
    for i in range(3):
        _write_json(
            tmp_path / f"trace_{i}.json",
            _make_otlp([_gen_ai_span(prompt=f"Q{i}", completion=f"A{i}", trace_id=f"t{i}")]),
        )

    result = OTelAdapter(tmp_path, _SOURCE_MODEL).load()

    assert len(result) == 3, f"Expected 3 pairs from 3 .json files, got {len(result)}"


def test_non_json_files_ignored_in_directory(tmp_path):
    """This test proves that .txt, .jsonl, and other non-.json files in a directory are ignored."""
    _write_json(tmp_path / "trace.json", _make_otlp([_gen_ai_span()]))
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "data.jsonl").write_text('{"not": "otlp"}\n')
    (tmp_path / "README.md").write_text("# nope")

    result = OTelAdapter(tmp_path, _SOURCE_MODEL).load()

    assert len(result) == 1, f"Expected only 1 pair (non-.json files ignored), got {len(result)}"


def test_empty_directory_returns_empty_list(tmp_path):
    """This test proves that an empty directory yields an empty list, not an error."""
    result = OTelAdapter(tmp_path, _SOURCE_MODEL).load()

    assert result == [], f"Expected empty list from empty directory, got: {result!r}"


def test_nonexistent_path_raises_file_not_found_error(tmp_path):
    """This test proves that a path that does not exist raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        OTelAdapter(missing, _SOURCE_MODEL).load()


def test_no_recursion_into_subdirectories(tmp_path):
    """This test proves that only top-level .json files are read, not subdirectory files."""
    # Top-level file — should load
    _write_json(tmp_path / "top.json", _make_otlp([_gen_ai_span(prompt="Top", completion="Yes")]))

    # Subdirectory file — should NOT be loaded
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    _write_json(
        subdir / "nested.json", _make_otlp([_gen_ai_span(prompt="Nested", completion="No")])
    )

    result = OTelAdapter(tmp_path, _SOURCE_MODEL).load()

    assert len(result) == 1, (
        f"Expected only 1 pair (no recursion into subdirectories), got {len(result)}"
    )
    assert result[0].prompt == "Top", f"Expected top-level span prompt, got: {result[0].prompt!r}"
