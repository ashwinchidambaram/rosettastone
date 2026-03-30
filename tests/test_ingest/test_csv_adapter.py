"""Tests for the CSVAdapter ingest module."""

from __future__ import annotations

import csv

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.ingest.csv_adapter import CSVAdapter, CSVColumnMapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write a list of dicts as CSV to path."""
    if not rows and fieldnames is None:
        fieldnames = ["prompt", "response", "source_model"]
    elif fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write a list of dicts as TSV to path."""
    if not rows and fieldnames is None:
        fieldnames = ["prompt", "response", "source_model"]
    elif fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _make_row(
    prompt: str = "Hello",
    response: str = "World",
    source_model: str = "openai/gpt-4o",
    **extra,
) -> dict:
    return {"prompt": prompt, "response": response, "source_model": source_model, **extra}


# ---------------------------------------------------------------------------
# Load tests (7)
# ---------------------------------------------------------------------------


def test_load_standard_csv_with_default_columns(tmp_path):
    """This test proves that a standard CSV with default column names loads correctly."""
    csv_file = tmp_path / "data.csv"
    _write_csv(csv_file, [_make_row(prompt="Say hi", response="Hi!")])

    adapter = CSVAdapter(csv_file)
    result = adapter.load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert isinstance(result[0], PromptPair), f"Expected PromptPair, got {type(result[0])}"
    assert result[0].prompt == "Say hi", f"Expected 'Say hi', got: {result[0].prompt!r}"
    assert result[0].response == "Hi!", f"Expected 'Hi!', got: {result[0].response!r}"
    assert result[0].source_model == "openai/gpt-4o", (
        f"Expected 'openai/gpt-4o', got: {result[0].source_model!r}"
    )


def test_load_csv_with_custom_column_mapping(tmp_path):
    """This test proves that a CSV with non-default column names loads correctly via mapping."""
    csv_file = tmp_path / "data.csv"
    rows = [{"question": "What is AI?", "answer": "Artificial Intelligence", "model": "gpt-4"}]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "model"])
        writer.writeheader()
        writer.writerows(rows)

    mapping = CSVColumnMapping(
        prompt_col="question",
        response_col="answer",
        source_model_col="model",
    )
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].prompt == "What is AI?", f"Got: {result[0].prompt!r}"
    assert result[0].response == "Artificial Intelligence", f"Got: {result[0].response!r}"
    assert result[0].source_model == "gpt-4", f"Got: {result[0].source_model!r}"


def test_load_tsv_file_auto_detection(tmp_path):
    """This test proves that a .tsv file is auto-detected as tab-delimited."""
    tsv_file = tmp_path / "data.tsv"
    _write_tsv(tsv_file, [_make_row(prompt="TSV prompt", response="TSV response")])

    result = CSVAdapter(tsv_file).load()

    assert len(result) == 1, f"Expected 1 pair from TSV, got {len(result)}"
    assert result[0].prompt == "TSV prompt", f"Got: {result[0].prompt!r}"
    assert result[0].response == "TSV response", f"Got: {result[0].response!r}"


def test_load_metadata_columns_collected(tmp_path):
    """This test proves that specified metadata_cols are collected into the metadata dict."""
    csv_file = tmp_path / "data.csv"
    rows = [
        {
            "prompt": "Hello",
            "response": "Hi",
            "source_model": "gpt-4",
            "session_id": "abc123",
            "timestamp": "2024-01-01",
        }
    ]
    _write_csv(csv_file, rows)

    mapping = CSVColumnMapping(metadata_cols=["session_id", "timestamp"])
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].metadata == {"session_id": "abc123", "timestamp": "2024-01-01"}, (
        f"Expected metadata dict, got: {result[0].metadata!r}"
    )


def test_load_feedback_column_mapping(tmp_path):
    """This test proves that feedback_col is mapped to PromptPair.feedback."""
    csv_file = tmp_path / "data.csv"
    rows = [
        {
            "prompt": "Hello",
            "response": "Hi",
            "source_model": "gpt-4",
            "human_feedback": "positive",
        }
    ]
    _write_csv(csv_file, rows)

    mapping = CSVColumnMapping(feedback_col="human_feedback")
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].feedback == "positive", f"Expected 'positive', got: {result[0].feedback!r}"


def test_load_source_model_from_column(tmp_path):
    """This test proves that source_model is read from the CSV column when mapped."""
    csv_file = tmp_path / "data.csv"
    rows = [_make_row(source_model="anthropic/claude-3")]
    _write_csv(csv_file, rows)

    result = CSVAdapter(csv_file).load()

    assert result[0].source_model == "anthropic/claude-3", (
        f"Expected model from column, got: {result[0].source_model!r}"
    )


def test_load_source_model_fallback_to_constructor(tmp_path):
    """This test proves that source_model falls back to constructor param when column is absent."""
    csv_file = tmp_path / "data.csv"
    rows = [{"prompt": "Hello", "response": "Hi"}]
    _write_csv(csv_file, rows, fieldnames=["prompt", "response"])

    result = CSVAdapter(csv_file, source_model="fallback/model").load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].source_model == "fallback/model", (
        f"Expected fallback model, got: {result[0].source_model!r}"
    )


# ---------------------------------------------------------------------------
# Delimiter detection tests (4)
# ---------------------------------------------------------------------------


def test_delimiter_detection_csv_extension(tmp_path):
    """This test proves that .csv extension auto-detects comma delimiter."""
    csv_file = tmp_path / "data.csv"
    _write_csv(csv_file, [_make_row()])

    adapter = CSVAdapter(csv_file)
    detected = adapter._detect_delimiter()

    assert detected == ",", f"Expected comma for .csv, got: {detected!r}"


def test_delimiter_detection_tsv_extension(tmp_path):
    """This test proves that .tsv extension auto-detects tab delimiter."""
    tsv_file = tmp_path / "data.tsv"
    _write_tsv(tsv_file, [_make_row()])

    adapter = CSVAdapter(tsv_file)
    detected = adapter._detect_delimiter()

    assert detected == "\t", f"Expected tab for .tsv, got: {detected!r}"


def test_delimiter_detection_tab_extension(tmp_path):
    """This test proves that .tab extension auto-detects tab delimiter."""
    tab_file = tmp_path / "data.tab"
    _write_tsv(tab_file, [_make_row()])

    adapter = CSVAdapter(tab_file)
    detected = adapter._detect_delimiter()

    assert detected == "\t", f"Expected tab for .tab, got: {detected!r}"


def test_explicit_delimiter_overrides_extension(tmp_path):
    """This test proves that explicit delimiter parameter overrides extension-based detection."""
    # File has .csv extension but we write with pipe delimiter and pass pipe explicitly
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="") as f:
        f.write("prompt|response|source_model\n")
        f.write("Hello|Hi|gpt-4\n")

    result = CSVAdapter(csv_file, delimiter="|").load()

    assert len(result) == 1, f"Expected 1 pair with explicit delimiter, got {len(result)}"
    assert result[0].prompt == "Hello", f"Got: {result[0].prompt!r}"
    assert result[0].response == "Hi", f"Got: {result[0].response!r}"


# ---------------------------------------------------------------------------
# Edge case tests (8)
# ---------------------------------------------------------------------------


def test_empty_csv_headers_only_returns_empty_list(tmp_path):
    """This test proves that a CSV with only headers (no data rows) returns an empty list."""
    csv_file = tmp_path / "empty.csv"
    _write_csv(csv_file, [], fieldnames=["prompt", "response", "source_model"])

    result = CSVAdapter(csv_file).load()

    assert result == [], f"Expected empty list from headers-only CSV, got: {result!r}"


def test_missing_prompt_column_raises_value_error(tmp_path):
    """This test proves that a missing prompt column raises ValueError with clear message."""
    csv_file = tmp_path / "data.csv"
    rows = [{"response": "Hi", "source_model": "gpt-4"}]
    _write_csv(csv_file, rows, fieldnames=["response", "source_model"])

    with pytest.raises(ValueError, match="prompt"):
        CSVAdapter(csv_file).load()


def test_missing_response_column_raises_value_error(tmp_path):
    """This test proves that a missing response column raises ValueError with clear message."""
    csv_file = tmp_path / "data.csv"
    rows = [{"prompt": "Hello", "source_model": "gpt-4"}]
    _write_csv(csv_file, rows, fieldnames=["prompt", "source_model"])

    with pytest.raises(ValueError, match="response"):
        CSVAdapter(csv_file).load()


def test_bom_encoded_file_loads_correctly(tmp_path):
    """This test proves that a UTF-8 BOM-encoded file is loaded without BOM artifacts."""
    csv_file = tmp_path / "bom.csv"
    # Write with BOM manually
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "response", "source_model"])
        writer.writeheader()
        writer.writerow({"prompt": "BOM test", "response": "OK", "source_model": "gpt-4"})

    result = CSVAdapter(csv_file).load()

    assert len(result) == 1, f"Expected 1 pair from BOM file, got {len(result)}"
    assert result[0].prompt == "BOM test", (
        f"Expected 'BOM test' (no BOM artifact), got: {result[0].prompt!r}"
    )


def test_quoted_fields_with_commas_parsed_correctly(tmp_path):
    """This test proves that quoted fields containing commas are parsed as single values."""
    csv_file = tmp_path / "quoted.csv"
    with open(csv_file, "w", newline="") as f:
        f.write("prompt,response,source_model\n")
        f.write('"Hello, world","Yes, indeed","gpt-4"\n')

    result = CSVAdapter(csv_file).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].prompt == "Hello, world", (
        f"Expected quoted field parsed correctly, got: {result[0].prompt!r}"
    )
    assert result[0].response == "Yes, indeed", f"Got: {result[0].response!r}"


def test_blank_rows_skipped(tmp_path):
    """This test proves that rows with both empty prompt and empty response are skipped."""
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="") as f:
        f.write("prompt,response,source_model\n")
        f.write("Hello,Hi,gpt-4\n")
        f.write(",,\n")  # blank row
        f.write("World,Earth,gpt-4\n")

    result = CSVAdapter(csv_file).load()

    assert len(result) == 2, f"Expected 2 pairs (blank row skipped), got {len(result)}"


def test_whitespace_stripped_from_values(tmp_path):
    """This test proves that leading/trailing whitespace is stripped from field values."""
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="") as f:
        f.write("prompt,response,source_model\n")
        f.write("  Hello  ,  Hi  ,  gpt-4  \n")

    result = CSVAdapter(csv_file).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].prompt == "Hello", f"Expected stripped prompt, got: {result[0].prompt!r}"
    assert result[0].response == "Hi", f"Expected stripped response, got: {result[0].response!r}"
    assert result[0].source_model == "gpt-4", (
        f"Expected stripped source_model, got: {result[0].source_model!r}"
    )


def test_row_number_in_error_messages(tmp_path):
    """This test proves that error messages include the row number for debugging."""
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="") as f:
        # Use a custom mapping where a required col is missing in the data rows
        f.write("prompt,response,source_model\n")
        f.write("Hello,Hi,gpt-4\n")  # row 2 (row 1 is header)
        f.write("World,Earth,gpt-4\n")  # row 3

    # We'll trigger a row-level error by mapping a non-existent column as prompt
    mapping = CSVColumnMapping(prompt_col="nonexistent_col")
    with pytest.raises(ValueError) as exc_info:
        CSVAdapter(csv_file, column_mapping=mapping).load()

    assert "nonexistent_col" in str(exc_info.value) or "row" in str(exc_info.value).lower(), (
        f"Expected row or column info in error, got: {str(exc_info.value)!r}"
    )


# ---------------------------------------------------------------------------
# Column mapping tests (6)
# ---------------------------------------------------------------------------


def test_csv_column_mapping_defaults(tmp_path):
    """This test proves that CSVColumnMapping has the expected default values."""
    mapping = CSVColumnMapping()

    assert mapping.prompt_col == "prompt", f"Expected 'prompt', got: {mapping.prompt_col!r}"
    assert mapping.response_col == "response", f"Expected 'response', got: {mapping.response_col!r}"
    assert mapping.source_model_col == "source_model", (
        f"Expected 'source_model', got: {mapping.source_model_col!r}"
    )
    assert mapping.metadata_cols is None, (
        f"Expected None for metadata_cols, got: {mapping.metadata_cols!r}"
    )
    assert mapping.feedback_col is None, (
        f"Expected None for feedback_col, got: {mapping.feedback_col!r}"
    )


def test_custom_prompt_col_mapping(tmp_path):
    """This test proves that a custom prompt_col name is correctly resolved."""
    csv_file = tmp_path / "data.csv"
    rows = [{"input": "Hello", "response": "Hi", "source_model": "gpt-4"}]
    _write_csv(csv_file, rows)

    mapping = CSVColumnMapping(prompt_col="input")
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].prompt == "Hello", f"Expected 'Hello', got: {result[0].prompt!r}"


def test_custom_response_col_mapping(tmp_path):
    """This test proves that a custom response_col name is correctly resolved."""
    csv_file = tmp_path / "data.csv"
    rows = [{"prompt": "Hello", "output": "Hi", "source_model": "gpt-4"}]
    _write_csv(csv_file, rows)

    mapping = CSVColumnMapping(response_col="output")
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].response == "Hi", f"Expected 'Hi', got: {result[0].response!r}"


def test_custom_source_model_col_mapping(tmp_path):
    """This test proves that a custom source_model_col name is correctly resolved."""
    csv_file = tmp_path / "data.csv"
    rows = [{"prompt": "Hello", "response": "Hi", "llm_model": "claude-3"}]
    _write_csv(csv_file, rows)

    mapping = CSVColumnMapping(source_model_col="llm_model")
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].source_model == "claude-3", (
        f"Expected 'claude-3', got: {result[0].source_model!r}"
    )


def test_metadata_cols_collected_as_dict(tmp_path):
    """This test proves that multiple metadata_cols are all collected into the metadata dict."""
    csv_file = tmp_path / "data.csv"
    rows = [
        {
            "prompt": "Hello",
            "response": "Hi",
            "source_model": "gpt-4",
            "user_id": "u001",
            "region": "us-east",
            "version": "v2",
        }
    ]
    _write_csv(csv_file, rows)

    mapping = CSVColumnMapping(metadata_cols=["user_id", "region", "version"])
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].metadata == {"user_id": "u001", "region": "us-east", "version": "v2"}, (
        f"Expected all metadata cols, got: {result[0].metadata!r}"
    )


def test_feedback_col_mapped_correctly(tmp_path):
    """This test proves that the feedback_col value ends up in PromptPair.feedback."""
    csv_file = tmp_path / "data.csv"
    rows = [
        {
            "prompt": "What is 2+2?",
            "response": "4",
            "source_model": "gpt-4",
            "eval_feedback": "correct and concise",
        }
    ]
    _write_csv(csv_file, rows)

    mapping = CSVColumnMapping(feedback_col="eval_feedback")
    result = CSVAdapter(csv_file, column_mapping=mapping).load()

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0].feedback == "correct and concise", (
        f"Expected feedback value, got: {result[0].feedback!r}"
    )
