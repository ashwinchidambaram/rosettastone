"""Contract tests verifying all report generators share the same interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosettastone.report.html_generator import generate_html_report
from rosettastone.report.markdown import generate_markdown_report
from rosettastone.report.pdf_generator import generate_pdf_report
from tests.factories import eval_result_factory, migration_result_factory, prompt_pair_factory


def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Individual generator acceptance tests
# ---------------------------------------------------------------------------


def test_markdown_generator_accepts_migration_result(tmp_path: Path) -> None:
    """generate_markdown_report returns a Path and the file exists on disk."""
    result = migration_result_factory()
    output = generate_markdown_report(result, tmp_path)

    assert isinstance(output, Path), f"Expected Path return, got {type(output)}"
    assert output.exists(), f"Output file does not exist: {output}"


def test_html_generator_accepts_migration_result(tmp_path: Path) -> None:
    """generate_html_report returns a Path and the file exists on disk."""
    result = migration_result_factory()
    output = generate_html_report(result, tmp_path)

    assert isinstance(output, Path), f"Expected Path return, got {type(output)}"
    assert output.exists(), f"Output file does not exist: {output}"


@pytest.mark.skipif(not _weasyprint_available(), reason="weasyprint not installed")
def test_pdf_generator_accepts_migration_result(tmp_path: Path) -> None:
    """generate_pdf_report returns a Path and the file exists on disk."""
    result = migration_result_factory()
    output = generate_pdf_report(result, tmp_path)

    assert isinstance(output, Path), f"Expected Path return, got {type(output)}"
    assert output.exists(), f"Output file does not exist: {output}"


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------


def test_all_generators_create_output_dir(tmp_path: Path) -> None:
    """All generators create the output directory if it does not already exist."""
    result = migration_result_factory()

    md_dir = tmp_path / "subdir_md" / "nested"
    generate_markdown_report(result, md_dir)
    assert md_dir.is_dir(), f"Markdown generator did not create: {md_dir}"

    html_dir = tmp_path / "subdir_html" / "nested"
    generate_html_report(result, html_dir)
    assert html_dir.is_dir(), f"HTML generator did not create: {html_dir}"


# ---------------------------------------------------------------------------
# Empty validation_results — no crash
# ---------------------------------------------------------------------------


def test_all_generators_handle_empty_validation_results(tmp_path: Path) -> None:
    """All generators complete without error when validation_results is empty."""
    result = migration_result_factory(validation_results=[])

    md_path = generate_markdown_report(result, tmp_path / "md_empty")
    assert md_path.exists()

    html_path = generate_html_report(result, tmp_path / "html_empty")
    assert html_path.exists()


# ---------------------------------------------------------------------------
# PII safety — raw prompt text must not appear in output
# ---------------------------------------------------------------------------

_SENTINEL_PROMPT = "SENTINEL_PROMPT_SHOULD_NOT_APPEAR_IN_REPORT_XYZ_99999"


def _sentinel_result() -> object:
    """Return a MigrationResult whose validation pair uses the sentinel prompt."""
    sentinel_pair = prompt_pair_factory(
        prompt=_SENTINEL_PROMPT,
        response="expected response",
    )
    return migration_result_factory(
        validation_results=[
            eval_result_factory(
                prompt_pair=sentinel_pair,
                new_response="generated response",
            )
        ]
    )


def test_markdown_output_no_raw_prompt(tmp_path: Path) -> None:
    """Markdown report does not echo raw prompt text into output (PII safety)."""
    result = _sentinel_result()
    output = generate_markdown_report(result, tmp_path)
    content = output.read_text(encoding="utf-8")

    assert _SENTINEL_PROMPT not in content, (
        f"Raw prompt text found in markdown output — potential PII leak: {output}"
    )


def test_html_output_no_raw_prompt(tmp_path: Path) -> None:
    """HTML report does not echo raw prompt text into output (PII safety)."""
    result = _sentinel_result()
    output = generate_html_report(result, tmp_path)
    content = output.read_text(encoding="utf-8")

    assert _SENTINEL_PROMPT not in content, (
        f"Raw prompt text found in HTML output — potential PII leak: {output}"
    )


# ---------------------------------------------------------------------------
# Parametrized: correct file suffix
# ---------------------------------------------------------------------------

_GENERATORS = [
    (generate_markdown_report, ".md"),
    (generate_html_report, ".html"),
]


@pytest.mark.parametrize("generator,suffix", _GENERATORS)
def test_generators_return_correct_suffix(generator, suffix: str, tmp_path: Path) -> None:
    """Each generator returns a file with the expected extension."""
    result = migration_result_factory()
    # Use unique subdirs to avoid filename collisions across parametrize runs
    subdir = tmp_path / suffix.lstrip(".")
    path = generator(result, subdir)

    assert path.suffix == suffix, (
        f"{generator.__name__} returned path with suffix {path.suffix!r}, expected {suffix!r}"
    )
