"""Test that no logging call passes prompt/response content variables.

This asserts the invariant: "never log prompt content at any level".
"""

from __future__ import annotations

import ast
from pathlib import Path

# Variable names that are high-confidence PII carriers (exact matches only).
# These are names that almost always hold raw prompt/response strings.
# We use exact matching (not substring) to avoid false positives from names
# like `output_dir`, `output_type`, `expected_output_type`, etc.
PII_VAR_NAMES = {"prompt", "response", "input", "user_input", "system_prompt"}

# Additional names that commonly carry raw message content in exception paths.
# Kept separate so the two invariant checks can be tuned independently.
RAISE_PII_VAR_NAMES = PII_VAR_NAMES | {"message_content"}

# Logging function names to scan
LOG_FUNC_NAMES = {"debug", "info", "warning", "error", "critical", "exception", "log"}

SRC_DIR = Path(__file__).parent.parent / "src" / "rosettastone"


def _is_pii_arg(node: ast.expr, var_names: frozenset[str] | set[str] = PII_VAR_NAMES) -> bool:
    """Return True if an AST node looks like a PII variable reference.

    We only flag exact-name matches on ``ast.Name`` nodes (local variables),
    not attribute accesses.  This avoids false positives for benign names like
    ``entry.output_dir``, ``template.expected_output_type``, or ``output_base``
    that contain PII-ish substrings but carry no actual prompt content.
    """
    if isinstance(node, ast.Name):
        return node.id.lower() in var_names
    return False


def _find_logging_calls_with_pii(filepath: Path) -> list[tuple[int, str]]:
    """Return list of (lineno, description) for suspicious logging calls."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match logger.info(...), log.debug(...), logging.warning(...) etc.
        is_log_call = False
        if isinstance(node.func, ast.Attribute):
            is_log_call = node.func.attr in LOG_FUNC_NAMES
        if not is_log_call:
            continue
        # Check all arguments (positional and format args in %-style)
        for arg in node.args[1:]:  # skip the format string (first arg)
            if _is_pii_arg(arg):
                violations.append(
                    (node.lineno, f"Logging call passes PII variable: {ast.unparse(arg)!r}")
                )
    return violations


def _find_raise_statements_with_pii(filepath: Path) -> list[tuple[int, str]]:
    """Return list of (lineno, description) for raise statements that embed PII variables.

    A raise statement can leak prompt/response content into exception messages
    which may surface in logs, error responses, or stack traces.  We check
    whether any PII-named variable appears as a direct argument to an exception
    constructor inside a ``raise`` statement.

    Only ``ast.Name`` nodes are checked (exact local-variable name match) to
    avoid false positives on attribute accesses such as ``config.prompt_template``.
    """
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if exc is None:
            continue  # bare re-raise — no argument to inspect

        # Gather all Name nodes that appear as arguments to the raised expression.
        # This covers: raise ValueError(prompt), raise SomeError(f"...", response), etc.
        # We walk all child nodes of the exception expression to catch f-string
        # interpolations and nested calls.
        for child in ast.walk(exc):
            if _is_pii_arg(child, var_names=RAISE_PII_VAR_NAMES):
                violations.append(
                    (
                        node.lineno,
                        f"raise statement embeds PII variable in exception: "
                        f"{ast.unparse(child)!r}",
                    )
                )
                break  # one violation per raise statement is enough

    return violations


def test_no_pii_in_logging_calls() -> None:
    """No logging call should pass prompt/response content variables."""
    all_violations: list[tuple[Path, int, str]] = []
    for py_file in SRC_DIR.rglob("*.py"):
        violations = _find_logging_calls_with_pii(py_file)
        for lineno, desc in violations:
            all_violations.append((py_file, lineno, desc))

    if all_violations:
        lines = ["PII invariant violations found:"]
        for filepath, lineno, desc in all_violations:
            rel = filepath.relative_to(SRC_DIR.parent.parent)
            lines.append(f"  {rel}:{lineno}: {desc}")
        raise AssertionError("\n".join(lines))


def test_no_pii_in_exception_messages() -> None:
    """No raise statement should embed prompt/response content variables in exception args.

    Exception messages can surface in logs, HTTP error responses, and tracebacks.
    Passing raw prompt or response variables directly into exception constructors
    is therefore treated the same as logging them — a PII leak.

    Checked variable names: prompt, response, input, user_input, system_prompt,
    message_content (all exact local-variable name matches; attribute accesses
    are ignored to avoid false positives).
    """
    all_violations: list[tuple[Path, int, str]] = []
    for py_file in SRC_DIR.rglob("*.py"):
        violations = _find_raise_statements_with_pii(py_file)
        for lineno, desc in violations:
            all_violations.append((py_file, lineno, desc))

    if all_violations:
        lines = ["PII invariant violations found in exception messages:"]
        for filepath, lineno, desc in all_violations:
            rel = filepath.relative_to(SRC_DIR.parent.parent)
            lines.append(f"  {rel}:{lineno}: {desc}")
        raise AssertionError("\n".join(lines))
