"""Structured JSON logging configuration for the RosettaStone server."""

from __future__ import annotations

import contextvars
import datetime
import json
import logging
import os

# ---------------------------------------------------------------------------
# Context variable for request ID propagation into background threads/tasks
# ---------------------------------------------------------------------------

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("_request_id_var", default="")


def set_request_id(request_id: str) -> None:
    """Store *request_id* in the current context."""
    _request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the request ID bound to the current context (empty string if unset)."""
    return _request_id_var.get()


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

# Extra fields that receive first-class treatment in the JSON output
_KNOWN_EXTRAS = frozenset({"request_id", "migration_id", "duration_ms", "cost_usd", "stage"})

# Fields that are always present on a LogRecord — we exclude them from the
# generic "extras" sweep so we don't duplicate them.
_STDLIB_ATTRS = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON strings.

    Mandatory fields: ``timestamp``, ``level``, ``logger``, ``message``.
    Optional first-class extras: ``request_id``, ``migration_id``,
    ``duration_ms``, ``cost_usd``, ``stage``.
    Any other key=value pairs passed via ``extra=`` are forwarded verbatim.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Ensure record.message is populated (mimics Formatter.format behaviour)
        record.message = record.getMessage()

        payload: dict[str, object] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Correlation IDs — prefer what was injected via extra=, fall back to
        # the context var so background tasks automatically inherit the value.
        request_id: str = record.__dict__.get("request_id") or get_request_id()  # type: ignore[assignment]
        if request_id:
            payload["request_id"] = request_id

        migration_id = record.__dict__.get("migration_id")
        if migration_id is not None:
            payload["migration_id"] = migration_id

        # Other first-class extras
        for key in ("duration_ms", "cost_usd", "stage"):
            val = record.__dict__.get(key)
            if val is not None:
                payload[key] = val

        # Any remaining user-supplied extras that aren't stdlib attributes
        for key, val in record.__dict__.items():
            if key not in _STDLIB_ATTRS and key not in _KNOWN_EXTRAS and not key.startswith("_"):
                payload[key] = val

        # Attach exception traceback if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Root-logger configuration
# ---------------------------------------------------------------------------


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger to emit JSON-structured log lines.

    The log level is resolved in the following order:
    1. The *level* argument (if not None).
    2. The ``ROSETTASTONE_LOG_LEVEL`` environment variable.
    3. Default: ``"INFO"``.
    """
    resolved_level = level or os.environ.get("ROSETTASTONE_LOG_LEVEL", "INFO")

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Replace (or add) a single StreamHandler with the JSON formatter so we
    # don't accumulate duplicate handlers across multiple configure_logging()
    # calls (common in test environments).
    json_formatter = JsonFormatter()

    # Remove any existing handlers that already use our formatter to avoid
    # duplication when create_app() is called more than once in tests.
    for handler in list(root.handlers):
        if isinstance(handler.formatter, JsonFormatter):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(json_formatter)
    root.addHandler(handler)
