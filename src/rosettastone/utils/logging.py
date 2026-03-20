"""Logging config — WARN default, never logs prompt content."""

from __future__ import annotations

import logging
import os

LOG_LEVEL = os.environ.get("ROSETTASTONE_LOG_LEVEL", "WARN").upper()


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured for rosettastone.

    Never log prompt content at any level to protect PII.
    """
    logger = logging.getLogger(f"rosettastone.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARN))
    return logger
