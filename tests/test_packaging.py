"""Tests for package metadata and distribution."""

from __future__ import annotations


def test_version_importable() -> None:
    """__version__ is importable from the package."""
    from rosettastone import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
    # Should be a PEP 440 version string
    assert "." in __version__ or __version__ == "0.0.0+dev"


def test_cli_entry_point_importable() -> None:
    """The CLI entry point function is importable."""
    from rosettastone.cli.main import app

    assert app is not None


def test_server_importable() -> None:
    """The FastAPI app factory is importable."""
    from rosettastone.server.app import create_app

    assert callable(create_app)


def test_migrator_importable() -> None:
    """Core Migrator is importable at the top level."""
    from rosettastone.core.migrator import Migrator

    assert Migrator is not None


def test_config_importable() -> None:
    """MigrationConfig is importable."""
    from rosettastone.config import MigrationConfig

    assert MigrationConfig is not None
