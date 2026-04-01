"""RosettaStone — automated LLM model migration tool."""

from importlib.metadata import PackageNotFoundError, version

from rosettastone.config import MigrationConfig
from rosettastone.core.migrator import Migrator
from rosettastone.core.types import MigrationResult

try:
    __version__ = version("rosettastone-llm")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["__version__", "Migrator", "MigrationConfig", "MigrationResult"]
