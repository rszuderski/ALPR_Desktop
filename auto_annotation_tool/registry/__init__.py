"""Minimalny publiczny interfejs lokalnego rejestru SQLite."""

from .database import RegistryDatabase
from .schema import SCHEMA_VERSION

__all__ = [
    "RegistryDatabase",
    "SCHEMA_VERSION",
]
