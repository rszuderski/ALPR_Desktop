"""Niskopoziomowy dostęp do lokalnego rejestru SQLite."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .migrations import migrate_database


class RegistryDatabase:
    """Zarządza plikiem SQLite rejestru Workspace."""

    DEFAULT_TIMEOUT_SECONDS = 30.0
    DEFAULT_BUSY_TIMEOUT_MS = 5000

    def __init__(
        self,
        db_path: str | Path,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.timeout = float(timeout)

    def _ensure_parent(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        self._ensure_parent()
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f"PRAGMA busy_timeout = {int(self.DEFAULT_BUSY_TIMEOUT_MS)}"
        )
        journal_mode_row = connection.execute(
            "PRAGMA journal_mode = WAL"
        ).fetchone()
        journal_mode = str(journal_mode_row[0] if journal_mode_row else "").lower()
        if journal_mode != "wal":
            connection.close()
            raise RuntimeError(
                f"Nie udało się włączyć SQLite WAL dla {self.db_path}. "
                f"journal_mode={journal_mode or 'unknown'}"
            )
        return connection

    def initialize(self) -> int:
        connection = self.connect()
        try:
            return migrate_database(connection)
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    def integrity_check(self) -> str:
        self.initialize()
        with self.read_connection() as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0] if row else "")

    def backup_to(self, destination: str | Path) -> Path:
        self.initialize()
        destination_path = Path(destination)

        if destination_path.resolve() == self.db_path.resolve():
            raise ValueError("Kopia zapasowa nie może wskazywać pliku źródłowego.")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source = self.connect()
        target = sqlite3.connect(str(destination_path), timeout=self.timeout)
        try:
            source.backup(target)
            target.commit()
        finally:
            target.close()
            source.close()
        return destination_path
