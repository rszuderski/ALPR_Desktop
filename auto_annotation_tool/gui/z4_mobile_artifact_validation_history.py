# -*- coding: utf-8 -*-
"""Historical, SHA-bound Desktop model checks; NOT reuse authorization.

This is a separate append-only audit trail. It never mutates mobile models,
training records or the existing mobile export-event ledger. Historical PASS
must never substitute the authoritative fresh validator before reuse.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from .z4_mobile_artifact_registry import sha256_file

SCHEMA = "alpr.desktop_package_validation.v1"
SCOPE = "DESKTOP_EXPORTER_FULL_PRECISION_CHECK_ONLY"
FILENAME = "desktop_package_validation.v1.jsonl"
MAX_LEDGER_BYTES = 6 * 1024 * 1024
MAX_RECORD_BYTES = 4096
MAX_RECORDS = 15000
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LOCK = threading.Lock()
_BLOCKING_STATES = frozenset({
    "MANIFEST_ERROR", "READ_ERROR", "REGISTERED_SHA_MISMATCH",
    "REGISTERED_SHA_UNREADABLE", "RESEARCH_SHA_MISMATCH",
    "RESEARCH_ONLY_NOT_PACKAGED",
})


def history_key(path) -> str:
    """Use the same key for recorded paths and freshly discovered files."""
    try:
        return str(Path(path).resolve(strict=False)).casefold()
    except (OSError, ValueError, RuntimeError):
        return str(path).casefold()


def _redirected(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)
        return bool(junction()) if callable(junction) else False
    except (OSError, ValueError):
        return True


def _ledger(root: Path) -> Path:
    return Path(root) / "_registry" / FILENAME


def _safe_ledger(root: Path, *, create: bool) -> Path:
    root = Path(root)
    folder = root / "_registry"
    target = folder / FILENAME
    if _redirected(root) or _redirected(folder) or _redirected(target):
        raise ValueError("Historia kontroli: zapis/odczyt przez dowiązanie zablokowany")
    if create:
        folder.mkdir(parents=True, exist_ok=True)
        if _redirected(folder) or _redirected(target):
            raise ValueError("Historia kontroli: katalog przekierowuje zapis")
    return target


def record_desktop_check(root: Path, *, path: str, kind: str, passed: bool,
                         sha256: str = "", detail: str = "") -> dict:
    """Persist an observed test, not any new approval status."""
    target = Path(path)
    if target.suffix.lower() != ".alprmodel" or kind not in ("MODEL", "PACKAGE"):
        raise ValueError("Nieobsługiwany plik/rodzaj wyniku kontroli")
    if _redirected(target):
        raise ValueError("Wynik kontroli nie może dotyczyć dowiązania")
    digest = str(sha256 or "").lower()
    if digest and not _SHA.fullmatch(digest):
        raise ValueError("Niewłaściwe SHA-256 wyniku kontroli")
    if passed and not digest:
        raise ValueError("Pozytywna kontrola wymaga SHA-256 pliku")
    event = {
        "schema": SCHEMA,
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": str(target.resolve(strict=False)),
        "kind": kind,
        "outcome": "PASSED" if passed else "FAILED",
        "sha256": digest,
        "scope": SCOPE,
        "detail": str(detail or "")[:550],
        "android_verified": False,
        "experiment_verified": False,
    }
    line = (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if len(line) > MAX_RECORD_BYTES:
        raise ValueError("Zbyt długi wpis historii kontroli")
    with _LOCK:
        journal = _safe_ledger(root, create=True)
        if journal.exists() and journal.stat().st_size + len(line) > MAX_LEDGER_BYTES:
            raise ValueError("Rejestr kontroli osiągnął limit; wymagany osobny audyt")
        if _redirected(journal):
            raise ValueError("Rejestr kontroli jest dowiązaniem")
        with journal.open("ab") as fp:
            fp.write(line)
            fp.flush()
            os.fsync(fp.fileno())
    return event


def _valid_record(data) -> bool:
    return (isinstance(data, dict)
            and data.get("schema") == SCHEMA
            and data.get("scope") == SCOPE
            and data.get("kind") in ("MODEL", "PACKAGE")
            and data.get("outcome") in ("PASSED", "FAILED")
            and data.get("android_verified") is False
            and data.get("experiment_verified") is False
            and isinstance(data.get("path"), str) and len(data["path"]) < 1800
            and isinstance(data.get("checked_utc"), str)
            and isinstance(data.get("sha256"), str)
            and (bool(_SHA.fullmatch(data["sha256"])) or
                 (data["outcome"] == "FAILED" and data["sha256"] == "")))



def read_check_events(root: Path) -> tuple[list[dict], str]:
    """Read the full *recorded* desktop-check audit, without any writes.

    A bad/truncated line is not an event. A truncated or redirected complete
    ledger is never presented as a complete history. This is NOT approval.
    """
    try:
        journal = _safe_ledger(root, create=False)
        if not journal.exists():
            return [], "Brak dziennika kontroli PC"
        if not journal.is_file() or journal.stat().st_size > MAX_LEDGER_BYTES:
            return [], "Dziennik kontroli PC jest niedostępny lub przekracza limit"
        records = []
        bad = 0
        with journal.open("rb") as fp:
            for index, raw in enumerate(fp):
                if index >= MAX_RECORDS:
                    return [], "Dziennik kontroli PC przekracza limit rekordów — odczyt zablokowany"
                if len(raw) > MAX_RECORD_BYTES or not raw.endswith(b"\n"):
                    bad += 1
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError):
                    bad += 1
                    continue
                if not _valid_record(record):
                    bad += 1
                    continue
                entry = dict(record)
                entry["_ledger_index"] = index
                records.append(entry)
        note = f"Pominięto {bad} niepoprawnych wpisów kontroli PC" if bad else ""
        return records, note
    except (OSError, ValueError, RuntimeError):
        return [], "Nie można bezpiecznie odczytać dziennika kontroli PC"

def read_last_checks(root: Path) -> dict[str, dict]:
    """Read bounded append order; ignore corrupt lines, never approve a model."""
    try:
        journal = _safe_ledger(root, create=False)
        if not journal.is_file() or journal.stat().st_size > MAX_LEDGER_BYTES:
            return {}
        latest: dict[str, dict] = {}
        with journal.open("rb") as fp:
            for i, raw in enumerate(fp):
                if i >= MAX_RECORDS:
                    return {}  # never call an incomplete audit authoritative
                if len(raw) > MAX_RECORD_BYTES or not raw.endswith(b"\n"):
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeError, ValueError):
                    continue
                if _valid_record(record):
                    latest[history_key(record["path"])] = record
        return latest
    except (OSError, ValueError, RuntimeError):
        return {}


def observe_last_checks(root: Path, rows, *, hash_fn=sha256_file) -> dict[str, dict]:
    """Hash the PRESENT bytes on background scan, not mtime/size as proof."""
    latest = read_last_checks(root)
    observations = {}
    for row in rows:
        if getattr(row, "kind", None) not in ("MODEL", "PACKAGE"):
            continue
        key = history_key(row.path)
        record = latest.get(key)
        if not record or record.get("kind") != row.kind:
            continue
        comparison = "UNCONFIRMED"
        digest = str(record.get("sha256") or "")
        if digest:
            try:
                path = Path(row.path)
                if _redirected(path) or not path.is_file():
                    comparison = "UNREADABLE"
                else:
                    before = path.stat()
                    actual = hash_fn(path)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        comparison = "CHANGED"
                    else:
                        comparison = "MATCH" if actual.lower() == digest else "CHANGED"
            except (OSError, ValueError, RuntimeError):
                comparison = "UNREADABLE"
        observations[key] = {"record": record, "comparison": comparison}
    return observations


def historical_status_label(row, observation: dict | None, *, fallback: str) -> str:
    """Pure UI mapping. Never hide provenance blockers or grant Android approval."""
    if not observation or str(getattr(row, "state", "")) in _BLOCKING_STATES:
        return fallback
    comparison = observation.get("comparison")
    outcome = (observation.get("record") or {}).get("outcome")
    if comparison == "CHANGED":
        return "Zmieniony od kontroli"
    if comparison == "UNREADABLE":
        return "Nie można sprawdzić zmian"
    if comparison == "MATCH" and outcome == "PASSED":
        return "Poprzednia kontrola PC: OK"
    if outcome == "FAILED":
        return "Poprzednia kontrola PC: błąd"
    return fallback


def historical_detail_text(observation: dict | None) -> str:
    if not observation:
        return ""
    record = observation.get("record") or {}
    date = str(record.get("checked_utc") or "nieznana")
    digest = str(record.get("sha256") or "brak")
    comparison = observation.get("comparison")
    outcome = record.get("outcome")
    if comparison == "CHANGED":
        message = "Plik zmienił się od zapisanej kontroli. Poprzedni wynik nie dotyczy obecnej wersji."
    elif comparison == "UNREADABLE":
        message = "Nie można obecnie porównać pliku z zapisanym SHA-256. Wynik nie jest aktualnym potwierdzeniem."
    elif comparison == "MATCH" and outcome == "PASSED":
        message = "Poprzednia kontrola techniczna na komputerze była zaliczona. Obecny plik ma to samo SHA-256."
    elif outcome == "FAILED":
        message = "Poprzednia kontrola techniczna była niezaliczona. Nie traktuj paczki jako sprawdzonej."
    else:
        message = "Nie można potwierdzić zgodności obecnego pliku z poprzednią kontrolą."
    return ("POPRZEDNIA KONTROLA PC (zapisana historia)\n"
            f"Data (UTC): {date}\nWynik: {str(outcome or 'nieznany')}\n"
            f"SHA-256 wtedy: {digest}\n{message}\n"
            "To tylko zapis wcześniejszej kontroli, NIE zatwierdzenie działania na Androidzie. "
            "Przed ponownym użyciem uruchom pełną kontrolę paczki.")
