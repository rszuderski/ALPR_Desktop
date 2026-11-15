# -*- coding: utf-8 -*-
"""Read-only Z4 history presentation; observations never become export events.

Three distinct authorities:
* recorded conversion/export events from mobile_export_events.v1.jsonl,
* recorded PC validations from desktop_package_validation.v1.jsonl,
* currently discovered .alprmodel files without a matching export event.

This module never writes, hashes or validates a mobile package. File mtime is
ONLY an observed file timestamp, never the historical export timestamp.
"""
from __future__ import annotations

from pathlib import Path

from .z4_mobile_artifact_validation_history import history_key
from .z4_mobile_operation_summary import summarize_operations


SOURCE_EXPORT = "EXPORT_EVENT"
SOURCE_CHECK = "PC_VALIDATION"
SOURCE_FILE = "DISK_DISCOVERY"
VALIDATION_SCHEMA = "alpr.desktop_package_validation.v1"


def build_history_presentation(
    artifacts, export_events, pc_checks, *, raw_exports: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """Construct independent UI rows while preserving original source records.

    No event, operation ID or export timestamp is manufactured for a disk file.
    Historic PC validation outcomes remain observations, not current approvals.
    """
    records: list[dict] = []
    exports = export_events if raw_exports else summarize_operations(export_events)
    recorded_paths: set[str] = set()
    for raw in export_events:
        if isinstance(raw, dict) and raw.get("artifact_path"):
            recorded_paths.add(history_key(raw["artifact_path"]))

    for i, event in enumerate(exports):
        if not isinstance(event, dict):
            continue
        row = dict(event)
        row["_history_origin"] = SOURCE_EXPORT
        row["_history_source_index"] = i
        row["_history_sort_date"] = str(event.get("observed_utc") or "")
        records.append(row)

    pc_count = 0
    for index, check in enumerate(pc_checks):
        if not isinstance(check, dict) or check.get("schema") != VALIDATION_SCHEMA:
            continue
        if check.get("outcome") not in ("PASSED", "FAILED"):
            continue
        if check.get("kind") not in ("MODEL", "PACKAGE"):
            continue
        path = str(check.get("path") or "")
        if not path:
            continue
        pc_count += 1
        records.append({
            "_history_origin": SOURCE_CHECK,
            "_history_source_index": check.get("_ledger_index", index),
            "_history_sort_date": str(check.get("checked_utc") or ""),
            "observed_utc": str(check.get("checked_utc") or ""),
            "operation": "DESKTOP_PC_VALIDATION",
            "outcome": "PC_" + str(check["outcome"]),
            "operation_id": "",  # A validation is NOT an export operation.
            "artifact_path": path,
            "artifact_sha256": str(check.get("sha256") or ""),
            "_source_record": dict(check),
        })

    discovery_count = 0
    for artifact in artifacts:
        if getattr(artifact, "kind", "") not in ("MODEL", "PACKAGE"):
            continue
        path = str(getattr(artifact, "path", "") or "")
        if not path.lower().endswith(".alprmodel") or history_key(path) in recorded_paths:
            continue
        discovery_count += 1
        modified = str(getattr(artifact, "modified_utc", "") or "")
        records.append({
            "_history_origin": SOURCE_FILE,
            "_history_source_index": discovery_count,
            "_history_sort_date": modified,
            "_file_modified_utc": modified,
            "observed_utc": "",  # mtime is NOT an observed conversion date.
            "operation": "DISCOVERED_PACKAGE",
            "outcome": "FOUND_NO_EXPORT_EVENT",
            "operation_id": "",
            "artifact_path": path,
            "artifact_sha256": "",  # Discovery does NOT establish a hash.
            "_discovery_state": str(getattr(artifact, "state", "")),
            "_discovery_role": str(getattr(artifact, "role", "")),
        })
    records.sort(key=lambda row: (row.get("_history_sort_date") or "", row["_history_origin"]), reverse=True)
    return records, {
        "export_operations": len(summarize_operations(export_events)),
        "export_events": len(export_events),
        "pc_checks": pc_count,
        "unlogged_packages": discovery_count,
    }


def history_source_label(event: dict) -> str:
    return {
        SOURCE_EXPORT: "REJESTR EKSPORTU",
        SOURCE_CHECK: "KONTROLA PC",
        SOURCE_FILE: "PLIK Z DYSKU",
    }.get(str(event.get("_history_origin") or SOURCE_EXPORT), "NIEZNANE")


def history_operation_label(event: dict, *, raw_exports: bool = False) -> str:
    origin = event.get("_history_origin")
    if origin == SOURCE_CHECK:
        return "Sprawdzenie paczki (PC)"
    if origin == SOURCE_FILE:
        return "Odnaleziony pakiet"
    op = str(event.get("operation") or "")
    label = {
        "COMPLETE_PACKAGE_EXPORT": "Eksport zestawu ALPR",
        "SINGLE_MODEL_EXPORT": "Eksport modelu",
    }.get(op, op)
    if op == "MODEL_CONVERSION":
        role = str((event.get("conversion") or {}).get("role") or "?")
        label = f"Konwersja {role}"
    count = int(event.get("_event_count") or 0)
    if count > 1 and not raw_exports:
        label += f" ({count} zdarzeń)"
    return label


def history_outcome_label(event: dict) -> str:
    return {
        "STARTED_NO_RESULT": "BRAK WYNIKU",
        "INCONSISTENT_HISTORY": "SPRZECZNOŚĆ",
        "UNCONFIRMED": "NIEPOTWIERDZONE",
        "SUCCEEDED": "ZAKOŃCZONO",
        "FAILED": "BŁĄD",
        "STARTED": "ROZPOCZĘTO",
        "PC_PASSED": "PASS (wówczas)",
        "PC_FAILED": "BŁĄD (wówczas)",
        "FOUND_NO_EXPORT_EVENT": "Brak wpisu eksportu",
    }.get(str(event.get("outcome") or ""), str(event.get("outcome") or ""))


def history_detail_card(event: dict) -> str:
    """No green status is inferred from historic ledger entries or file mtime."""
    source = event.get("_history_origin")
    if source == SOURCE_CHECK:
        check = event.get("_source_record") or {}
        return "\n".join((
            "KONTROLA TECHNICZNA PC — ORYGINALNY WPIS HISTORYCZNY", "",
            f"Czas kontroli (UTC): {check.get('checked_utc') or 'nieznany'}",
            f"Wynik w chwili kontroli: {check.get('outcome') or '?'}",
            f"Rodzaj pliku: {check.get('kind') or '?'}",
            f"Plik: {check.get('path') or '-'}",
            f"SHA-256 zapisany podczas kontroli: {check.get('sha256') or '-'}",
            f"Zakres: {check.get('scope') or '?'}",
            f"Szczegóły: {check.get('detail') or '-'}", "",
            "To nie jest wpis konwersji ani eksportu.",
            "Historyczny PASS nie poświadcza obecnych bajtów pliku, gotowości Androida ani jakości ALPR.",
            "Przed użyciem paczki wykonaj jej nową pełną kontrolę.",
        ))
    if source == SOURCE_FILE:
        return "\n".join((
            "PAKIET ODNALEZIONY NA DYSKU — NIE JEST WPISANYM EKSPORTEM", "",
            f"Plik: {event.get('artifact_path') or '-'}",
            f"Data modyfikacji PLIKU (UTC): {event.get('_file_modified_utc') or '-'}",
            f"Rola według metadanych: {event.get('_discovery_role') or '-'}",
            f"Stan odczytu: {event.get('_discovery_state') or '-'}", "",
            "Brak powiązanego wpisu w odczytanym rejestrze operacji eksportu.",
            "Data pliku NIE jest datą konwersji. Nie przypisano plikowi zdarzenia, SHA ani zatwierdzenia.",
            "Odkrycie paczki NIE potwierdza działania na Androidzie lub jakości modelu.",
        ))
    return ""
