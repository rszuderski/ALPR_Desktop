# -*- coding: utf-8 -*-
"""Compact, read-only presentation for the mobile library (no approval or I/O).

All checks and transfer permissions remain in the original modules. Truncated
hashes are display labels only; the complete records are retained in JSON/copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PureWindowsPath
import re

from .z4_mobile_artifact_details import checkpoint_sha_by_role
from .z4_mobile_artifact_transfer_ui import transfer_guidance
from .z4_mobile_history_presentation import (
    SOURCE_CHECK, SOURCE_FILE, history_source_label, history_operation_label,
    history_outcome_label,
)

_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


def short_sha(value: object) -> str:
    text = str(value or "")
    return text[:12] + "…" + text[-8:] if _SHA.fullmatch(text) else "—"


def file_name(value: object) -> str:
    # Works with source Windows paths also during Linux QA; never resolves paths.
    return PureWindowsPath(str(value or "")).name or "—"


def display_utc(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "—"
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if instant.tzinfo is None:
        return text + " (strefa niepodana)"
    return instant.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M:%S")


def size_mib(value: object) -> str:
    try:
        number = int(value)
    except (ValueError, TypeError):
        return "—"
    if number < 0:
        return "—"
    return f"{number / (1024 * 1024):.2f}".replace(".", ",")


@dataclass(frozen=True)
class StateView:
    label: str
    tone: str
    next_step: str


def artifact_state(row, *, validated=False, failed=False, running=False, history=None) -> StateView:
    # Never grant a permission in this module. Existing guard decides first.
    verdict = transfer_guidance(
        row, desktop_validated=validated, validation_failed=failed,
        validation_running=running, history_entry=history,
    )
    label, tone = {
        "Gotowa do skopiowania": ("PC: OK · pakiet", "success"),
        "Model PC OK — niepełny ALPR": ("PC: OK · model", "success"),
        "Najpierw kontrola PC": ("Do kontroli PC", "warning"),
        "Trwa kontrola na PC": ("Kontrola w toku", "info"),
        "Nie jest paczką na telefon": ("Format nieobsługiwany", "warning"),
        "Nie kopiuj — problem z paczką": ("Problem z paczką", "danger"),
        "Nie kopiuj — kontrola niezaliczona": ("Kontrola niezaliczona", "danger"),
        "Nie kopiuj — plik zmieniony": ("SHA zmienione", "danger"),
        "Nie kopiuj — brak kontroli SHA": ("Brak odczytu SHA", "danger"),
    }.get(verdict.label, (verdict.label, "neutral"))
    next_step = verdict.next_step
    if tone == "success":
        if getattr(row, "kind", "") == "MODEL":
            next_step = "To składnik ALPR. Do pełnego potoku dobierz drugą rolę i utwórz parę MT+MZ."
        else:
            next_step = "Sprawdź zgodność z planem badania; następnie osobno zweryfikuj import i wykonanie na telefonie."
    elif verdict.label == "Najpierw kontrola PC":
        next_step = "Uruchom pełną kontrolę PC przyciskiem pod tabelą."
    return StateView(label, tone, next_step)


@dataclass(frozen=True)
class Sheet:
    title: str
    subtitle: str
    state: str
    tone: str
    fields: tuple[tuple[str, str], ...]
    next_step: str
    note: str


def _input_description(manifest) -> str:
    data = manifest.get("input") if isinstance(manifest, dict) else None
    if not isinstance(data, dict) or not data:
        return "Nie podano"
    sides = " × ".join(str(data.get(key) or "?") for key in ("width", "height", "channels"))
    properties = " / ".join(str(data[key]) for key in ("layout", "color_space", "data_type") if data.get(key))
    return sides + ("; " + properties if properties else "")


def artifact_sheet(row, *, validated=False, failed=False, running=False, history=None, failure="") -> Sheet:
    state = artifact_state(row, validated=validated, failed=failed, running=running, history=history)
    manifest = getattr(row, "manifest", {}) or {}
    scope = {"MODEL": "pojedynczy model", "PACKAGE": "pakiet ALPR",
             "RESEARCH_TFLITE": "plik badawczy"}.get(getattr(row, "kind", ""), "artefakt")
    role = str(getattr(row, "role", "") or "—")
    record = (history or {}).get("record") or {}
    fields = [("Plik", str(getattr(row, "name", "") or file_name(getattr(row, "path", "")))),
              ("ID modelu / pakietu", str(getattr(row, "model_id", "") or "—"))]
    if getattr(row, "kind", "") == "MODEL":
        fields.append(("Wejście (manifest)", _input_description(manifest)))
    for which, sha in checkpoint_sha_by_role(row):
        fields.append((f"SHA checkpointu {which} (skrót)", short_sha(sha)))
    if record:
        fields += [("Ostatnia kontrola PC (UTC)", display_utc(record.get("checked_utc"))),
                   ("SHA pliku z kontroli (skrót)", short_sha(record.get("sha256")))]
        compare = (history or {}).get("comparison")
        fields.append(("Porównanie SHA przy odczycie", {
            "MATCH": "Zgodne z zapisem kontroli", "CHANGED": "Plik zmieniony",
            "UNREADABLE": "Nie można odczytać", "UNCONFIRMED": "Niepotwierdzone",
        }.get(compare, "Niepotwierdzone")))
    if failure or getattr(row, "error", ""):
        fields.append(("Przyczyna", str(failure or row.error)))
    status = state.label
    if state.tone == "success":
        status = "Kontrola PC zaliczona" if validated else "Poprzednia kontrola PC zaliczona · SHA zgodne"
    return Sheet(
        f"{role} · {scope}",
        f"{getattr(row, 'variants', '—')}  ·  {size_mib(getattr(row, 'size_bytes', None))} MiB",
        status, state.tone, tuple(fields), state.next_step,
        "Kontrola PC nie potwierdza importu, wykonania ani jakości ALPR na Androidzie. Pełne identyfikatory: dane źródłowe / kopiowanie.",
    )


def event_tone(event: dict) -> str:
    code = str(event.get("outcome") or "")
    if code in {"FAILED", "PC_FAILED", "INCONSISTENT_HISTORY"}:
        return "danger"
    if code in {"STARTED", "STARTED_NO_RESULT", "UNCONFIRMED"}:
        return "warning"
    if code in {"SUCCEEDED", "PC_PASSED"}:
        return "success"
    return "neutral"


def event_sheet(event: dict) -> Sheet:
    origin = event.get("_history_origin")
    is_file = origin == SOURCE_FILE
    is_check = origin == SOURCE_CHECK
    fields = [("Plik", file_name(event.get("artifact_path")))]
    if is_file:
        fields += [("Data modyfikacji PLIKU (UTC)", display_utc(event.get("_file_modified_utc"))),
                   ("Rola (metadane)", str(event.get("_discovery_role") or "—")),
                   ("Stan odczytu", str(event.get("_discovery_state") or "—"))]
        state = "Brak powiązanego wpisu w odczytanym dzienniku"
        note = "Data pliku NIE jest datą konwersji. Odnalezienie paczki nie dowodzi eksportu, kontroli PC ani działania na telefonie."
        step = "Szczegóły bieżącego pliku sprawdź w zakładce Pakiety i modele."
    else:
        fields += [("Czas wpisu (UTC)", display_utc(event.get("observed_utc"))),
                   ("SHA pliku zapisane we wpisie (skrót)", short_sha(event.get("artifact_sha256")))]
        state = history_outcome_label(event)
        if is_check:
            record = event.get("_source_record") or {}
            fields.append(("Zakres", "Kontrola techniczna PC"))
            if record.get("detail"):
                fields.append(("Opis zapisany w kontroli", str(record["detail"])))
            note = "Historyczny PASS dotyczy chwili kontroli. Nie poświadcza obecnych bajtów ani wykonania i jakości na Androidzie."
            step = "To kontrola PC, nie konwersja. Przed ponownym użyciem pliku wymagana jest pełna kontrola."
        else:
            fields.append(("ID operacji", str(event.get("operation_id") or "Nie zapisano")))
            if event.get("detail"):
                fields.append(("Opis operacji", str(event["detail"])))
            note = "Zapis eksportu nie potwierdza obecnej integralności pliku, wykonania na Androidzie ani wyniku eksperymentu."
            step = "Pełny przebieg i zapisane zdarzenia są w danych źródłowych (JSON)."
    return Sheet(history_operation_label(event), history_source_label(event), state,
                 event_tone(event), tuple(fields), step, note)


# FIX03 — pure presentation helpers. No I/O, no inferred export events.
def inventory_caption(rows) -> str:
    """Library files and history records are different populations."""
    total = len(rows)
    packages = sum(getattr(row, "kind", "") in ("MODEL", "PACKAGE") for row in rows)
    other = total - packages
    return (f"Biblioteka: {total} plików · rozpoznane modele/pakiety: {packages} · "
            f"pozostałe: {other} (typ lub odczyt do sprawdzenia)")


def library_split_position(height, table_chrome, row_height, details_min, *, selected, preference=None) -> int:
    """Choose a bounded sash using measured chrome and table row metrics.

    The automatic position prioritizes rows and shows a compact empty inspector.
    With a selection, reserve useful scrollable detail space and prefer four rows
    whenever they fit. Explicit user fractions override this preference, not the
    bounds. Fractions are retained by the caller; resizing cannot overwrite them.
    """
    height = max(1, int(height))
    usable = max(1, height - 8)
    chrome = max(0, int(table_chrome))
    row = max(1, int(row_height))
    details = max(1, min(int(details_min), max(1, usable - chrome - row)))
    available = max(0, usable - chrome - details)
    min_rows = min(2, max(1, available // row))
    upper = max(1, usable - details)
    lower = min(upper, chrome + min_rows * row)
    if preference is None:
        detail_target = max(details, min(260, int(usable * .37))) if selected else details
        desired = usable - detail_target
        if selected:
            desired = max(desired, chrome + 4 * row)
    else:
        desired = int(usable * max(0.0, min(1.0, float(preference))))
    return int(max(lower, min(upper, desired)))
