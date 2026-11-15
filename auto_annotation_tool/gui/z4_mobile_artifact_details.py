# -*- coding: utf-8 -*-
"""Z4LIB027: human-readable, strictly manifest-derived mobile model details.

No I/O, no model approval; a manifest can describe a model but cannot prove
its graph, its Android eligibility or the completion of F4/final_test.
"""
from __future__ import annotations
import re
from pathlib import Path
from .z4_mobile_artifact_status_ui import artifact_status_guidance
from .z4_mobile_artifact_validation_history import historical_detail_text, historical_status_label
from .z4_mobile_artifact_transfer_ui import transfer_guidance

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_ROLE = {"vehicle": "MP", "plate": "MT", "character": "MZ"}


def _dict(value):
    return value if isinstance(value, dict) else {}


def _value(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "TAK" if value else "NIE"
    return str(value)


def _sha(value):
    token = str(value or "").strip().lower()
    return token if _SHA256.fullmatch(token) else ""


def checkpoint_sha_by_role(row) -> list[tuple[str, str]]:
    """Return *only* declared and syntactically valid SHA, never guess from names."""
    result = []
    root = _dict(getattr(row, "manifest", {}))
    source = _dict(root.get("source"))
    root_sha = _sha(source.get("checkpoint_sha256"))
    if root_sha:
        result.append((str(getattr(row, "role", "model")), root_sha))
    children = _dict(getattr(row, "child_metadata", {}))
    for role in ("vehicle", "plate", "character"):
        child = _dict(children.get(role))
        child_sha = _sha(child.get("checkpoint_sha256"))
        if child_sha:
            result.append((_ROLE[role], child_sha))
    return result


def artifact_search_text(row) -> str:
    """Include checkpoint SHAs in the existing read-only library search."""
    base = " ".join(str(getattr(row, key, "") or "") for key in
                    ("role", "name", "model_id", "variants", "state", "path"))
    return base + " " + " ".join(sha for _, sha in checkpoint_sha_by_role(row))


def _input_lines(value, *, prefix=""):
    obj = _dict(value)
    if not obj:
        return [prefix + "Wejście: brak danych w manifeście"]
    w, h = obj.get("width"), obj.get("height")
    side = f"{_value(w)} × {_value(h)}"
    ch = obj.get("channels")
    if ch is not None:
        side += f" × {ch} kanały"
    rest = " / ".join(_value(obj.get(key)) for key in ("layout", "color_space", "data_type"))
    return [prefix + f"Wejście: {side}; {rest}"]


def format_artifact_card(row, *, desktop_validated_this_selection=False, validation_failure="", history_entry=None) -> str:
    """A short, scannable card. Does not promote any technical status."""
    manifest = _dict(getattr(row, "manifest", {}))
    kind = str(getattr(row, "kind", ""))
    state = str(getattr(row, "state", ""))
    source = _dict(manifest.get("source"))
    guidance = artifact_status_guidance(
        row, desktop_validated=desktop_validated_this_selection,
        validation_failed=bool(validation_failure),
    )
    display_label = guidance.label
    explanation = guidance.meaning
    if not desktop_validated_this_selection and not validation_failure:
        display_label = historical_status_label(row, history_entry, fallback=guidance.label)
        if display_label == "Poprzednia kontrola PC: OK":
            explanation = ("Poprzednia kontrola PC była zaliczona dla tej samej wersji pliku (SHA-256). "
                           "To nie jest nowy test ani zgoda na użycie w Androidzie.")
        elif display_label == "Zmieniony od kontroli":
            explanation = "Plik zmienił się od poprzedniej kontroli. Dawny wynik nie dotyczy obecnej wersji."
        elif display_label == "Poprzednia kontrola PC: błąd":
            explanation = "Poprzednia kontrola była niezaliczona. Plik wymaga wyjaśnienia błędu."
    phone = transfer_guidance(
        row, desktop_validated=desktop_validated_this_selection,
        validation_failed=bool(validation_failure), history_entry=history_entry,
    )
    lines = [
        "PRZENIESIENIE NA TELEFON",
        "Gotowość: " + phone.label,
        "Co to oznacza: " + phone.explanation,
        "Co zrobić: " + phone.next_step,
        "Uwaga: skopiowanie pliku nie dowodzi jego uruchomienia w Androidzie.",
        "",
        "CO WIEMY I CO DALEJ",
        "Stan: " + display_label,
        "Co to oznacza: " + explanation,
        "Następny krok: " + guidance.next_step,
        "",
        "SZCZEGÓŁY TECHNICZNE — opis paczki, nie test na telefonie",
        "",
        "PLIK I STATUS",
        f"  Nazwa: {_value(getattr(row, 'name', None))}",
        f"  Rola: {_value(getattr(row, 'role', None))}     Typ: {_value(kind)}",
        f"  Format/precyzja: {_value(getattr(row, 'variants', None))}",
        f"  Kod stanu (diagnostyka): {_value(state)}",
        f"  Kontrola Desktop: {'PASS dla ostatnio sprawdzonego zaznaczenia' if desktop_validated_this_selection else 'NIEPOTWIERDZONA w tym zaznaczeniu'}",
        f"  Rozmiar paczki: {int(getattr(row, 'size_bytes', 0)) / (1024*1024):.2f} MiB",
        f"  Data (UTC): {_value(getattr(row, 'modified_utc', None))}",
    ]
    if history_entry is not None:
        note = historical_detail_text(history_entry)
        if note:
            lines[11:11] = ["", note, ""]
    error = str(getattr(row, "error", "") or "").strip()
    if error:
        lines.append("  BŁĄD: " + error)
    if validation_failure:
        lines.append("  OSTATNIA KONTROLA — PRZYCZYNA: " + str(validation_failure)[:800])
    lines += [
        "",
        "IDENTYFIKACJA I PROWENIENCJA",
        f"  ID modelu: {_value(getattr(row, 'model_id', None))}",
        f"  Schemat manifestu: {_value(getattr(row, 'schema', None))}",
    ]
    checks = checkpoint_sha_by_role(row)
    if checks:
        for role, sha in checks:
            lines.append(f"  Checkpoint SHA-256 ({role}): {sha}")
    else:
        lines.append("  Checkpoint SHA-256: BRAK w czytanych polach manifestu")
    if kind == "MODEL":
        lines += _input_lines(manifest.get("input"), prefix="  ")
        lines.append("  Typ zadania: " + _value(manifest.get("task")))
        for key, title in (("training_run_id", "ID treningu"),
                           ("dataset_id", "Dataset"),
                           ("dataset_manifest_sha256", "Manifest danych SHA-256"),
                           ("dataset_split_sha256", "Podział SHA-256")):
            if source.get(key):
                lines.append("  " + title + ": " + _value(source[key]))
        output = _dict(manifest.get("output"))
        if output:
            lines.append("  Wyjście/decoder: " + " / ".join(_value(output.get(k))
                          for k in ("output_format", "decoder")))
            if "class_count" in output:
                lines.append("  Liczba klas: " + _value(output.get("class_count")))
    elif kind == "PACKAGE":
        children = _dict(getattr(row, "child_metadata", {}))
        if children:
            lines.append("  Podmodele (metadane z manifestów potomnych):")
            for role in ("vehicle", "plate", "character"):
                child = _dict(children.get(role))
                if child:
                    lines.append(f"    {_ROLE[role]}: {_value(child.get('model_id'))}; "
                                 f"{_value(child.get('variants'))}")
                    lines += _input_lines(child.get("input"), prefix="      ")
        else:
            lines.append("  Brak odczytanych metadanych podmodeli; sprawdź zakładkę JSON.")
    lines += [
        "",
        "OGRANICZENIA: odkrycie/odczyt manifestu ≠ pełna walidacja grafu.",
        "Kontrola na komputerze nie potwierdza działania modelu na telefonie z Androidem ani jakości rozpoznawania.",
        "Do szczegółowego audytu użyj zakładki «Manifest JSON» i przycisku pełnej kontroli.",
    ]
    return "\n".join(lines)


def format_event_card(event: dict) -> str:
    e = _dict(event)
    lines = [
        "OPERACJA — podsumowanie zapisanej historii, bez nowej weryfikacji pliku", "",
        "  Rodzaj: " + _value(e.get("operation")),
        "  Wynik: " + _value(e.get("outcome")),
        "  Czas (UTC): " + _value(e.get("observed_utc")),
        "  Identyfikator operacji: " + _value(e.get("operation_id")),
        "  Plik: " + _value(e.get("artifact_path")),
        "  SHA-256 artefaktu (z logu): " + _value(e.get("artifact_sha256")),
        "", "Historyczny wpis nie potwierdza aktualnej integralności pliku.",
        "Pełny zapis operacji znajduje się w zakładce «Manifest JSON».",
    ]
    return "\n".join(lines)
