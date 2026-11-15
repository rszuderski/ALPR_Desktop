"""Read-only, SHA-pinned restoration of original ALPR session data.

A saved browser report is a compact *index*, not the research archive. This
module reopens the archive only when its digest and report identity match the
index. It never updates research data or rewrites a session file.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any, Callable

_SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")


@dataclass(frozen=True)
class SourceLinkResult:
    status: str
    detail: str
    bundle: Any = None
    path: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "verified" and self.bundle is not None


def stored_source(report: Any) -> str:
    """Prefer a persisted source path; never search for similar filenames."""
    value = str(getattr(report, "source_path", "") or "").strip()
    if value:
        return value
    raw = getattr(report, "raw", None)
    ingest = raw.get("desktop_ingest") if isinstance(raw, dict) else None
    return str(ingest.get("source_path") or "").strip() if isinstance(ingest, dict) else ""


def _identity_session(source: Any) -> str:
    index = getattr(source, "experiment_index", None)
    if isinstance(index, dict):
        return str(index.get("experiment_session_id") or index.get("session_id") or "").strip()
    return ""


def restore_source(report: Any, selected_path: str | Path, *, reader: Callable | None = None) -> SourceLinkResult:
    """Validate SHA, report identity, session identity and manifest before trust.

    Caller runs this off the Tk thread. A wrong source is never rendered as data
    from another report, even if it has a similar name, variant or timestamp.
    """
    required_hash = str(getattr(report, "source_archive_sha256", "") or "").strip().lower()
    if not _SHA256.fullmatch(required_hash):
        return SourceLinkResult("no_fingerprint", "Zapisany raport nie ma wiarygodnego SHA-256 oryginalnego archiwum.")
    filename = str(selected_path or "").strip()
    if not filename:
        return SourceLinkResult("missing", "Brak ścieżki do archiwum. Wybierz oryginalny plik .alprsession.")
    path = Path(filename)
    try:
        if not path.is_file():
            return SourceLinkResult("missing", "Nie znaleziono oryginalnego archiwum. Możesz wskazać jego nową lokalizację.", path=str(path))
    except OSError:
        return SourceLinkResult("missing", "Nie można odczytać wskazanego pliku źródłowego.", path=str(path))
    if reader is None:
        from ..ranking import read_mobile_report_bundles
        reader = read_mobile_report_bundles
    try:
        # Standard parser hashes the WHOLE file, checks archive entry SHA-256 and
        # validates the session format. No in-place extraction is performed.
        bundles = reader(path)
    except Exception as exc:
        return SourceLinkResult("invalid", "Odrzucono archiwum: " + str(exc)[:220], path=str(path))
    if not bundles:
        return SourceLinkResult("invalid", "Plik nie zawiera prawidłowego raportu ALPR.", path=str(path))
    mismatches = []
    for bundle in bundles:
        candidate_hash = str(getattr(bundle, "source_archive_sha256", "") or "").strip().lower()
        if candidate_hash != required_hash:
            mismatches.append("SHA-256 archiwum nie zgadza się z zapisanym raportem")
            continue
        validation = getattr(bundle, "validation", None)
        if validation is None or not bool(getattr(validation, "ok", False)):
            mismatches.append("archiwum nie przeszło kontroli integralności")
            continue
        candidate = getattr(bundle, "report", None)
        if candidate is None or any(str(getattr(candidate, key, "") or "") != str(getattr(report, key, "") or "")
                                    for key in ("report_id", "package_id", "variant_id", "measured_at")):
            mismatches.append("identyfikator lub wariant raportu jest niezgodny")
            continue
        expected_session = _identity_session(report)
        actual_session = str((getattr(bundle, "collection_session", None) or {}).get("session_id") or "").strip()
        if not actual_session:
            actual_session = str((getattr(bundle, "experiment_session", None) or {}).get("experiment_session_id") or "").strip()
        if expected_session and actual_session != expected_session:
            mismatches.append("identyfikator sesji badawczej jest niezgodny")
            continue
        return SourceLinkResult("verified", "Oryginalne archiwum zweryfikowane (SHA-256 i tożsamość sesji).", bundle, str(path))
    return SourceLinkResult("mismatch", "Nie połączono źródła: " + "; ".join(dict.fromkeys(mismatches))[:240], path=str(path))


def relinked_report(report: Any, path: str) -> Any:
    """Update only the saved source locator after a verified manual relink.

    Research measurements, source SHA, GT, review statuses and device/provenance
    metadata are left intact; no source archive is modified.
    """
    raw = dict(getattr(report, "raw", {}) or {})
    ingest = dict(raw.get("desktop_ingest") or {})
    ingest["source_path"] = str(path)
    raw["desktop_ingest"] = ingest
    if isinstance(raw.get("desktop_experiment_index"), dict):
        raw_index = dict(raw["desktop_experiment_index"])
        if "source_path" in raw_index:
            raw_index["source_path"] = str(path)
        raw["desktop_experiment_index"] = raw_index
    index = dict(getattr(report, "experiment_index", {}) or {})
    if "source_path" in index:
        index["source_path"] = str(path)
    return replace(report, source_path=str(path), raw=raw, experiment_index=index)


def crop_entry_for_row(row: dict[str, Any], entry_names: set[str]) -> str:
    """Resolve a sample to an EXISTING ZIP entry, without manufacturing a crop."""
    capture_id = str(row.get("capture_id") or row.get("id") or "").strip()
    explicit = str(row.get("crop") or row.get("path") or row.get("image") or "").strip().replace("\\", "/")
    candidates = [explicit] if explicit else []
    if capture_id and "/" not in capture_id and "\\" not in capture_id and capture_id not in {".", ".."}:
        candidates.extend(f"samples/crops/{capture_id}.{ext}" for ext in ("jpg", "jpeg", "png", "webp"))
    return next((s for s in candidates if s in entry_names and s.startswith("samples/crops/")), "")
