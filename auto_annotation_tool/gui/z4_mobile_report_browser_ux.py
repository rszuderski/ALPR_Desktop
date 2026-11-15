"""Presentation-only helpers for the Z4 ALPR mobile-report browser.

No edits to source .alprsession archives, predictions, GT, statistics or sidecars.
Sidecar reads are bounded, validated and used exclusively for a status preview.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
import math
import unicodedata
from pathlib import Path
from typing import Any

from .mobile_report_operator_metadata import phone_session_metadata

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")

SORT_COLUMNS = ('Nazwa','Data','Wariant','Scena','Powt.','Weryfikacja','p95')


def report_measurement_time(report):
    """Use source measurement metadata, never the ingest clock or a filename."""
    index=_get(report,'experiment_index') or {}
    raw=_get(report,'raw') or {}
    value=index.get('measured_at') or raw.get('measured_at')
    if not value:return None
    try:return datetime.fromisoformat(str(value).replace('Z','+00:00'))
    except (ValueError,TypeError,OverflowError):return None


def report_repeat(report):
    parts=str(_get(report,'source_path') or '').replace('\\','/').split('/')
    folders=[p.lower() for p in parts if re.fullmatch(r'r\d+',p,re.IGNORECASE)]
    if any(p.lower()=='02_sesje' for p in parts) and folders:
        return folders[-1],int(folders[-1][1:])
    value=(_get(report,'experiment_index') or {}).get('replicate_index')
    if not isinstance(value,bool) and re.fullmatch(r'\d+',str(value)):
        return f'r{int(value):02d}',int(value)
    return '—',None


def report_view_key(report):
    """Stable source identity, independent of its Treeview position."""
    return tuple(str(_get(report,k) or '') for k in
        ('report_id','source_archive_sha256','source_path','package_id','variant_id','measured_at'))


def report_sort_value(report,column,identity,verification,p95,phone_metadata=None):
    if column=='Nazwa':
        metadata=phone_metadata or phone_session_metadata(report)
        value=metadata.name.value or metadata.note.value
        return unicodedata.normalize('NFKC',value).casefold() if value else None
    if column=='Data':
        stamp=report_measurement_time(report)
        # Without an offset the absolute instant is unknown; keep it after
        # comparable timestamps and retain the literal date in the table.
        try:return stamp.timestamp() if stamp and stamp.tzinfo is not None else None
        except (ValueError,OverflowError,OSError):return None
    if column=='Powt.':return report_repeat(report)[1]
    if column=='p95':
        try:value=float(p95)
        except (ValueError,TypeError,OverflowError):return None
        return value if math.isfinite(value) else None
    value=(identity.get('variant') if column=='Wariant' else identity.get('scenario') if column=='Scena'
           else getattr(verification,'label',None) if column=='Weryfikacja' else None)
    if value in (None,'','—','Wariant nieokreślony'):return None
    return unicodedata.normalize('NFKC',str(value)).casefold()


def sorted_report_view(reports,value,descending):
    tied=sorted(reports,key=report_view_key)
    known=[];missing=[]
    for report in tied:
        key=value(report)
        (missing if key is None else known).append(report if key is None else (key,report))
    known.sort(key=lambda item:item[0],reverse=descending)
    return [report for _,report in known]+missing


def report_sort_preference(settings):
    if settings is None:return 'Data',True
    column=settings.get('mobile_report_browser','sort_column','Data')
    descending=settings.get('mobile_report_browser','sort_descending',True)
    return (column,descending) if column in SORT_COLUMNS and isinstance(descending,bool) else ('Data',True)


def save_report_sort_preference(settings,column,descending):
    if settings is None:return
    settings.set('mobile_report_browser','sort_column',column)
    settings.set('mobile_report_browser','sort_descending',bool(descending))
    settings.save_session()


def _text(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def report_identity(report: Any, bundle: Any = None) -> dict[str, str]:
    """Human-readable selection labels. Never replace canonical raw identifiers."""
    index = _get(bundle, "experiment_session") if bundle is not None else None
    if not isinstance(index, dict) or not index:
        index = _get(report, "experiment_index") or {}
    if not isinstance(index, dict) or not index:
        raw = _get(report, "raw") or {}
        index = raw.get("desktop_experiment_index", {}) if isinstance(raw, dict) else {}
    if not isinstance(index, dict):
        index = {}
    experiment = _text(index.get("experiment_variant") or index.get("variant") or index.get("experiment_variant_id"))
    technical = _text(_get(report, "variant_id"))
    variant = experiment or technical or "Wariant nieokreślony"
    session_id = _text(index.get("experiment_session_id") or index.get("session_id"))
    if not session_id:
        collection = _get(bundle, "collection_session") or {}
        if isinstance(collection, dict):
            session_id = _text(collection.get("session_id"))
    device = _get(report, "device") or {}
    device_name = " ".join(x for x in (_text(_get(device, "manufacturer")), _text(_get(device, "model") or _get(device, "name"))) if x)
    package = _text(_get(report, "package_id")) or "Pakiet nieokreślony"
    measured = _text(_get(report, "measured_at"))
    compact_date = measured[:16].replace("T", " ")
    try:
        candidate = measured.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(candidate)
        compact_date = parsed.strftime("%d.%m %H:%M")
    except (ValueError, TypeError):
        pass
    report_id = _text(_get(report, "report_id"))
    scenario = _text(index.get("scenario_id"))
    series = _text(index.get("series_id"))
    return {
        "variant": variant, "variant_short": variant if len(variant) <= 21 else variant[:18] + "…",
        "technical_variant": technical, "session_id": session_id, "report_id": report_id,
        "date": compact_date or "—", "package": package,
        "device": device_name or "Telefon nieokreślony", "scenario": scenario, "series": series,
    }


def report_filter_blob(report: Any, bundle: Any = None) -> str:
    info = report_identity(report, bundle)
    phone = phone_session_metadata(report, bundle)
    return " ".join([*info.values(), phone.name.value, phone.note.value,
                     _text(_get(report, "measured_at")), _text(_get(report, "source_archive_sha256"))]).casefold()


def matches_report_filter(report: Any, search: str, bundle: Any = None) -> bool:
    """All whitespace-separated tokens must appear somewhere in report metadata."""
    words = [word.casefold() for word in (search or "").split()]
    if not words:
        return True
    blob = report_filter_blob(report, bundle)
    info = report_identity(report, bundle)
    for word in words:
        # A variant identifier must not accidentally match the series ID
        # (e.g. F4-TV-FP32-A matches F4-TV-FP32-A12-... of variant B).
        if re.fullmatch(r"f4-tv-fp32-[a-d]", word, flags=re.IGNORECASE):
            if info["variant"].casefold() != word:
                return False
        elif word not in blob:
            return False
    return True


def sidecar_preview(root: Path | str, archive_sha256: str, session_id: str = "") -> dict[str, Any]:
    """Read bounded Desktop review metadata. Invalid/mismatched sidecars are NOT trusted.

    The result is display-only: total reviewed subjects, CER, NMCE and IoU require
    the canonical review backend, not these advisory sidecar counts.
    """
    empty = {"state": "none", "title": "Brak weryfikacji Desktop", "detail": "Nie znaleziono ocen operatora dla tego archiwum.", "revision": None, "gt_entries": None}
    source = _text(archive_sha256).lower()
    if not _SHA256.fullmatch(source):
        return {**empty, "state": "unverified", "title": "Weryfikacja Desktop: nieustalona", "detail": "Brak wiarygodnego SHA-256 archiwum."}
    file = Path(root) / "human_reviews" / f"{source}.review.json"
    try:
        if not file.is_file():
            return empty
        if file.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Plik weryfikacji jest zbyt duży dla podglądu.")
        payload = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Nieprawidłowa struktura weryfikacji.")
        if _text(payload.get("source_archive_sha256")).lower() != source:
            raise ValueError("SHA-256 sidecara nie zgadza się z archiwum.")
        original_session = _text(payload.get("session_id"))
        if session_id and original_session != session_id:
            raise ValueError("Identyfikator sesji w sidecarze jest niezgodny.")
        status = _text(payload.get("review_status")).upper()
        if status not in ("NOT_STARTED", "IN_PROGRESS", "COMPLETED"):
            raise ValueError("Nieznany status weryfikacji.")
        subjects = payload.get("subjects", {})
        if not isinstance(subjects, dict):
            raise ValueError("Nieprawidłowy rejestr GT w sidecarze.")
        revision = payload.get("review_revision", 0)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError("Nieprawidłowy numer rewizji.")
        entered_gt = sum(1 for row in subjects.values() if isinstance(row, dict) and _text(row.get("ground_truth")))
        scene_gt=payload.get('provenance',{}).get('f5a_scene_gt_v1')
        if scene_gt is not None:
            from ..ranking.mobile_f5a_scene_gt import progress
            if scene_gt.get('session_id')!=original_session or scene_gt.get('source_archive_sha256')!=source:
                raise ValueError('GT scen dotyczy innego źródła.')
            if 'exposure_review' in scene_gt:
                from ..ranking.mobile_f5a_scene_gt import exposure_progress
                counts=exposure_progress(scene_gt)
                return dict(state='in_progress',title=f"Oceniono {counts['reviewed']} z {counts['total']} zdjęć",
                    detail=f"Potwierdzone: {counts['confirmed']} · skorygowane: {counts['corrected']} · błędy: {counts['errors']} · N/A: {counts['na']}",
                    revision=revision,gt_entries=entered_gt,exposure_review=counts)
            counts=progress(scene_gt)
            return dict(state='in_progress',title=f"GT scen: {counts['approved']}/{counts['total']}",
                detail='Zatwierdzone GT scen. Kwalifikacja wyników MT/MZ i przypadki N/A są raportowane osobno.',
                revision=revision,gt_entries=entered_gt,scene_gt=counts)
        titles = {"NOT_STARTED": "Weryfikacja Desktop: nierozpoczęta", "IN_PROGRESS": "Weryfikacja Desktop: w toku", "COMPLETED": "Weryfikacja Desktop: ukończona"}
        return {"state": status.lower(), "title": titles[status], "detail": f"Rewizja {revision} • GT wpisane: {entered_gt} (nie liczba ukończonych ocen).", "revision": revision, "gt_entries": entered_gt}
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return {**empty, "state": "error", "title": "Weryfikacja Desktop: wymaga sprawdzenia", "detail": str(exc)[:210]}


def source_archive_sha(report: Any, bundle: Any = None) -> str:
    return _text(_get(bundle, "source_archive_sha256") or _get(report, "source_archive_sha256"))


def desktop_status(report: Any, bundle: Any, review_root: Path | str) -> dict[str, Any]:
    index = report_identity(report, bundle)
    return sidecar_preview(review_root, source_archive_sha(report, bundle), index["session_id"])


def selected_report_caption(report: Any, bundle: Any = None) -> str:
    meta = report_identity(report, bundle)
    fields = [f"Pakiet: {meta['package']}", f"Telefon: {meta['device']}"]
    if meta["scenario"]:
        fields.append(f"Scenariusz: {meta['scenario']}")
    if meta["session_id"]:
        fields.append(f"Sesja: {meta['session_id']}")
    return "\n".join(fields)
