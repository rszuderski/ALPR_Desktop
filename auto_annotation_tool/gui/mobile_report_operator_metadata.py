"""Read-only presentation of operator metadata actually exported by Android.

Audited Android v1 archives export experiment.notes and experiment.session.notes.
They do not export a separate operator session name. Never reinterpret a series,
scenario, archive filename, device/model name or session ID as that missing name.
No archive reads, persistence, review decisions or metric changes happen here.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


PHONE_NOTE_PATHS = (('experiment', 'notes'), ('experiment', 'session', 'notes'))
_MISSING = object()


def preserve_phone_notes(original: dict, compacted: dict) -> None:
    """Only the existing archive fields bypass diagnostic text truncation."""
    for path in PHONE_NOTE_PATHS:
        value = _at(original, path)
        if not isinstance(value, str):
            continue
        target = compacted
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target[path[-1]] = value


@dataclass(frozen=True)
class PhoneField:
    state: str  # PRESENT / EMPTY / MISSING / ERROR / TRUNCATED
    value: str = ''
    source: str = ''
    detail: str = ''


@dataclass(frozen=True)
class PhoneSessionMetadata:
    name: PhoneField
    note: PhoneField


def _at(payload: dict, path: tuple[str, ...]):
    node: Any = payload
    for part in path:
        if not isinstance(node, dict):
            return None  # Present but malformed parent is not an absent field.
        if part not in node:
            return _MISSING
        node = node[part]
    return node


def phone_session_metadata(report: Any, bundle: Any = None) -> PhoneSessionMetadata:
    """Prefer the matching verified source, otherwise use the saved report payload.

    The existing bundle cache/recovery supplies complete data for old compacted
    notes. A preview dictionary is never converted into invented operator text.
    """
    name = PhoneField('MISSING', detail='Brak osobnego pola nazwy sesji w sprawdzonym formacie Androida.')
    payload = getattr(report, 'raw', None)
    if bundle is not None:
        candidate = getattr(bundle, 'report', None)
        fingerprint = str(getattr(report, 'source_archive_sha256', '')).lower()
        same = (bool(re.fullmatch('[a-f0-9]{64}', fingerprint))
                and str(getattr(bundle, 'source_archive_sha256', '')).lower() == fingerprint
                and getattr(candidate, 'report_id', None) == getattr(report, 'report_id', None))
        if not same or not getattr(getattr(bundle, 'validation', None), 'ok', False):
            error = PhoneField('ERROR', detail='Niepotwierdzona tożsamość lub integralność źródła.')
            return PhoneSessionMetadata(error, error)
        payload = getattr(bundle, 'report_payload', None)
    if not isinstance(payload, dict):
        return PhoneSessionMetadata(name, PhoneField('ERROR', detail='Nieczytelna struktura raportu.'))
    values = [(path, _at(payload, path)) for path in PHONE_NOTE_PATHS]
    found = [(path, value) for path, value in values if value is not _MISSING]
    if not found:
        return PhoneSessionMetadata(name, PhoneField('MISSING', source='archive' if bundle is not None else 'index'))
    if any(isinstance(v, dict) and '_preview_text' in v for _, v in found):
        return PhoneSessionMetadata(name, PhoneField('TRUNCATED', detail='Zapisany indeks zawiera tylko fragment notatki.'))
    if any(not isinstance(v, str) for _, v in found):
        return PhoneSessionMetadata(name, PhoneField('ERROR', detail='Pole notatki nie jest tekstem.'))
    if len({v for _, v in found}) != 1:
        return PhoneSessionMetadata(name, PhoneField('ERROR', detail='Sprzeczne kopie notatki w raporcie.'))
    value = found[0][1]
    return PhoneSessionMetadata(name, PhoneField(
        'PRESENT' if value.strip() else 'EMPTY', value,
        ' / '.join('report.json → ' + '.'.join(path) for path, _ in found)))


def phone_name_caption(metadata: PhoneSessionMetadata) -> str:
    if metadata.name.state == 'PRESENT':
        return metadata.name.value
    if metadata.name.state == 'ERROR':
        return 'Błąd odczytu nazwy sesji'
    if metadata.name.state == 'EMPTY':
        return 'Pusta nazwa sesji w archiwum'
    return 'Brak zapisanej nazwy sesji'


def phone_note_caption(metadata: PhoneSessionMetadata) -> str:
    note = metadata.note
    if note.state == 'PRESENT':
        return note.value
    return {
        'EMPTY': 'Notatka w archiwum jest pusta.',
        'MISSING': 'Brak notatki w archiwum' if note.source == 'archive' else 'Brak notatki w zapisanym raporcie.',
        'TRUNCATED': 'Pełna notatka wymaga odczytu oryginalnego archiwum. Zapisany indeks zawiera tylko fragment.',
        'ERROR': 'Błąd odczytu notatki z telefonu: ' + note.detail,
    }[note.state]


def phone_list_caption(metadata: PhoneSessionMetadata) -> str:
    """A short list preview; the separate note section always exposes full text."""
    if metadata.name.state == 'PRESENT':
        return metadata.name.value
    if metadata.note.state == 'PRESENT':
        preview = ' '.join(metadata.note.value.split())
        return 'Brak nazwy · notatka:\n' + (preview[:85] + '…' if len(preview) > 86 else preview)
    return phone_name_caption(metadata)
