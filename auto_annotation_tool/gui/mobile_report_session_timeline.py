"""Measured session timeline for mobile ALPR; read-only and fail-closed.

Trace ``elapsed_ms``/``timestamp_ms`` mark a telemetry measurement, not a
camera sensor exposure. All x-values are taken from saved evidence.  This
module never synthesizes timestamps for unmeasured/missing frames.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _aware_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def session_time_context(bundle: Any, report: Any) -> dict[str, Any]:
    """Select trusted recording window; do not guess start from first trace."""
    session = getattr(bundle, 'collection_session', None)
    if not isinstance(session, dict):
        session = {}
    raw = getattr(report, 'raw', None)
    if not isinstance(raw, dict):
        raw = {}
    start = _aware_iso(session.get('started_at'))
    end = _aware_iso(session.get('finished_at'))
    started_ms = _number(raw.get('session_started_ms'))
    finished_ms = _number(raw.get('session_finished_ms'))
    duration_ms = _number(raw.get('session_duration_ms'))

    if start is not None and end is not None and end > start:
        actual = (end - start).total_seconds() * 1000.0
        if duration_ms is None or abs(duration_ms - actual) <= 1000:
            duration_ms = actual
        else:
            # Conflicting evidence: do not construct an artificial X clock.
            return {'duration_ms': None, 'start_ms': None, 'start': start, 'end': end,
                    'reason': 'Niespójny czas trwania sesji'}
        epoch_start_ms = start.timestamp() * 1000.0
        if started_ms is not None and abs(epoch_start_ms - started_ms) > 1000:
            return {'duration_ms': None, 'start_ms': None, 'start': start, 'end': end,
                    'reason': 'Niespójny początek sesji'}
        started_ms = epoch_start_ms
    elif started_ms is not None and finished_ms is not None and finished_ms > started_ms:
        actual = finished_ms - started_ms
        if duration_ms is not None and abs(duration_ms - actual) > 1000:
            return {'duration_ms': None, 'start_ms': None, 'start': None, 'end': None,
                    'reason': 'Niespójny czas trwania sesji'}
        duration_ms = actual
        start = datetime.fromtimestamp(started_ms / 1000.0, timezone.utc)
        end = datetime.fromtimestamp(finished_ms / 1000.0, timezone.utc)
    else:
        return {'duration_ms': None, 'start_ms': None, 'start': start, 'end': end,
                'reason': 'Brak pełnego czasu sesji'}
    if duration_ms is None or duration_ms <= 0:
        return {'duration_ms': None, 'start_ms': None, 'start': start, 'end': end,
                'reason': 'Niepoprawny czas sesji'}
    return {'duration_ms': duration_ms, 'start_ms': started_ms,
            'start': start, 'end': end, 'reason': ''}


def session_elapsed_seconds(rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> tuple[float, ...] | None:
    """Use a common true session clock for ALL series or do not offer it.

    Elapsed trace timestamps may be validated against device wall-clock stamps
    when present; a mismatch prevents displaying a seemingly precise session
    timeline. We accept up to 0.5 s recorder jitter, not reordered samples.
    """
    start_ms = context.get('start_ms')
    duration_ms = context.get('duration_ms')
    if start_ms is None or duration_ms is None or not rows:
        return None
    result = []
    for row in rows:
        elapsed = _number(row.get('elapsed_ms'))
        wall = _number(row.get('timestamp_ms'))
        if elapsed is None:
            if wall is None:
                return None
            elapsed = wall - start_ms
        elif wall is not None and abs((wall - start_ms) - elapsed) > 500:
            return None
        if not (0 <= elapsed <= duration_ms + 1000):
            return None
        result.append(elapsed / 1000.0)
    if any(b <= a for a, b in zip(result, result[1:])):
        return None
    return tuple(result)


def fmt_elapsed(seconds: float, *, milliseconds: bool = False) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return '—'
    if milliseconds:
        total = int(round(seconds * 1000))
        hours, rem = divmod(total, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        sec, milli = divmod(rem, 1000)
        return (f'{hours}:{minutes:02d}:{sec:02d}.{milli:03d}' if hours
                else f'{minutes}:{sec:02d}.{milli:03d}').replace('.', ',')
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    return f'{hours}:{minutes:02d}:{sec:02d}' if hours else f'{minutes}:{sec:02d}'


def _local_clock(moment: datetime | None) -> str:
    if moment is None or moment.tzinfo is None:
        return '—'
    local = moment.astimezone()
    offset = local.strftime('%z')
    zone = f'UTC{offset[:3]}:{offset[3:]}' if offset else ''
    return f'{local:%d.%m %H:%M:%S} ({zone})'


def session_caption(context: Mapping[str, Any]) -> str:
    duration = context.get('duration_ms')
    if duration is None:
        return 'Sesja: brak wiarygodnego przedziału czasu; dostępna oś ID klatki.'
    return (f'Czas trwania sesji: {fmt_elapsed(duration / 1000, milliseconds=True)}'
            f'  ·  start {_local_clock(context.get("start"))}'
            f'  ·  koniec {_local_clock(context.get("end"))}')


def time_ticks(lo: float, hi: float, *, max_labels: int = 6) -> tuple[tuple[float, str], ...]:
    """Nice readable wall-duration ticks, anchored to start of the session."""
    if not all(map(math.isfinite, (lo, hi))) or hi <= lo:
        return ()
    span = hi - lo
    candidates = (.05, .1, .2, .5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900,
                  1200, 1800, 3600, 7200, 14400, 28800, 86400)
    target = span / max(1, max_labels - 1)
    step = next((s for s in candidates if s >= target), 86400 * math.ceil(target / 86400))
    first = int(math.ceil((lo - step * 1e-6) / step))
    last = int(math.floor((hi + step * 1e-6) / step))
    marks = []
    for n in range(first, min(last + 1, first + max_labels + 2)):
        value = round(n * step, 6)
        if lo - 1e-8 <= value <= hi + 1e-8:
            marks.append((value, fmt_elapsed(value, milliseconds=(step < 1))))
    return tuple(marks)


def point_caption(row: Mapping[str, Any], time_seconds: float | None,
                  stage_title: str, value_ms: float, context: Mapping[str, Any]) -> str:
    frame = str(row.get('frame_id') or '').strip()
    parts = [(f'Klatka {frame}' if frame else 'Pomiar')]
    if time_seconds is not None:
        parts.append(f'pomiar +{fmt_elapsed(time_seconds, milliseconds=True)} od startu')
    elif row.get('timestamp_ms'):
        parts.append('czas od startu: niepotwierdzony')
    stamp = _number(row.get('timestamp_ms'))
    if stamp is not None and context.get('start_ms') is not None and time_seconds is not None:
        if abs((stamp - context['start_ms']) / 1000.0 - time_seconds) <= .5:
            utc = datetime.fromtimestamp(stamp / 1000.0, timezone.utc)
            parts.append('zapis '+utc.astimezone().strftime('%H:%M:%S.%f')[:-3])
    parts.append(f'{stage_title}: {value_ms:.1f} ms')
    return '  ·  '.join(parts)


STAGE_DESCRIPTIONS = {
    'total_ms': 'ALPR: cały proces',
    'plate_inference_ms': 'MT: inferencja modelu tablic',
    'character_inference_ms': 'MZ: inferencja modelu znaków',
    'vehicle_inference_ms': 'MP: inferencja modelu pojazdów',
}
STAGE_SHORT = {
    'total_ms': 'ALPR: całość',
    'plate_inference_ms': 'MT: model tablic',
    'character_inference_ms': 'MZ: model znaków',
    'vehicle_inference_ms': 'MP: model pojazdów',
}
