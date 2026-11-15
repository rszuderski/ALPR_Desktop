"""Z4MOBREP013/014 — source-aware mobile report KPI cards and validated counters.

No mutation of mobile experiments, GT, reviews, or report statistics. Unknown
values remain unknown; never infer camera frame loss from skipped frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


def _dict(obj: Any) -> dict:
    return obj if isinstance(obj, dict) else {}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _count(obj: Any) -> int | None:
    if isinstance(obj, bool) or obj is None:
        return None
    try:
        number = float(str(obj).strip())
    except (ValueError, TypeError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        return None
    return int(number)


def _nonnegative_float(obj: Any) -> float | None:
    if isinstance(obj, bool) or obj is None:
        return None
    try:
        number = float(str(obj).strip())
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _nested(obj: Any, *keys: str) -> Any:
    for key in keys:
        row = obj
        for segment in key.split('.'):
            if isinstance(row, dict) and segment in row:
                row = row[segment]
            else:
                row = None
                break
        if row is not None and row != '':
            return row
    return None


def _counter(value: Any, keys: tuple[str, ...]) -> int | None:
    value = _dict(value)
    # The Android contract uses frames_skipped_frame_gate, older Desktop data
    # used frames_skipped_gate. If both are present and disagree, do not guess.
    candidates = [_count(value[name]) for name in keys if name in value]
    candidates = [v for v in candidates if v is not None]
    return candidates[0] if candidates and len(set(candidates)) == 1 else None


_COUNTERS = {
    'received': ('frames_received',),
    'processed': ('frames_processed', 'processed_frames'),
    'gate': ('frames_skipped_frame_gate', 'frames_skipped_gate', 'frames_skipped_by_gate'),
    'transform': ('frames_skipped_camera_transform',),
}


def _aggregate_sources(report: Any, bundle: Any) -> list[dict]:
    result = []
    for parent in (_get(bundle, 'report_payload', {}), _get(report, 'raw', {})):
        root = _dict(parent)
        for path in ('summary.frame_flow', 'summary.frame_flow_summary', 'summary.counters',
                     'frame_flow_summary', 'frame_flow.counters', 'frame_flow',
                     'capture.frame_flow', 'metrics.frame_flow', 'summary', 'counters'):
            source = _nested(root, path)
            if isinstance(source, dict):
                if source.get('_kind') == 'object':
                    source = source.get('_preview')
                if isinstance(source, dict):
                    result.append(source)
    snapshot = _verified_stored_frame_totals(report)
    if snapshot:
        result.append(snapshot)
    return result


def _verified_stored_frame_totals(report: Any) -> dict:
    """Only a snapshot tied to the same archive SHA is eligible for display."""
    raw = _dict(_get(report, 'raw', {}))
    snapshot = _dict(raw.get('_desktop_frame_flow_totals_v1'))
    source_sha = str(_get(report, 'source_archive_sha256') or '').strip().lower()
    if (snapshot.get('schema') != 'alpr.desktop.frame_flow_totals.v1'
            or len(source_sha) != 64 or snapshot.get('source_archive_sha256') != source_sha):
        return {}
    return snapshot


def _full_bucket_totals(bundle: Any) -> dict[str, int]:
    """Sum complete, per-second CSV buckets only; never sum truncated previews."""
    rows = _get(bundle, 'frame_flow_rows', ()) if bundle is not None else ()
    total = _count(_get(bundle, 'frame_flow_total')) if bundle is not None else None
    if not isinstance(rows, (list, tuple)) or not rows or total != len(rows):
        return {}
    usable = [row for row in rows if isinstance(row, dict) and
              str(row.get('bucket_kind', row.get('kind', ''))).lower() not in {'total', 'summary'}]
    if len(usable) != len(rows):
        return {}  # A mixed total/bucket table is ambiguous.
    summed: dict[str, int] = {}
    for field, keys in _COUNTERS.items():
        values = [_counter(row, keys) for row in usable]
        if all(value is not None for value in values):
            summed[field] = sum(values)  # Explicit bucket counts, not cumulative counters.
    return summed


def frame_counts(report: Any, bundle: Any = None) -> dict[str, int | None]:
    """Choose only sourced numbers; 0 is a measured zero, not an absent value."""
    values: dict[str, int | None] = {key: None for key in _COUNTERS}
    for source in _aggregate_sources(report, bundle):
        for name, keys in _COUNTERS.items():
            if values[name] is None:
                values[name] = _counter(source, keys)
    bucket_totals = _full_bucket_totals(bundle)
    for name in _COUNTERS:
        if values[name] is None:
            values[name] = bucket_totals.get(name)
    if values['processed'] is None:
        source = _dict(_get(report, 'raw', {}))
        values['processed'] = _count(_nested(source, 'summary.processed_frames'))
    if values['processed'] is None and bundle is not None:
        value = _count(_get(bundle, 'trace_total'))
        if value is not None and value > 0:
            values['processed'] = value
    # A bad or mismatched summary must not imply a valid ratio.
    received, processed = values['received'], values['processed']
    if received is not None and processed is not None and processed > received:
        values['received'] = None
    return values


def verified_frame_snapshot(report: Any, bundle: Any) -> dict:
    """Capture sourced counts for a *local report index* without editing ZIP/GT.

    The schema and the SHA let the presentation layer distinguish a cached
    aggregate from real-time revalidation of the original .alprsession.
    """
    validation = _get(bundle, 'validation')
    report_sha = str(_get(report, 'source_archive_sha256') or '').strip().lower()
    bundle_sha = str(_get(bundle, 'source_archive_sha256') or '').strip().lower()
    if (bundle is None or _get(validation, 'ok') is not True
            or len(report_sha) != 64 or report_sha != bundle_sha):
        return {}
    # Reject inconsistent direct aggregates before frame_counts can mask the
    # invalid denominator as "unknown" while leaving a bogus gate count.
    for aggregate in _aggregate_sources(report, bundle):
        declared_received = _counter(aggregate, _COUNTERS['received'])
        if declared_received is None:
            continue
        for dimension in ('processed', 'gate', 'transform'):
            value = _counter(aggregate, _COUNTERS[dimension])
            if value is not None and value > declared_received:
                return {}
    totals = frame_counts(report, bundle)
    received, processed, gate, transform = (totals[k] for k in ('received', 'processed', 'gate', 'transform'))
    if received is not None:
        if any(n is not None and n > received for n in (processed, gate, transform)):
            return {}  # Source counters disagree; do not persist questionable totals.
    if all(n is None for n in totals.values()):
        return {}
    snapshot = {
        'schema': 'alpr.desktop.frame_flow_totals.v1',
        'source_archive_sha256': bundle_sha,
        'source_kind': 'validated_mobile_report_bundle',
    }
    columns = {'received': 'frames_received', 'processed': 'frames_processed',
               'gate': 'frames_skipped_frame_gate', 'transform': 'frames_skipped_camera_transform'}
    snapshot.update({columns[k]: v for k, v in totals.items() if v is not None})
    return snapshot


def _latency_samples(report: Any, bundle: Any = None) -> int | None:
    latency = _dict(_get(report, 'latency', {}))
    for path in ('pipeline.count', 'pipeline.n', 'pipeline.sample_count',
                 'pipeline.samples', 'pipeline_ms.count', 'e2e.count',
                 'end_to_end.count', 'latency_pipeline_ms_count'):
        value = _nested(latency, path)
        number = _count(value)
        if number is not None:
            return number
    # Trace count is a proxy for sample size ONLY if an indexed bundle exists.
    # Standalone stored reports may lack the original traces.
    if bundle is not None:
        total = _count(_get(bundle, 'trace_total'))
        if total is not None and total > 0:
            return total
    return None


def _latency_p95(report: Any) -> float | None:
    # Same field precedence as z4_mobile_report_browser._pipeline_p95.
    latency = _dict(_get(report, 'latency', {}))
    value = _nested(latency, 'pipeline.p95_ms', 'pipeline.latency_pipeline_ms_p95',
                    'pipeline_ms_p95', 'latency_pipeline_ms_p95', 'p95_ms')
    return _nonnegative_float(value)


def _number(value: int | None) -> str:
    return f'{value:,}'.replace(',', ' ') if value is not None else '—'


def _metric_ms(value: float | None) -> str:
    return f'{value:.1f} ms'.replace('.', ',') if value is not None else '—'


@dataclass(frozen=True)
class Card:
    heading: str
    value: str
    subtitle: str
    explanation: str


def quality_state(report: Any, review: dict | None) -> tuple[str, str, str]:
    quality = _dict(_get(report, 'quality', {}))
    phone = ('Nieustalone' if quality.get('quality_source') == 'human_review'
             else 'Brak GT' if quality.get('available') is False
             else 'GT dostępny' if quality.get('available') is True else 'Nieustalone')
    status = str(_dict(review).get('state') or 'unverified').lower()
    desktop_map = {
        'completed': 'Desktop: ukończona',
        'in_progress': 'Desktop: w toku',
        'not_started': 'Desktop: nierozpoczęta',
        'none': 'Desktop: brak oceny',
        'error': 'Desktop: sprawdź zapis',
        'unverified': 'Desktop: nieustalony',
    }
    desktop = desktop_map.get(status, 'Desktop: nieustalony')
    detail = str(_dict(review).get('detail') or '').strip()
    return phone, desktop, detail


def cards_for_report(report: Any, bundle: Any = None, review: dict | None = None) -> dict[str, Card]:
    flow = frame_counts(report, bundle)
    processed, received, gate, transformed = (flow[x] for x in ('processed', 'received', 'gate', 'transform'))
    summary = f'{_number(processed)} z {_number(received)}' if received is not None else f'{_number(processed)} klatek'
    if gate is not None:
        flow_hint = f'{_number(gate)} pominięte przez bramkę'
    elif received is None:
        flow_hint = 'Liczba odebranych: brak danych'
    else:
        flow_hint = 'Brak liczby pominiętych przez bramkę'
    flow_tip = ('KLATKI: wejścia do analizatora kamery ≠ klatki analizowane przez ALPR. '
                f'Odebrane: {_number(received)}. Zakończona analiza: {_number(processed)}. '
                f'Pominięte przez frame_gate: {_number(gate)}. '
                f'Pominięte przy transformacji: {_number(transformed)}. '
                'Pominięcia przez bramkę są świadome — nie są utratą danych ani nie mierzą strat CameraX. '
                '„—” oznacza brak wiarygodnego licznika.')

    p95 = _latency_p95(report)
    n = _latency_samples(report, bundle)
    latency_hint = f'95. percentyl · n={n}' if n is not None else '95. percentyl · n nieznane'
    latency_tip = (f'E2E p95: {_metric_ms(p95)}. To 95. percentyl opóźnienia całego potoku '
                   'MT→MZ i pozostałych etapów (nie tylko inferencji modeli). '
                   f'Liczebność śladu / pomiarów: {_number(n)}. '
                   'Przy małej próbie p95 należy traktować opisowo, jako wynik pilotażowy.')

    phone, desktop, review_detail = quality_state(report, review)
    quality_tip = ('GT z Androida opisuje jakość zapisaną w pierwotnym raporcie. '
                   'Weryfikacja Desktop to osobny sidecar przypięty SHA-256 archiwum; '
                   'wpisane GT nie jest liczbą ukończonych ocen. '
                   f'Android: {phone}. {desktop}. {review_detail or ""} '
                   'Nie wnioskuj o Exact Match/CER wyłącznie z faktu wpisania GT.')

    if bundle is not None:
        validation = _get(bundle, 'validation', None)
        hashes = _count(_get(validation, 'checked_hashes'))
        ok = _get(validation, 'ok', None)
        warnings = _get(validation, 'warnings', ()) or ()
        errors = _get(validation, 'errors', ()) or ()
        if ok is False:
            source_value, source_hint = 'Błąd archiwum', f'Błędy: {len(errors)}'
        elif ok is True and hashes is not None and hashes > 0:
            source_value, source_hint = 'SHA zweryfikowane', f'{hashes} plików · ostrzeżenia: {len(warnings)}'
        else:
            source_value, source_hint = 'Odczytano raport', 'Brak dowodu weryfikacji SHA'
        source_tip = (f'Kontrola przy bieżącym odczycie archiwum: status {ok}; '
                      f'porównane SHA: {_number(hashes)}; ostrzeżenia: {len(warnings)}; '
                      f'błędy: {len(errors)}. Samo otwarcie zapisanego wpisu to nie ponowna walidacja.')
    else:
        source_value, source_hint = 'Zapisany raport', 'Bez ponownej walidacji archiwum'
        source_tip = ('Wyświetlono raport z lokalnego katalogu raportów. '
                      'To NIE jest nowa kontrola CRC/SHA plików źródłowych. '
                      'Aby sprawdzić integralność, załaduj oryginalne .alprsession.')
    sha = str(_get(bundle, 'source_archive_sha256') or _get(report, 'source_archive_sha256') or '').strip()
    if sha:
        source_tip += f' SHA-256 źródła: {sha}.'

    return {
        'integrity': Card('Źródło', source_value, source_hint, source_tip),
        'frames': Card('Klatki ALPR', summary, flow_hint, flow_tip),
        'quality': Card('GT i weryfikacja', 'Android: ' + phone, desktop, quality_tip),
        'latency': Card('Opóźnienie E2E p95', _metric_ms(p95), latency_hint, latency_tip),
    }


class HoverCardHelp:
    """Small tooltip attached to Tk labels; created lazily on first hover."""
    def __init__(self, parent, text_getter):
        self.parent = parent
        self.text_getter = text_getter
        self._pending = None
        self._popup = None
        self._anchor = None

    def bind(self, widget, key: str) -> None:
        widget.bind('<Enter>', lambda event, selected=key: self._schedule(event, selected), add='+')
        widget.bind('<Leave>', lambda _event: self.hide(), add='+')
        widget.bind('<ButtonPress>', lambda _event: self.hide(), add='+')

    def _schedule(self, event, key: str) -> None:
        self.hide()
        self._anchor = event.widget
        try:
            self._pending = self.parent.after(350, lambda selected=key: self._show(selected))
        except Exception:
            self._pending = None

    def _show(self, key: str) -> None:
        self._pending = None
        try:
            import tkinter as tk
            if not self.parent.winfo_exists():
                return
            content = str(self.text_getter(key) or '').strip()
            if not content:
                return
            popup = tk.Toplevel(self.parent)
            popup.withdraw()
            popup.overrideredirect(True)
            popup.configure(bg='#313633')
            label = tk.Label(popup, text=content, bg='#313633', fg='#f7f8f4',
                             font=('Segoe UI', 9), justify=tk.LEFT, anchor=tk.NW,
                             wraplength=370, padx=12, pady=9)
            label.pack()
            popup.update_idletasks()
            anchor = self._anchor or self.parent
            px = anchor.winfo_rootx() + 8
            py = anchor.winfo_rooty() + anchor.winfo_height() + 5
            sw, sh = anchor.winfo_screenwidth(), anchor.winfo_screenheight()
            px = max(0, min(px, sw - popup.winfo_reqwidth() - 8))
            if py + popup.winfo_reqheight() > sh - 8:
                py = max(0, anchor.winfo_rooty() - popup.winfo_reqheight() - 5)
            popup.geometry(f'+{px}+{py}')
            popup.deiconify()
            self._popup = popup
        except Exception:
            self.hide()

    def hide(self) -> None:
        if self._pending is not None:
            try:
                self.parent.after_cancel(self._pending)
            except Exception:
                pass
        self._pending = None
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
        self._popup = None


def summary_measure_rows(report: Any, bundle: Any = None) -> list[tuple[str, str, str]]:
    flow = frame_counts(report, bundle)
    result = [
        ('Klatki odebrane', _number(flow['received']),
         'Wejścia do analizatora CameraX. To nie liczba klatek przetworzonych przez MT.'),
        ('Klatki przetworzone', _number(flow['processed']),
         'Ukończone przebiegi ALPR / klatki ze śladem inferencji.'),
        ('Pominięte przez frame_gate', _number(flow['gate']),
         'Celowo pominięte przez aplikację, nie utracone przez CameraX.'),
    ]
    if flow['transform'] is not None:
        result.append(('Pominięte przy transformacji kamery', _number(flow['transform']),
                       'Osobny licznik, nie sumować go z pominięciami frame_gate jako „straty kamery”.'))
    p95 = _latency_p95(report)
    n = _latency_samples(report, bundle)
    result.append(('Opóźnienie E2E p95', _metric_ms(p95),
                   '95. percentyl pełnego potoku ALPR; ' +
                   (f'n={n} z pełnego śladu/podsumowania.' if n is not None else 'liczebność nieustalona.')))
    return result


def device_details(report: Any) -> str:
    device = _dict(_get(report, 'device', {}))
    android = str(device.get('android_version') or '').strip()
    sdk = str(device.get('android_sdk') or '').strip()
    if android:
        return f'Android {android}' + (f' · API {sdk}' if sdk else '')
    if sdk:
        return f'Android API {sdk}'
    return 'Szczegóły techniczne urządzenia dostępne w zakładce Parametry / Dane JSON.'
