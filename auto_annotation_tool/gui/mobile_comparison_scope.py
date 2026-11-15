"""Source-bound eligibility for timing versus full assisted quality comparisons."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json

SCOPES = {'quality': 'Jakość — MT/MZ (pełne GT)',
          'performance': 'Wydajność — czasy z telefonu'}

def scope_value(value, count=2):
    value = value or ('single' if count == 1 else 'quality')
    if value not in (*SCOPES, 'single'):
        raise ValueError('Nieznany zakres porównania.')
    return value

def trace_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), default=str).encode('utf-8')).hexdigest()

@dataclass(frozen=True)
class ComparisonAccess:
    state: str
    performance_eligible: bool = False
    quality_ready: bool = False
    performance_reason: str = ''
    quality_reason: str = ''
    record: object = None
    traces: tuple = ()
    bundle: object = None

    def allowed(self, scope):
        return self.performance_eligible if scope == 'performance' else self.quality_ready

    def reason(self, scope):
        return self.performance_reason if scope == 'performance' else self.quality_reason

    @property
    def description(self):
        return ('Wydajność: ' + ('dostępna' if self.performance_eligible else 'niedostępna')
            + '; pełna jakość: ' + ('dostępna' if self.quality_ready else 'niedostępna')
            + ('. ' + self.quality_reason if not self.quality_ready else ''))

def assess_import(report, source, results_root, review_root=None):
    """Read the original again; a cached bundle or store label is not evidence."""
    from .mobile_report_source_recovery import restore_source
    from .mobile_import_comparison import record_from_import, _identity, _read_all_traces, _sha_file
    from .mobile_compare_review_gate import imported_verification
    from ..config import CONFIG
    if review_root is None:
        review_root = Path(getattr(CONFIG, 'DIR_7_RANKINGS_MOBILE_PACKAGES', CONFIG.DIR_7_RANKINGS/'mobile_packages'))
    try:
        restored = restore_source(report, source)
        if not restored.ok:
            raise ValueError(restored.detail)
        bundle = restored.bundle
        identity = _identity(report, bundle)
        sid = identity['session_id']
        # No priority fallback may conceal disagreement between source identities.
        ids = {sid}
        for obj in (getattr(report, 'experiment_index', {}), getattr(bundle, 'experiment_session', {}),
                    getattr(bundle, 'collection_session', {})):
            if isinstance(obj, dict):
                ids.update(str(obj[k]) for k in ('session_id', 'experiment_session_id') if obj.get(k))
        if len(ids) != 1:
            raise ValueError('Sprzeczne identyfikatory session_id w źródle i raporcie.')
        gate = imported_verification(sid, bundle.source_archive_sha256, results_root, review_root)
        if not gate.approved:
            return ComparisonAccess(gate.state, performance_reason=gate.reason, quality_reason=gate.reason)
        record, rows, warnings = record_from_import(report, bundle, results_root, review_root=review_root)
        if (record.review_revision != gate.revision or record.directory != gate.export_dir
                or not (record.operator_approved or record.canvas_ready)):
            raise ValueError('Źródło i zatwierdzony eksport nie są zgodne.')
        _, trace_state = _read_all_traces(bundle)
        if _sha_file(Path(source)) != bundle.source_archive_sha256:
            raise ValueError('Archiwum zmieniło się podczas odczytu śladów czasu.')
        timing = trace_state == 'complete' and bool(rows) and all(record.latency[k].n > 0 for k in ('mt', 'pipeline'))
        reason = 'Zweryfikowane czasy Androida. Każdy etap ma własne n.' if timing else (
            'Brak pełnych źródłowych śladów czasu MT/potoku; porównanie wydajności niedostępne (N/A).')
        if timing and not record.latency['mz'].n:
            reason += ' MZ: brak wykonanych pomiarów (N/A).'
        report_identity = {k: str(getattr(report, k, '') or '') for k in
            ('report_id', 'package_id', 'variant_id', 'measured_at', 'source_archive_sha256')}
        report_identity['experiment_index'] = {'experiment_session_id': sid}
        source_info = dict(path=str(Path(source).resolve()), results_root=str(Path(results_root).resolve()),
            review_root=str(Path(review_root).resolve()), report=report_identity,
            trace_sha256=trace_digest(rows), trace_count=len(rows), warnings=list(warnings),
            conditions_report=copy.deepcopy(bundle.report), conditions_bundle=bundle)
        record = replace(record, comparison_source=source_info)
        return ComparisonAccess(gate.state, timing, bool(gate.ready and record.canvas_ready),
            reason, 'Pełna ocena MT/MZ dostępna.' if gate.ready else gate.reason,
            record, tuple(rows), bundle)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        text = f'Nie potwierdzono źródła lub eksportu: {type(exc).__name__}: {exc}'
        return ComparisonAccess('INVALID', performance_reason=text, quality_reason=text)

def _refresh_record(record, scope):
    evidence = record.comparison_source
    if not evidence:
        raise ValueError('Brak zweryfikowanego oryginalnego źródła porównania.')
    access = assess_import(SimpleNamespace(**evidence['report']), evidence['path'],
        evidence['results_root'], evidence['review_root'])
    if not access.allowed(scope):
        raise ValueError(access.reason(scope))
    current = access.record
    for field in ('session_id', 'source_sha256', 'review_revision', 'manifest_sha256',
                  'latency', 'scenario', 'apk_sha256', 'device', 'mt_checkpoint', 'mz_checkpoint', 'model_identity',
                  'canvas', 'review_status', 'operator_approved'):
        if getattr(current, field) != getattr(record, field):
            raise ValueError('Źródło lub rewizja zmieniły się od wyboru sesji. Otwórz porównanie ponownie.')
    if evidence['trace_sha256'] != current.comparison_source['trace_sha256']:
        raise ValueError('Ślady czasowe zmieniły się od wyboru sesji.')
    for field in ('input_mode', 'input_context'):
        if getattr(current, field, None) != getattr(record, field, None):
            raise ValueError('Zamrożone wejście zmieniło się od wyboru sesji. Otwórz porównanie ponownie.')
    return access

def validate_comparison(records, row_sets=None, *, comparison_scope=None, require_sources=False):
    """Common guard at launch, chart creation and both sides of atomic export."""
    records = list(records)
    from ..ranking.mobile_tv_bbox_evidence import cohort_conflict as tv_conflict
    conflict = tv_conflict(getattr(r, 'input_context', None) or {'input_mode': getattr(r, 'input_mode', 'CAMERA_FULL_FRAME')} for r in records)
    if conflict:
        raise ValueError('NOT_COMPARABLE / N/A: ' + conflict)
    from ..ranking.mobile_mz_input_evidence import diagnostic_cohort_conflict
    if diagnostic_cohort_conflict(getattr(r, 'measurement_purpose', '') for r in records):
        raise ValueError('NOT_COMPARABLE / N/A: cele diagnostyczne i oficjalne wymagają odrębnych kohort.')
    scope = scope_value(comparison_scope, len(records))
    if not records or (require_sources and scope != 'single' and len(records) < 2):
        raise ValueError('Porównanie wymaga co najmniej dwóch różnych sesji.')
    if scope == 'single' and len(records) != 1:
        raise ValueError('Wykresy jednej sesji wymagają dokładnie jednego źródła.')
    if len({r.session_id for r in records}) != len(records) or len({r.source_sha256 for r in records}) != len(records):
        raise ValueError('Nie można zliczać tej samej sesji ani tej samej kopii archiwum dwa razy.')
    if row_sets is not None and len(row_sets) != len(records):
        raise ValueError('Brak kompletu śladów wybranych sesji.')
    if len(records) > 1:
        if scope == 'quality' and any(not r.canvas_ready for r in records):
            raise ValueError('Wymagany stan READY: tylko zatwierdzone i kompletne oceny MT+MZ są dopuszczone do pełnej jakości.')
        if len({r.scenario for r in records}) > 1:
            raise ValueError('Rozdziel scenariusze DAY i NIGHT — nie są jednym kontrolowanym porównaniem.')
        if len({r.apk_sha256 for r in records}) > 1 or len({r.device for r in records}) > 1:
            raise ValueError('Różne APK lub telefony: porównaj osobne grupy; warunki nie są zgodne.')
        if len({r.mt_checkpoint for r in records}) > 1 or len({r.mz_checkpoint for r in records}) > 1:
            raise ValueError('Różne checkpointy MT/MZ: nie jest to izolowany wpływ kwantyzacji.')
    if scope == 'performance' or (scope == 'quality' and (require_sources or any(r.comparison_source for r in records))):
        for i, record in enumerate(records):
            access = _refresh_record(record, scope)
            if row_sets is not None and trace_digest(row_sets[i]) != access.record.comparison_source['trace_sha256']:
                raise ValueError('Ślady w oknie nie odpowiadają oryginalnemu archiwum.')
    return scope

def comparison_warnings(records, scope):
    from .mobile_comparison_core import validate_selection
    from .mobile_report_scientific_comparability import assess_cohort
    warnings = list(validate_selection(records))
    if scope == 'performance':
        replacements={
            'Nie wszystkie sesje mają ukończoną ocenę Canvas z GT — braków nie traktować jako 0%.':
                'Zakres pomiarów jakości jest ograniczony; nie blokuje potwierdzonych czasów Androida.',
            'Ocena naukowa nie jest formalnie zweryfikowana; jakości używaj wyłącznie z Canvas (ASSISTED).':
                'Warunki kontrolowanego eksperymentu nie zostały formalnie potwierdzone.'}
        warnings=[replacements.get(w,w) for w in warnings]
    warnings.append('Porównanie opisowe. Dopuszczenie techniczne nie zatwierdza kontrolowanych warunków eksperymentu.')
    reports = [r.comparison_source['conditions_report'] for r in records if r.comparison_source]
    for record in records:
        evidence = record.comparison_source
        if not evidence:
            continue
        warnings.extend(evidence['warnings'])
        audit = assess_cohort(evidence['conditions_report'], reports, bundle=evidence['conditions_bundle'])
        for row in audit['rows']:
            # Retain independent conditions, including thermal/model/APK evidence.
            warnings.append(f'{record.session_id}: {row[0]} — {row[1]} / {row[2]} · {row[3]}. {row[4]}')
    if scope == 'performance':
        warnings.append('Pełna jakość MT/MZ poza zakresem tego porównania; brak GT nie jest 0%.')
    if len({r.latency['mz'].n for r in records}) > 1:
        warnings.append('Różne liczby ważnych śladów MZ; n podano osobno. MZ może sumować wiele wywołań w śladzie.')
    for r in records:
        if not r.latency['mz'].n:
            warnings.append(f'{r.session_id}: MZ N/A — brak ważnych pomiarów etapu, nie 0 ms.')
    return list(dict.fromkeys(warnings))
