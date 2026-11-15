"""F5A performance projection for the existing Z4 analyzer (no inference/GT).

Archive validation remains owned by the native reader. Descriptor equivalence
here applies only to descriptive performance, never to F4/F6 or GT provenance.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re
from statistics import median

ALGORITHM_ID = 'alpr.desktop.f5a.performance.v1'
NEW_METRICS = {
    'mp_trace_p50_ms': ('MP — suma inferencji w śladzie (p50)', 'ms', 'mp_stage_per_trace'),
    'mp_trace_p95_ms': ('MP — suma inferencji w śladzie (p95)', 'ms', 'mp_stage_per_trace'),
    'mt_trace_p50_ms': ('MT — suma inferencji w śladzie (p50)', 'ms', 'mt_stage_per_trace'),
    'mt_trace_p95_ms': ('MT — suma inferencji w śladzie (p95)', 'ms', 'mt_stage_per_trace'),
    'mp_calls': ('Zapisane wywołania MP', 'wywołania', 'recorded_mp_call'),
    'mt_calls_per_exposure': ('MT — wywołania na ekspozycję 30 s', 'wywołania/ekspozycję', 'mapped_30s_exposure'),
    'mz_calls_per_exposure': ('MZ — wywołania na ekspozycję 30 s', 'wywołania/ekspozycję', 'mapped_30s_exposure'),
}
METRIC_IDS = ('mp_trace_p50_ms', 'mt_trace_p50_ms', 'mz_trace_p50_ms', 'pipeline_p50_ms',
              'mp_trace_p95_ms', 'mt_trace_p95_ms', 'mz_trace_p95_ms', 'pipeline_p95_ms',
              'mp_calls', 'mt_calls', 'mz_calls', 'mt_p50_ms', 'mt_p95_ms', 'mz_call_p95_ms',
              'mt_calls_per_exposure', 'mz_calls_per_exposure')
DEFAULT_FIGURES = METRIC_IDS[:11]
EXCLUDED_STATUSES = {'scene_superseded', 'runtime_error', 'pipeline_error', 'cancelled', 'not_run'}
COMPLETED_STATUSES = {'preliminary', 'recognized', 'no_plate', 'stabilizing', 'scan_queue_updated', 'completed', 'complete'}
LIMITATIONS = [
    'Porównanie opisowe zapisanej konfiguracji R0/R1/R2; bez testów istotności i wniosku przyczynowego.',
    'Czasy: mediana percentyli sesyjnych, z równą wagą sesji. Nie jest to percentyl połączonych śladów.',
    'Ślady i wywołania nie są niezależnymi ekspozycjami. Plan 40 zdjęć × 3 powtórzenia nie dowodzi 120 zmapowanych ekspozycji.',
    'SCENE_FRAME jest zdarzeniem telefonu, nie identyfikatorem zdjęcia TV. Brak mapowania ekspozycji: N/A dla wywołań na ekspozycję.',
    'ALPR_MT_TRACE to zakończony ślad z dodatnim czasem MT. Nie obejmuje pełnej latencji kamera–ekran ani TV–HUD; zegary nie są synchronizowane.',
    'MP/MT/MZ: suma czasów inferencji etapu w śladzie, bez preprocessingu. Nie sumujemy median i nie dzielimy percentyli przez liczbę wywołań.',
    'Liczniki sumują wyłącznie jawnie zapisane wartości, także w anulowanych śladach. Brak licznika nie oznacza zera; suma częściowa jest oznaczona.',
    'Zgodność zapisanej geometrii nie potwierdza fizycznego ustawienia aparatu. Stan termiczny i inne warunki obserwowane pozostają w diagnostyce.',
    'R0 ma MP i vehicle_tracking OFF, R1/R2 ON. Wynik dotyczy tych konfiguracji; nie izoluje samego kosztu MP.',
    'CPU2 oznacza konfigurację; efektywna liczba wątków, przypisanie operatorów i arytmetyka kerneli mogą pozostać UNVERIFIED w telemetrii runtime.',
    'Wynik czasowy nie jest oceną jakości. GT nie jest wymagane do automatycznej analizy czasu.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def material_descriptor(descriptor):
    """Remove only enumerated technical fields, retaining all actual geometry."""
    value = deepcopy(descriptor)
    for part in (value, value.get('calibration_snapshot', {})):
        for key in ('digest_sha256', 'calibration_digest_sha256', 'frozen_at_utc', 'confirmed_at_utc'):
            part.pop(key, None)
        geometry = part.get('source_geometry', {})
        for key in ('binding_generation', 'viewport_revision'):
            geometry.pop(key, None)
    return value


def _number(value):
    if value in (None, '') or isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _count(value):
    value = _number(value)
    return int(value) if value is not None and value.is_integer() else None


def source_metadata(source):
    from ..ranking.mobile_package_experiments import MobileSessionImportItem, _observe_session_import
    bundle = source.bundle
    coll, raw = bundle.collection_session, bundle.report_payload
    execution = coll.get('execution') or {}
    item = MobileSessionImportItem(path=bundle.path)
    _observe_session_import(bundle, item)
    blocks = item.observations['block']
    source.metadata['block'] = blocks[0] if len(blocks) == 1 else None
    source.metadata['block_evidence'] = list(blocks)
    rep = coll.get('replicate_index')
    source.metadata['repeat'] = f'r{rep:02d}' if type(rep) is int and rep > 0 else '—'
    tv = getattr(bundle, 'tv_bbox_evidence', {}) or {}
    stages = execution.get('stages') or {}
    flags = execution.get('feature_flags') or {}
    variant = coll.get('variant')
    duration = raw.get('session_duration_ms')
    notes = str((raw.get('experiment') or {}).get('notes') or '')
    declared_plan = bool(re.search(r'10\s*(?:zdjęć\s*)?[×x]\s*30\s*s', notes, re.I))
    reasons = []
    if source.metadata.get('identity_status') != 'VERIFIED':
        reasons.append('F5A_IDENTITY_UNPROVEN')
    if not tv.get('available') or tv.get('status') != 'VALID':
        reasons.append('F5A_TV_INPUT_UNPROVEN')
    if coll.get('state') != 'COMPLETED' or coll.get('collection_complete') is not True:
        reasons.append('SESSION_NOT_COMPLETED')
    if any(coll.get(k) != 0 for k in ('dropped_sample_count', 'dropped_telemetry_count', 'integrity_loss_count')):
        reasons.append('SESSION_COMPLETENESS_UNPROVEN')
    retention = raw.get('data_retention') or {}
    if source.metadata.get('trace_state') != 'complete' or retention.get('complete') is not True or retention.get('trace_records_evicted') != 0:
        reasons.append('INCOMPLETE_TRACE_SOURCE')
    mp = stages.get('mp') or {}
    if variant not in ('R0', 'R1', 'R2') or mp.get('enabled') is not (variant != 'R0'):
        reasons.append('F5A_MP_POLICY_CONFLICT')
    if flags.get('vehicle_tracking') is not (variant != 'R0'):
        reasons.append('F5A_TRACKING_POLICY_UNPROVEN')
    if not source.metadata.get('apk_sha256') or not duration:
        reasons.append('F5A_ENVIRONMENT_UNPROVEN')
    if any(not stages.get(role, {}).get('model_id') for role in ('mt', 'mz')):
        reasons.append('F5A_MODELS_UNPROVEN')
    device = raw.get('device') or {}
    runtime = raw.get('runtime_execution_evidence') or {}
    runtime_stages = runtime.get('stages') or {}
    runtime_fields = ('runtime', 'runtime_version', 'backend_effective', 'backend_effective_scope',
        'cpu_threads_requested', 'cpu_threads_option_accepted', 'cpu_threads_effective',
        'delegate_effective', 'xnnpack_effective', 'operator_assignment_status', 'inputs', 'outputs',
        'model_sha256', 'model_sha_matches_manifest', 'artifact_sha256_actual', 'public_io_precision_status',
        'static_graph_classification', 'kernel_precision_provider')
    observed = {role: {k: info.get(k) for k in runtime_fields} for role, info in runtime_stages.items()}
    if runtime.get('error_event_count', 0) or runtime.get('malformed_records', 0):
        reasons.append('RUNTIME_EXECUTION_ERRORS')
    if any(info.get('model_sha_matches_manifest') is False for info in runtime_stages.values()):
        reasons.append('RUNTIME_MODEL_SHA_MISMATCH')
    common_execution = {k: deepcopy(v) for k, v in execution.items()
                        if k not in ('variant', 'roi_budget_policy', 'stages', 'tv_bbox_crop', 'feature_flags', 'package_size_bytes')}
    common_flags = {k: v for k, v in flags.items() if k != 'vehicle_tracking'}
    material = dict(input_mode=coll.get('input_mode'), descriptor=material_descriptor(tv.get('descriptor') or {}),
        execution=common_execution, feature_flags=common_flags,
        mt=stages.get('mt'), mz=stages.get('mz'),
        runtime_observed={k: observed.get(k) for k in ('plate', 'character')},
        model_refs={k: v for k, v in (coll.get('model_refs') or {}).items() if k in ('plate', 'character')},
        apk_sha256=source.metadata.get('apk_sha256'), duration_ms=duration,
        device={k: device.get(k) for k in ('manufacturer', 'model', 'device', 'android_version', 'android_sdk', 'abis', 'cpu_cores')})
    source.metadata['f5a_performance'] = dict(algorithm_id=ALGORITHM_ID,
        status='PASS' if not reasons else 'NOT_QUALIFIED', missing_reasons=reasons,
        material=material, material_sha256=digest(material),
        active_mp=dict(configured=mp, observed=observed.get('vehicle')) if mp.get('enabled') else None,
        runtime_evidence=runtime,
        technical=dict(descriptor=tv.get('descriptor'), frozen_at_utc=coll.get('tv_bbox_crop_frozen_at_utc'), app_build=coll.get('app_build')),
        environment=device, completion={k: coll.get(k) for k in ('state', 'completion_reason', 'collection_complete',
            'dropped_sample_count', 'dropped_telemetry_count', 'mt_detection_roles_complete')},
        planned_exposures=10 if declared_plan else None, planned_exposure_seconds=30 if declared_plan else None,
        planned_exposure_evidence=notes, mapped_exposures=None,
        scene_events=len((getattr(bundle, 'scene_evidence', {}) or {}).get('scenes') or []))


def project_timings(source, rows, trace_state):
    from .mobile_analysis import _metric, _missing
    from .mobile_import_comparison import _percentile
    source_metadata(source)
    diag = source.metadata['f5a_performance']
    diag['trace_status_counts'] = dict(Counter(str(r.get('status') or 'UNKNOWN') for r in rows))
    diag['trace_metrics'] = {}
    complete = trace_state == 'complete'
    retained = (source.bundle.report_payload.get('data_retention') or {})
    if retained and (retained.get('complete') is not True or retained.get('trace_records_evicted') != 0):
        complete = False
    valid = [r for r in rows if str(r.get('status') or '').lower() in COMPLETED_STATUSES]
    for prefix, field in (('mp_trace', 'vehicle_inference_ms'), ('mt_trace', 'plate_inference_ms'),
                          ('mz_trace', 'character_inference_ms'), ('pipeline', 'total_ms')):
        measured = []
        for row in valid:
            value = _number(row.get(field))
            if value is None or value <= 0:
                continue
            if prefix == 'pipeline' and not (_number(row.get('plate_inference_ms')) or 0) > 0:
                continue
            measured.append((row.get('frame_id'), value))
        values = [v for _, v in measured]
        diag['trace_metrics'][prefix] = dict(source_column=field, frame_ids=[f for f, _ in measured],
            values=values, n=len(values), n_all_traces=len(rows),
            accepted_statuses=sorted(COMPLETED_STATUSES), excluded_statuses=sorted(EXCLUDED_STATUSES),
            trace_class='ALPR_MT_TRACE' if prefix == 'pipeline' else prefix)
        for pct, q in (('p50', .5), ('p95', .95)):
            key = f'{prefix}_{pct}_ms'
            if not complete:
                _missing(source, key, 'INCOMPLETE_TRACE_SOURCE')
            elif prefix == 'mp_trace' and (source.bundle.collection_session.get('execution', {}).get('stages', {}).get('mp') or {}).get('enabled') is False:
                _missing(source, key, 'MP_DISABLED', n=0, total=0, explanation='R0: MP wyłączony; brak pomiaru nie jest czasem 0 ms.')
            else:
                _metric(source, key, _percentile(values, q), len(values), len(rows),
                    reasons=('STAGE_NOT_RUN_INVALID_OR_CANCELLED_TRACE',) if len(values) < len(rows) else (),
                    explanation=f'{field}; percentyl w sesji, interpolacja liniowa. n = ślady z pomiarem, ogółem = wszystkie zapisane ślady. '
                                + ('ALPR_MT_TRACE; bez pełnej latencji kamera–ekran.' if prefix == 'pipeline' else 'Suma inferencji etapu w śladzie, bez preprocessingu.'))
    timed_mt = [r for r in valid if (_number(r.get('plate_inference_ms')) or 0) > 0]
    for pct in ('p50', 'p95'):
        key = f'mt_{pct}_ms'
        if not timed_mt or any(_count(r.get('mt_runs_this_frame')) != 1 for r in timed_mt):
            _missing(source, key, 'MT_CALL_TIMING_UNPROVEN',
                explanation='Brak dowodu dokładnie jednego MT w każdym mierzonym śladzie; suma MT/ślad pozostaje osobną miarą.')
        else:
            from dataclasses import replace
            source.metrics[key] = replace(source.metrics[f'mt_trace_{pct}_ms'], metric_id=key,
                explanation='Każdy mierzony ślad potwierdza dokładnie jedno MT; czas etapu jest czasem pojedynczej inferencji.')
    for prefix, field in (('mp', 'vehicle_runs'), ('mt', 'mt_runs_this_frame'), ('mz', 'mz_runs')):
        counts = [(r.get('frame_id'), _count(r.get(field)), str(r.get('status') or '')) for r in rows]
        known = [(f, v, st) for f, v, st in counts if v is not None]
        diag['trace_metrics'][prefix + '_calls'] = dict(source_column=field,
            frame_ids=[f for f, _, _ in known], values=[v for _, v, _ in known],
            unknown_frame_ids=[f for f, v, _ in counts if v is None],
            calls_in_cancelled_traces=sum(v for _, v, st in known if st.lower() in EXCLUDED_STATUSES))
        if not complete or not known:
            _missing(source, prefix + '_calls', 'INCOMPLETE_TRACE_SOURCE' if not complete else 'CALL_COUNT_UNAVAILABLE', n=len(known), total=len(rows))
        else:
            _metric(source, prefix + '_calls', sum(v for _, v, _ in known), len(known), len(rows),
                reasons=('CALL_COUNTER_MISSING_PARTIAL_SUM',) if len(known) != len(rows) else (),
                explanation=f'Suma jawnych {field}, także ze śladów anulowanych; n = ślady z licznikiem. Puste pole pozostaje nieznane, nie jest zerem.')
    for key in ('mt_calls_per_exposure', 'mz_calls_per_exposure'):
        _missing(source, key, 'EXPOSURE_MAPPING_UNPROVEN', explanation='Brak źródłowej mapy ślad → zdjęcie TV / ekspozycja 30 s. Nie dzielimy przez planowane 120.')


def _flatten(value, path=''):
    if isinstance(value, dict):
        return {p: v for k, item in value.items() for p, v in _flatten(item, (path + '.' + k).strip('.')).items()}
    return {path: value}


def qualify(sessions, duplicates=()):
    """Report differences without deleting measured per-session observations."""
    eligible = [s for s in sessions if s.source_status == 'VERIFIED' and s.metadata.get('experiment') == 'F5A']
    diagnostics = [s.metadata.get('f5a_performance') or {} for s in eligible]
    reasons = []
    if len(eligible) != len(sessions) or not sessions:
        reasons.append('F5A_SOURCE_NOT_PROVEN')
    if duplicates:
        reasons.append('DUPLICATE_SOURCE_NOT_INDEPENDENT')
    if any(d.get('status') != 'PASS' for d in diagnostics):
        reasons.append('F5A_SESSION_NOT_QUALIFIED')
    if len({(s.metadata.get('series_id'), s.metadata.get('is_pilot')) for s in eligible}) > 1:
        reasons.append('F5A_SERIES_OR_PILOT_MISMATCH')
    flat = [_flatten(d.get('material') or {}) for d in diagnostics]
    differences = {}
    for key in sorted(set().union(*(f.keys() for f in flat))):
        groups = {}
        for s, f in zip(eligible, flat):
            encoded = json.dumps(f.get(key), sort_keys=True, ensure_ascii=False)
            groups.setdefault(encoded, []).append(s.input.session_id)
        if len(groups) > 1:
            differences[key] = [dict(value=json.loads(k), session_ids=v) for k, v in groups.items()]
    if differences:
        reasons.append('F5A_MATERIAL_CONDITIONS_MISMATCH')
    if len({digest(d['active_mp']) for d in diagnostics if d.get('active_mp')}) > 1:
        reasons.append('F5A_ACTIVE_MP_MISMATCH')
    combinations = [(s.metadata.get('variant'), s.metadata.get('repeat'), s.metadata.get('block')) for s in eligible]
    expected = {(v, f'r{r:02d}', b) for v in ('R0', 'R1', 'R2') for r in (1, 2, 3) for b in 'ABCD'}
    balanced = len(combinations) == 36 and set(combinations) == expected and not duplicates
    return dict(algorithm_id=ALGORITHM_ID, status='PASS_DESCRIPTIVE' if not reasons else 'NOT_COMPARABLE',
        missing_reasons=reasons, material_differences=differences,
        material_groups={d.get('material_sha256'): [s.input.session_id for s in eligible
            if s.metadata.get('f5a_performance', {}).get('material_sha256') == d.get('material_sha256')] for d in diagnostics},
        balanced_36_sessions=balanced, missing_combinations=sorted(expected - set(combinations)),
        planned_exposures_per_variant={v: sum(d.get('planned_exposures') or 0 for s, d in zip(eligible, diagnostics)
            if s.metadata.get('variant') == v) if any(s.metadata.get('variant') == v for s in eligible) and all(d.get('planned_exposures') is not None for s, d in zip(eligible, diagnostics)
            if s.metadata.get('variant') == v) else None for v in ('R0', 'R1', 'R2')},
        mapped_exposures_per_variant=None,
        environment_by_session={s.input.session_id: d.get('environment') for s, d in zip(eligible, diagnostics)},
        limitations=LIMITATIONS)


def aggregate(cohort, metric_id, scenario):
    """Equal-session timing summary; counts are explicit-counter sums."""
    from .mobile_analysis import metric_definition
    result = []
    for variant in ('R0', 'R1', 'R2'):
        sessions = [s for s in cohort.sessions if s.metadata.get('variant') == variant and s.metadata.get('scenario') == scenario]
        metrics = [(s, s.metrics.get(metric_id)) for s in sessions]
        used = [(s, m) for s, m in metrics if m is not None and m.value is not None]
        reasons = list(dict.fromkeys(r for _, m in metrics if m for r in m.missing_reasons))
        gate = cohort.performance.get('status') == 'PASS_DESCRIPTIVE'
        if not gate:
            reasons += cohort.performance.get('missing_reasons', ['F5A_SESSION_NOT_QUALIFIED'])
        is_count = metric_id in ('mp_calls', 'mt_calls', 'mz_calls')
        value = (sum(m.value for _, m in used) if is_count else median(m.value for _, m in used)) if used and gate else None
        result.append(dict(variant=variant, metric_id=metric_id, value=value, unit=metric_definition(metric_id)[1],
            status='N/A' if value is None else 'SUBSET' if len(used) != len(sessions) or any(m.status == 'SUBSET' for _, m in used) else 'FULL',
            aggregation='sum_of_explicit_session_counters' if is_count else 'median_of_session_' + ('p95' if 'p95' in metric_id else 'p50'),
            n_sessions=len(used), n_sessions_selected=len(sessions),
            n_source_observations=sum(m.n_scored or 0 for _, m in used),
            n_source_observations_total=sum(m.n_total_eligible or 0 for _, m in metrics if m),
            missing_reasons=list(dict.fromkeys(reasons)),
            source_session_ids=[s.input.session_id for s in sessions],
            source_sha256=[s.input.source_sha256 for s in sessions],
            session_values=[dict(session_id=s.input.session_id, value=m.value, n=m.n_scored, total=m.n_total_eligible) for s, m in metrics if m]))
    return result
