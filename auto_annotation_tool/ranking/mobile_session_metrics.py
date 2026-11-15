"""Read-only F4-EVAL-2.0 view model, with independent metric denominators."""
from __future__ import annotations

import copy
import math
from .mobile_mt_object_evaluation import POLICY_ID, POLICY_SHA256, format_rate
from .mobile_frame_review_state import ELIGIBILITY_SHA256

SCHEMA = 'alpr.mobile.live_session_metrics.v2'
KEY_FIELDS = ('session_id', 'source_archive_sha256', 'review_revision', 'matching_policy_id','completion_eligibility_sha256')


def result_key(session):
    if session is None:
        return None
    return (session.session_id, session.review.source_archive_sha256,
            session.review.review_revision, POLICY_ID,ELIGIBILITY_SHA256)


def snapshot_key(snapshot):
    return tuple((snapshot or {}).get(k) for k in KEY_FIELDS)


def current(snapshot, key):
    return bool(snapshot and key and snapshot_key(snapshot) == key
                and snapshot.get('matching_policy_sha256') == POLICY_SHA256
                and snapshot.get('status') != 'STALE_REVIEW')


def canvas_metric_projection(canvas, mt):
    """Present the same v2 sums in existing captions without mutating v1 data."""
    canvas, mt = canvas or {}, mt or {}
    if (mt.get('matching_policy_id') != POLICY_ID or mt.get('matching_policy_sha256') != POLICY_SHA256
            or mt.get('status') == 'STALE_REVIEW'
            or any(mt.get(k) != canvas.get(k) for k in KEY_FIELDS[:3])):
        return canvas
    from ..registration_text import normalize_registration
    geo, ocr = mt['geometry_summary'], mt['ocr_summary']
    return dict(canvas, geometry_n=geo['n_geometry'], polygon_iou_mean_pct=geo['polygon_iou_mean_pct'],
        nmce_mean_pct=geo['nmce_mean_pct'], evaluated_reads=ocr['n_reads'], exact_reads=ocr['n_exact'],
        gt_characters=ocr['n_gt_characters'], text_errors=ocr['n_text_errors'],
        conditional_empty_reads=sum(r['canonical_ocr_eligible'] and not normalize_registration(r.get('raw_prediction') or '')
            for r in ocr['rows']))


def from_results(session, canvas, mt):
    """Project already-computed sources, retaining their exact denominators."""
    key = result_key(session)
    identity = dict(zip(KEY_FIELDS, key))
    coherent = (all(canvas.get(k) == identity[k] == mt.get(k) for k in KEY_FIELDS[:3])
                and (mt.get('completion_eligibility') or {}).get('sha256')==ELIGIBILITY_SHA256
                and mt.get('matching_policy_id') == POLICY_ID
                and mt.get('matching_policy_sha256') == POLICY_SHA256
                and mt.get('status') != 'STALE_REVIEW')
    geometry = mt.get('geometry_summary', {}) if coherent else {}
    ocr = mt.get('ocr_summary', {}) if coherent else {}
    geometry_n = geometry.get('n_geometry', 0)
    reads = ocr.get('n_reads', 0)
    chars = ocr.get('n_gt_characters', 0)
    cer = ocr.get('micro_cer')
    complete = (coherent and session.review.review_status == 'COMPLETED'
                and mt.get('status') == 'COMPLETE' and geometry.get('status') == 'FULL'
                and ocr.get('status') == 'FULL')
    result = dict(schema=SCHEMA, **identity, matching_policy_sha256=POLICY_SHA256,
        completion_eligibility=copy.deepcopy(mt.get('completion_eligibility')) if coherent else None,
        gt_mode='ASSISTED', status='COMPLETE' if complete else 'PARTIAL' if coherent else 'STALE_REVIEW',
        methodology_version='F4-EVAL-2.0', adopted_date='2026-11-01', chronology='RETROSPECTIVE_NOT_PREREGISTERED',
        sources=dict(mt='mt_object_evaluation_v2', geometry='mt_object_evaluation_v2', ocr='mz_quality_context_v2'),
        mt={k: copy.deepcopy(mt.get(k)) if coherent else None for k in
            ('tp','fp','fn','n_gt','n_final_mt','precision','recall','f1')},
        geometry=dict(n_geometry=geometry_n,
            status=geometry.get('status', 'N/A'),
            polygon_iou_mean_pct=geometry.get('polygon_iou_mean_pct') if geometry_n else None,
            nmce_mean_pct=geometry.get('nmce_mean_pct') if geometry_n else None,
            unavailable_reasons=[] if geometry_n else ['NO_CANONICAL_GEOMETRY_PAIRS']),
        ocr=dict(micro_cer=cer if reads else None, n_reads=reads, n_gt_characters=chars,
            status=ocr.get('status', 'N/A'),
            n_text_errors=ocr.get('n_text_errors', 0),
            n_exact=ocr.get('n_exact') if reads and chars else None,
            unavailable_reasons=[] if reads and chars else ['NO_CANONICAL_TEXT_READS']),
        unavailable_reasons=[] if coherent else ['SOURCE_OR_REVISION_MISMATCH'])
    result['mt'].update(n_qualified=mt.get('n_qualified',0) if coherent else 0,
        n_total=mt.get('n_total',0) if coherent else 0,
        unavailable_reasons=copy.deepcopy(mt.get('unavailable_reasons',[])) if coherent else ['SOURCE_OR_REVISION_MISMATCH'])
    from .mobile_experiment_identity import from_bundle
    if from_bundle(getattr(session, 'bundle', None)).get('family') == 'F5A':
        result['experiment_family'] = 'F5A'
    return result


def percent(value, places=2):
    """Percentage points, including valid CER/NMCE values above 100%."""
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
        return '—'
    return f'{value:.{places}f}%'.replace('.',',')


def presentation(snapshot, key):
    """Small shared formatter; never computes metrics from widget text."""
    if not current(snapshot,key):
        return dict(values={k:'—' for k in ('precision','recall','f1','iou','nmce','cer')},
            status='Aktualizowanie zapisanych danych…', exact='Exact MZ: —/—',
            denominators='MT: — · geometria: — · MZ: —', refreshing=True)
    mt,geo,ocr=snapshot['mt'],snapshot['geometry'],snapshot['ocr']
    values={k:format_rate(mt[k]) for k in ('precision','recall','f1')}
    values.update(iou=percent(geo['polygon_iou_mean_pct'],1),nmce=percent(geo['nmce_mean_pct']),
                  cer=percent(100*ocr['micro_cer']) if ocr['micro_cer'] is not None else '—')
    exact=(f"{ocr['n_exact']}/{ocr['n_reads']}" if ocr['n_reads'] and ocr['n_gt_characters'] else '—/—')
    geometry=(f"{geo['n_geometry']} par" if geo['n_geometry'] else '0 (brak par)')
    text=(f"{ocr['n_reads']} odczytów / {ocr['n_gt_characters']} znaków GT"
          if ocr['n_reads'] and ocr['n_gt_characters'] else '0 (brak ocenionych odczytów)')
    return dict(values=values, status=('pełny zakres' if snapshot['status']=='COMPLETE'
                else 'tylko ocenione dane'), exact='Exact MZ: '+exact,
        assessment_scope=('ASSISTED_SUBSET' if 0 < mt['n_qualified'] < mt['n_total']
                          else 'ASSISTED' if mt['n_qualified'] else 'N/A'),
        denominators=f"MT: {mt['n_qualified']}/{mt['n_total']} · geometria: {geometry}\nMZ: {text}", refreshing=False)
