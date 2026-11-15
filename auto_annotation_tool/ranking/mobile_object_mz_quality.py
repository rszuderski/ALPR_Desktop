"""Conditional OCR for F4-EVAL-2.0; source calls/timing are independent of rows."""
from __future__ import annotations
import copy
from collections import Counter

from .mobile_human_review import align_plate_text
from ..registration_text import normalize_registration

SCHEMA = 'alpr.mobile.mz_quality_context.v2'


def score_readout(row):
    """Keep executed false positives, wrong text and empty executed predictions."""
    out = copy.deepcopy(row)
    out.update(quality_state='EXECUTION_UNKNOWN', quality_reason='MZ_PARENT_UNPROVEN',
        canonical_ocr_eligible=False, exact_match=None, edit_distance=None,
        gt_characters=None, cer=None)
    if row.get('executed') is False:
        out.update(quality_state='NOT_RUN', quality_reason='NOT_RUN')
    elif row.get('executed') is True:
        out['quality_state'] = 'EXECUTED_UNASSESSED'
        if row.get('parent_proven') is not True:
            return out
        if row.get('source_mt_role') == 'SUPPRESSED':
            out.update(quality_state='EXECUTED_EXCLUDED_RAW', quality_reason='MZ_EXCLUDED_RAW_SUPPRESSED')
        elif row.get('mt_outcome') == 'FP' or row.get('operator_decision') == 'NOT_PLATE':
            out.update(quality_state='EXECUTED_EXCLUDED_MT_FP', quality_reason='EXECUTED_EXCLUDED_MT_FP')
        elif row.get('mt_outcome') != 'TP':
            out['quality_reason'] = 'MT_UNIT_NOT_QUALIFIED'
        elif row.get('crop_parent_proven') is not True:
            out['quality_reason'] = 'MZ_PARENT_UNPROVEN'
        elif row.get('readout_unambiguous') is not True:
            out['quality_reason'] = 'AMBIGUOUS_SOURCE_READOUTS'
        elif row.get('gtz_confirmed') is not True or not normalize_registration(row.get('gtz') or ''):
            out['quality_reason'] = 'GTZ_UNAVAILABLE'
        else:
            aligned = align_plate_text(row['gtz'], row.get('raw_prediction') or '')
            out.update(quality_state='EXECUTED_ELIGIBLE', quality_reason='CONFIRMED_TP_CROP_GTZ',
                canonical_ocr_eligible=True, exact_match=aligned.exact_match,
                edit_distance=aligned.edit_distance, gt_characters=len(aligned.ground_truth), cer=aligned.cer)
    return out


def _execution_counters(session):
    """Full source trace counters; never infer calls from exported readout rows."""
    from .mobile_package_experiments import iter_full_trace_rows
    result = dict(n_mz_calls=None, n_mt_calls=None, n_traces=None,
        reason='CALL_COUNT_UNAVAILABLE', observation_unit='recorded_call_counter_per_trace',
        mz_timing_unit='mz_stage_sum_per_trace', mz_single_call_p95_ms=None,
        mz_single_call_timing_reason='PER_CALL_TIMING_UNAVAILABLE')
    try:
        preview = list(session.bundle.trace_rows or ())
        expected = int(session.bundle.trace_total or len(preview))
        traces = preview if expected <= len(preview) else list(iter_full_trace_rows(session.bundle))
        if not traces or len(traces) != expected:
            return result
        valid = [r for r in traces if str(r.get('status', '')).lower() not in
            ('scene_superseded', 'runtime_error', 'pipeline_error', 'cancelled', 'not_run')]
        def count(column):
            values = [int(r[column]) for r in valid]
            if any(v < 0 for v in values):
                raise ValueError('Negative call count')
            return sum(values)
        result['n_traces'] = len(traces)
        for key, column in (('n_mz_calls', 'mz_runs'), ('n_mt_calls', 'mt_runs_this_frame')):
            try:
                result[key] = count(column)
            except (KeyError, ValueError, TypeError):
                result[key] = None
        result['reason'] = None if result['n_mz_calls'] is not None else 'MZ_CALL_COUNTER_INCOMPLETE'
        result['n_known_mz_calls'] = sum(int(r['mz_runs']) for r in valid if str(r.get('mz_runs', '')).isdigit())
        result['n_unknown_mz_counter_traces'] = sum(not str(r.get('mz_runs', '')).isdigit() for r in valid)
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return result


def snapshot(session, evaluation):
    """Read exact attempt/crop lineage; no GT propagation across independent MT."""
    from . import mobile_canvas_review as core, mobile_candidate_stacks as stacks
    candidates = {}
    for gid in session.mt_invocations:
        try:
            scene = core._legacy_scene(session, gid, include_history=False)
            state = stacks.snapshot(session, gid, scene)
        except (ValueError, KeyError, TypeError):
            continue
        for c in state['candidates']:
            candidates.setdefault(c['attempt_id'], []).append((gid, c))
    outcomes = {o['attempt_id']: o for u in evaluation['units'] if u['status'] == 'MEASURED'
                for o in u['final_outcomes']}
    pairs = {p['attempt_id']: p for u in evaluation['units'] if u['status'] == 'MEASURED'
             for p in u['matched_pairs']}
    gt = {(u['mt_invocation_id'], p['gt_id']): p for u in evaluation['units'] for p in u['physical_gt']}
    sources = list(session.samples.items())
    represented = {s.get('attempt_id') for _, s in sources}
    sources.extend((None, dict(attempt_id=aid)) for aid in session.attempts if aid not in represented)
    rows = []
    for sid, sample in sources:
        aid = sample.get('attempt_id'); attempt = session.attempts.get(aid, {})
        found = candidates.get(aid, [])
        gid, candidate = found[0] if len(found) == 1 else (None, {})
        proven = bool(attempt and len(found) == 1 and attempt.get('mt_invocation_id') == gid
            and str(attempt.get('session_id')) == session.session_id)
        for key in ('session_id', 'scene_generation', 'mt_invocation_id'):
            if sample.get(key) and str(sample[key]) != str(attempt.get(key, '')):
                proven = False
        flag = str(attempt.get('mz_executed', '')).lower()
        executed = {'true': True, '1': True, 'false': False, '0': False}.get(flag)
        if executed is None and attempt.get('mz_status') == 'NOT_RUN':
            executed = False
        if executed is None and (sample.get('fresh_mz_successful') in (True, 'true')
                or sample.get('timing', {}).get('character_inference_ms') is not None):
            executed = True
        crop = sample.get('image_entry') or sample.get('evidence_entry') or attempt.get('evidence_entry')
        crop_proven = bool(proven and crop and crop in session.entry_names
            and attempt.get('evidence_kind') == 'plate_crop' and crop == attempt.get('evidence_entry'))
        execution_id = sample.get('mz_execution_id') or attempt.get('mz_execution_id')
        siblings = [session.samples[k] for k in session.sample_ids_by_attempt.get(aid, [])]
        unambiguous = len(siblings) <= 1 or bool(execution_id and all(r.get('mz_execution_id') for r in siblings)
            and len({r['mz_execution_id'] for r in siblings}) == len(siblings))
        outcome = outcomes.get(aid, {}); pair = pairs.get(aid, {})
        plate = gt.get((gid, outcome.get('gt_id')), {})
        # A crop at a disjoint source Pose is not proved to contain this GTR,
        # even if the object-level operator decision is positive.
        if pair.get('polygon_iou') == 0:
            crop_proven = False
        raw = (sample.get('raw_prediction', sample.get('fresh_prediction', sample.get('prediction', '')))
            if sid is not None else attempt.get('raw_prediction', attempt.get('prediction', '')))
        row = dict(session_id=session.session_id, source_archive_sha256=session.review.source_archive_sha256,
            review_revision=session.review.review_revision, source_sample_id=sid, attempt_id=aid,
            mt_invocation_id=gid, mt_detection_index=candidate.get('mt_detection_index'),
            source_mt_role=candidate.get('role', 'UNKNOWN'), parent_proven=proven,
            crop_parent_proven=crop_proven, readout_unambiguous=unambiguous,
            source_execution_id=execution_id, executed=executed,
            operator_decision=candidate.get('operator_decision'), mt_outcome=outcome.get('outcome'),
            mt_outcome_reason=outcome.get('reason'), gt_id=outcome.get('gt_id'),
            gtz=plate.get('gt_text'), gtz_confirmed=bool(plate.get('text_confirmed')
                and not plate.get('text_exclusion_reason')), raw_prediction=raw,
            source_mz_status=attempt.get('mz_status'), crop_entry=crop,
            crop_origin='source_attempt_plate_crop', timing=copy.deepcopy(sample.get('timing', {})),
            source_mz_ms=sample.get('mz_ms'), source_pipeline_ms=sample.get('pipeline_ms'))
        rows.append(score_readout(row))
    return dict(schema=SCHEMA, session_id=session.session_id,
        source_archive_sha256=session.review.source_archive_sha256, review_revision=session.review.review_revision,
        matching_policy_id=evaluation['matching_policy_id'], matching_policy_sha256=evaluation['matching_policy_sha256'],
        rows=rows, **summarize_readouts(rows), execution_counters=_execution_counters(session),
        counting_note='Readout rows and conditional OCR denominators are not inference-call counts. Full source trace counters include FP MT; stage timings are unchanged.')


def summarize_readouts(rows):
    """Pure sums for the live result and independent export replay."""
    eligible = [r for r in rows if r['canonical_ocr_eligible']]
    errors = sum(r['edit_distance'] for r in eligible)
    chars = sum(r['gt_characters'] for r in eligible)
    executed_true = [r for r in rows if r['executed'] is not False]
    # Excluded FP/RAW and NOT_RUN do not enlarge the conditional OCR target.
    target = [r for r in executed_true if r['quality_state'] not in ('EXECUTED_EXCLUDED_MT_FP', 'EXECUTED_EXCLUDED_RAW')]
    return dict(n_source_readouts=len(rows), n_reads=len(eligible), n_exact=sum(r['exact_match'] for r in eligible),
        n_text_errors=errors, n_gt_characters=chars, micro_cer=errors/chars if chars else None,
        exact_rate=sum(r['exact_match'] for r in eligible)/len(eligible) if eligible else None,
        n_target=len(target), n_unscored=len(target)-len(eligible),
        status='FULL' if eligible and len(target) == len(eligible) else 'SUBSET' if eligible else 'N/A',
        state_counts=dict(Counter(r['quality_state'] for r in rows)),
        unavailable_reasons=sorted({r['quality_reason'] for r in target if not r['canonical_ocr_eligible']}))
