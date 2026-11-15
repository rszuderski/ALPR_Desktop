"""F4-EVAL-2.0: source-bound, operator-assisted FINAL object evaluation.

Retrospective policy adopted on 2026-11-01. BBOX/Pose are representations of
one FINAL, never independent detections. Reading this module cannot save GT or
decisions. The archived IoU50 BBOX scorer remains a separate, unchanged v1.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from weakref import WeakKeyDictionary

from .mobile_mt_detection_evaluation import rates, format_rate
from .mobile_pose_metrics import validate_mt_quad, compare_mt_pose_to_gt
from .mt_eval_metrics import _bbox, box_iou

SCHEMA = 'alpr.mobile.mt_object_evaluation.v2'
POLICY_ID = 'f4_mt_final_operator_assisted_object_v2'
POLICY = dict(id=POLICY_ID, version='F4-EVAL-2.0', adopted_date='2026-11-01',
    chronology='RETROSPECTIVE_NOT_PREREGISTERED', gt_mode='ASSISTED',
    population='source_proven_FINAL_no_refilter', unit='mt_invocation_id+source_frame+ROI',
    positive='operator_confirmed_PLATE_with_one_saved_confirmed_physical_GTR',
    negative='operator_confirmed_NOT_PLATE',
    matching='explicit_relation_confidence_desc_index_asc_attempt_id_asc_1to1',
    duplicate='DUPLICATE_FINAL_is_FP', incomplete='whole_unit_REVIEW_REQUIRED_no_partial_TP_FN',
    geometry='secondary_continuous_source_BBOX_IoU_and_source_Pose_GTR_Polygon_IoU_NMCE',
    bbox_threshold_for_primary=None, spatial_overlap_is_operator_evidence=False,
    ocr='conditional_on_TP_proven_crop_executed_MZ_confirmed_GTZ_including_empty_predictions',
    false_mt_ocr='EXECUTED_EXCLUDED_MT_FP', primary_tn=False,
    e2e='N/A_without_planned_scene_window_contract',
    document='docs/experiments/F4_EVAL_2_0_OPERATOR_ASSISTED_OBJECT_20261101.md')
_POLICY_JSON = json.dumps(POLICY, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
POLICY_SHA256 = hashlib.sha256(_POLICY_JSON.encode('utf-8')).hexdigest()
_cache, _frames = WeakKeyDictionary(), WeakKeyDictionary()
_lock = threading.RLock()
IDENTITY_FIELDS = ('session_id', 'source_archive_sha256', 'review_revision', 'unit_id',
    'source_frame_id', 'mt_invocation_id', 'source_binding', 'image_entry', 'image_size', 'roi')


def _approved(policy):
    try:
        return json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(',', ':')) == _POLICY_JSON
    except (TypeError, ValueError):
        return False


def _geometry(candidate, gt, size):
    """Continuous diagnostics; missing source geometry is never reconstructed."""
    box = _bbox(candidate.get('bbox')) if candidate.get('bbox_source_verified') is True else None
    if box and not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
        box = None
    result = dict(source_bbox=copy.deepcopy(candidate.get('bbox')),
        source_pose_quad=copy.deepcopy(candidate.get('pose_quad')), gt_quad=copy.deepcopy(gt['quad']),
        bbox_iou=box_iou(box, gt['bbox']) if box else None,
        bbox_reason=None if box else 'SOURCE_BBOX_UNAVAILABLE_OR_DEGENERATE',
        polygon_iou=None, polygon_iou_pct=None, nmce_pct=None, pose_reason=None,
        disagreement_codes=[])
    try:
        if candidate.get('pose_source_verified') is not True:
            raise ValueError('Unproven source Pose')
        result.update(compare_mt_pose_to_gt(candidate.get('pose_quad'), gt['quad'], *size))
    except (ValueError, TypeError, KeyError):
        result['pose_reason'] = 'SOURCE_POSE_UNAVAILABLE_OR_INVALID'
    if not box:
        result['disagreement_codes'].append('BBOX_UNAVAILABLE_OBJECT_DECISION_INDEPENDENT')
    elif result['bbox_iou'] < .5:
        result['disagreement_codes'].append('BBOX_IOU_BELOW_V1_THRESHOLD')
    if result['polygon_iou'] is not None and result['bbox_iou'] is not None:
        if (result['polygon_iou'] >= .5) != (result['bbox_iou'] >= .5):
            result['disagreement_codes'].append('BBOX_POSE_THRESHOLD_DISAGREEMENT')
    if (result['polygon_iou'] == 0 or result['bbox_iou'] == 0):
        result['disagreement_codes'].append('SOURCE_GEOMETRY_DISJOINT_FROM_CONFIRMED_GTR')
    return result


def score_unit_v2(unit, *, policy=POLICY):
    """Pure API. An explicit evidence-bearing relation is required for each TP."""
    result = {k: copy.deepcopy(unit.get(k)) for k in IDENTITY_FIELDS}
    result.update(schema=SCHEMA, matching_policy_id=POLICY_ID, matching_policy_sha256=POLICY_SHA256,
        methodology_version='F4-EVAL-2.0', adopted_date='2026-11-01',
        chronology='RETROSPECTIVE_NOT_PREREGISTERED', gt_mode='ASSISTED',
        status='NOT_EVALUABLE', n_qualified=0, n_total=1,
        n_gt=None, n_final_mt=None, tp=None, fp=None, fn=None,
        n_raw_suppressed=unit.get('n_raw_suppressed'), matched_pairs=[], final_outcomes=[],
        unmatched_gt_ids=[], excluded_gt_ids=[], geometry_diagnostics=[],
        unavailable_reasons=list(unit.get('unavailable_reasons', [])),
        review_tasks=copy.deepcopy(unit.get('review_tasks', [])), partial_fp_lower_bound=None,
        source_finals=copy.deepcopy(unit.get('finals', [])), physical_gt=copy.deepcopy(unit.get('gt', [])))
    result['evaluation_input'] = copy.deepcopy(unit)
    if 'completion_eligibility' in unit:
        result['completion_eligibility']=copy.deepcopy(unit['completion_eligibility'])
    if 'source_role_evidence' in unit:
        result['source_role_evidence']=copy.deepcopy(unit['source_role_evidence'])

    def unavailable(reason=None, *, review=False):
        if reason and reason not in result['unavailable_reasons']:
            result['unavailable_reasons'].append(reason)
        result['status'] = 'REVIEW_REQUIRED' if review else 'NOT_EVALUABLE'
        result.update(rates(None, None, None))
        return result

    if 'completion_eligibility' in unit:
        from .mobile_frame_review_state import ELIGIBILITY,ELIGIBILITY_SHA256
        contract=unit['completion_eligibility']
        if (not isinstance(contract,dict) or contract.get('sha256')!=ELIGIBILITY_SHA256
                or any(contract.get(k)!=v for k,v in ELIGIBILITY.items())):
            return unavailable('UNSUPPORTED_COMPLETION_ELIGIBILITY')
    if not _approved(policy):
        result.update(matching_policy_id=None, matching_policy_sha256=None)
        return unavailable('NO_APPROVED_OBJECT_POLICY')
    if unit.get('execution_status') != 'COMPLETED':
        return unavailable('MT_NOT_COMPLETED')
    if unit.get('source_verified') is not True:
        return unavailable('SOURCE_NOT_VERIFIED')
    try:
        width, height = unit['image_size']
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 or v != int(v) for v in (width, height)):
            return unavailable('INVALID_SOURCE_SIZE')
        width, height = int(width), int(height)
        roi = _bbox(unit['roi'])
        if roi is None or not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
            return unavailable('INVALID_SOURCE_ROI')
        if not all(result.get(k) for k in IDENTITY_FIELDS if k not in ('review_revision', 'image_size', 'roi')):
            return unavailable('SOURCE_IDENTITY_INCOMPLETE')
        if type(result['review_revision']) is not int or result['review_revision'] < 0:
            return unavailable('REVIEW_REVISION_UNAVAILABLE')
        if type(unit.get('n_raw_suppressed')) is not int or unit['n_raw_suppressed'] < 0:
            return unavailable('RAW_POPULATION_UNPROVEN')
        finals = unit['finals']
        ids, indices = set(), set()
        for c in finals:
            aid, index, confidence = c['attempt_id'], c['index'], c['confidence']
            if not isinstance(aid, str) or not aid or aid in ids:
                return unavailable('ATTEMPT_ID_CONFLICT')
            if type(index) is not int or index < 0 or index in indices:
                return unavailable('FINAL_INDEX_CONFLICT')
            if (type(confidence) not in (float, int) or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1):
                return unavailable('FINAL_CONFIDENCE_UNAVAILABLE')
            if c.get('role') != 'FINAL':
                return unavailable('FINAL_POPULATION_UNPROVEN')
            ids.add(aid); indices.add(index)
        # A lower bound is permitted only after the complete source population
        # and source identity above have been proved. It is never a formal TP/FN.
        lower = sum(c.get('decision') == 'NOT_PLATE' and c.get('operator_confirmed') is True
                    and bool(c.get('operator_review_evidence')) for c in finals)
        if lower:
            result['partial_fp_lower_bound'] = lower
        if result['unavailable_reasons']:
            return unavailable()
        if any(c.get('decision') not in ('PLATE', 'NOT_PLATE') for c in finals):
            return unavailable('FINAL_DECISION_UNRESOLVED', review=True)
        if any(c.get('operator_confirmed') is not True or not c.get('operator_review_evidence') for c in finals):
            return unavailable('OPERATOR_DECISION_UNPROVEN', review=True)
        plates, gt_ids = {}, set()
        for g in unit['gt']:
            pid = g['gt_id']
            if not isinstance(pid, str) or not pid or pid in gt_ids:
                return unavailable('GT_ID_CONFLICT')
            gt_ids.add(pid)
            if g.get('confirmed') is not True:
                return unavailable('GT_GEOMETRY_UNCONFIRMED', review=True)
            try:
                quad = validate_mt_quad(g.get('quad'), width, height)
            except ValueError:
                return unavailable('GT_SOURCE_GEOMETRY_INVALID')
            box = [min(p[0] for p in quad), min(p[1] for p in quad),
                   max(p[0] for p in quad), max(p[1] for p in quad)]
            if box[2] <= roi[0] or box[0] >= roi[2] or box[3] <= roi[1] or box[1] >= roi[3]:
                result['excluded_gt_ids'].append(pid)
                continue
            if not (roi[0] <= box[0] < box[2] <= roi[2] and roi[1] <= box[1] < box[3] <= roi[3]):
                return unavailable('GT_PARTIAL_ROI_COVERAGE')
            plates[pid] = dict(g, quad=quad, bbox=box)
        for c in finals:
            if c['decision'] == 'PLATE':
                if c.get('gt_id') not in plates or c.get('relation_confirmed') is not True:
                    return unavailable('MT_GTR_RELATION_UNPROVEN', review=True)
                evidence = c['operator_review_evidence']
                if (evidence.get('attempt_id') != c['attempt_id']
                        or evidence.get('gt_id') != c['gt_id']
                        or evidence.get('source_binding') != unit['source_binding']):
                    return unavailable('MT_GTR_RELATION_IDENTITY_CONFLICT')
        if unit.get('complete') is not True or unit.get('gt_complete') is not True:
            return unavailable('REVIEW_INCOMPLETE', review=True)
        occupied = set()
        for c in sorted(finals, key=lambda c: (-c['confidence'], c['index'], c['attempt_id'])):
            pid = c.get('gt_id') if c['decision'] == 'PLATE' else None
            out = dict(attempt_id=c['attempt_id'], object_id=c.get('object_id'), index=c['index'],
                gt_id=pid, operator_decision=c['decision'],
                operator_review_evidence=copy.deepcopy(c['operator_review_evidence']))
            if c['decision'] == 'NOT_PLATE':
                out.update(outcome='FP', reason='OPERATOR_NOT_PLATE')
            elif pid in occupied:
                out.update(outcome='FP', reason='DUPLICATE_FINAL')
            else:
                occupied.add(pid)
                out.update(outcome='TP', reason='OPERATOR_CONFIRMED_PHYSICAL_PLATE')
                diag = _geometry(c, plates[pid], (width, height))
                pair = dict(gt_id=pid, attempt_id=c['attempt_id'], object_id=c.get('object_id'),
                    mt_detection_index=c['index'], matched_by='OPERATOR_CONFIRMED_RELATION',
                    operator_review_evidence=copy.deepcopy(c['operator_review_evidence']), **diag)
                result['matched_pairs'].append(pair)
                result['geometry_diagnostics'].append(dict(attempt_id=c['attempt_id'], gt_id=pid, **diag))
            result['final_outcomes'].append(out)
        result['matched_pairs'].sort(key=lambda p: (p['mt_detection_index'], p['attempt_id']))
        result['final_outcomes'].sort(key=lambda p: (p['index'], p['attempt_id']))
        result['unmatched_gt_ids'] = sorted(set(plates) - occupied)
        tp, n_gt, n_final = len(occupied), len(plates), len(finals)
        result.update(n_gt=n_gt, n_final_mt=n_final, tp=tp, fp=n_final-tp, fn=n_gt-tp,
            n_qualified=1, status='MEASURED')
        result.update(rates(tp, n_final-tp, n_gt-tp, n_gt=n_gt, n_final_mt=n_final))
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return unavailable('INVALID_UNIT_INPUT: ' + str(error))


score_unit = score_unit_v2


def unit_from_session(session, gid):
    """Validate the persisted identity contract, never the spatial UI projection."""
    from . import mobile_canvas_review as core, mobile_candidate_stacks as stacks
    from . import mobile_canvas_partial_review as partial
    from .mobile_gt_link_contract import validate as validate_links
    sha = session.review.source_archive_sha256
    unit = dict(session_id=session.session_id, source_archive_sha256=sha,
        review_revision=session.review.review_revision, unit_id=sha+':'+gid, mt_invocation_id=gid,
        source_frame_id=None, source_binding=None, image_entry=None, image_size=None, roi=None,
        gt=[], finals=[], n_raw_suppressed=None, source_verified=False, complete=False,
        gt_complete=False, execution_status='NOT_COMPLETED', unavailable_reasons=[], review_tasks=[])
    from .mobile_frame_review_state import ELIGIBILITY,ELIGIBILITY_SHA256,for_session,NO_REFERENCE,SOURCE_NA
    unit['completion_eligibility']=dict(ELIGIBILITY,sha256=ELIGIBILITY_SHA256)
    group = session.mt_invocations.get(gid)
    if not group or not group.executed or group.cancelled or group.execution_failed:
        return unit
    from .mobile_scene_frame_review import unit_for_mt
    scene_input = unit_for_mt(session, gid)
    if scene_input is not None:
        scene_input['completion_eligibility'] = unit['completion_eligibility']
        return scene_input
    unit['execution_status'] = 'COMPLETED'
    try:
        session.verify_source()
        from .mobile_mt_postdedup import source_roles_v2
        unit['source_role_evidence']=source_roles_v2(session,gid)
        scene = core._legacy_scene(session, gid, include_history=False)
        partial.validate(session, gid, scene)
        validate_links(scene)
        view = stacks.snapshot(session, gid, scene)
        stat = session.path.stat()
        key = (sha, stat.st_size, stat.st_mtime_ns, gid, scene['image_entry'], tuple(scene['image_size']))
        with _lock:
            verified = key in _frames.get(session, set())
        if not verified:
            from .mobile_canvas_resolution import verify_frame
            verify_frame(session, gid)
            with _lock:
                _frames.setdefault(session, set()).add(key)
        unit.update(source_verified=True, source_frame_id=sha+':'+scene['image_entry'],
            source_binding=scene['source_binding'], image_entry=scene['image_entry'],
            image_size=scene['image_size'], roi=scene['roi'])
        progress = for_session(session,gid,scene)
        unit['completion_eligibility'].update(operational_state=progress['state'],legacy_checked=scene['checked'])
        unit.update(complete=progress['metric_complete'], gt_complete=progress['metric_complete'])
        if progress['state'] in {NO_REFERENCE,SOURCE_NA}:unit['unavailable_reasons'].append(progress['reason'])
        elif progress['reason']=='GT_REFERENCE_GEOMETRY_UNAVAILABLE':unit['unavailable_reasons'].append(progress['reason'])
        if scene.get('exclusion_reason'):
            unit['unavailable_reasons'].append('SCENE_EXCLUDED')
        if view['conflicts']:
            unit['unavailable_reasons'].append('GT_IDENTITY_CONFLICT')
        unknown = any(c['role'] == 'UNKNOWN' for c in view['candidates'])
        if unknown:
            unit['unavailable_reasons'].append('FINAL_POPULATION_UNPROVEN')
        unit['n_raw_suppressed'] = None if unknown else sum(c['role'] == 'SUPPRESSED' for c in view['candidates'])
        for pid in view['active_plate_ids']:
            p = view['plates'][pid]
            confirmed = pid in progress['confirmed_gt_ids']
            unit['gt'].append(dict(gt_id=pid, quad=copy.deepcopy(p.get('gt_quad')),
                confirmed=confirmed,
                gt_text=p.get('gt_text'), text_confirmed=p.get('text_confirmed'),
                text_exclusion_reason=p.get('text_exclusion_reason'), geometry_state=p.get('geometry_state')))
            if not confirmed:
                optional=pid in progress.get('optional_geometry_ids',[])
                unit['review_tasks'].append(dict(gt_id=pid, task='OPTIONAL_GT_GEOMETRY_FOR_OBJECT_METRIC' if optional else 'CONFIRM_OR_LINK_SAVED_GTR',optional=optional))
        for c in view['candidates']:
            if c['role'] != 'FINAL':
                continue
            aid, oid = c['attempt_id'], c['object_id']
            row = session.attempts[aid]
            confidence = row.get('mt_confidence')
            if confidence is None or str(confidence).strip() == '':
                confidence = row.get('plate_confidence')
            try:
                confidence = float(confidence)
            except (TypeError, ValueError, OverflowError):
                confidence = None
            try:
                index = int(row['mt_detection_index'])
                if str(index) != str(row['mt_detection_index']).strip() or index != c['mt_detection_index']:
                    index = None
            except (TypeError, ValueError, KeyError):
                index = None
            box, coordinates = None, False
            try:
                coordinates = (row.get('source_frame_coordinate_space') == 'mt_source_bitmap_xy'
                    and row.get('source_frame_origin') == 'pipeline_bitmap_before_roi'
                    and [float(row[k]) for k in ('source_frame_width_px', 'source_frame_height_px')] == list(scene['image_size']))
                raw_box = [float(row[k]) for k in ('plate_left', 'plate_top', 'plate_right', 'plate_bottom')]
                box = raw_box if all(math.isfinite(v) for v in raw_box) else None
            except (TypeError, ValueError, KeyError, OverflowError):
                pass
            decision = progress['object_decisions'].get(oid, 'UNRESOLVED')
            link = view['links'].get(oid, {})
            # Persisted partial decisions and legacy checked identity are both
            # source-bound operator evidence. UI overlap and operator_bad flags
            # are deliberately absent from this adapter.
            explicit = ((partial.KEY in scene and scene[partial.KEY]['decisions'].get(oid) == decision)
                or (partial.KEY not in scene and scene['checked'])
                or link.get('identity_decision')
                or link.get('provenance') in {'operator_accept_unchanged_final_reference', 'operator_canvas_association'})
            confirmed = bool(explicit and decision in ('PLATE', 'NOT_PLATE'))
            pid = link.get('plate_id') if decision == 'PLATE' else None
            evidence = dict(kind='PERSISTED_FRAME_DECISION' if partial.KEY in scene else 'LEGACY_OPERATOR_REVIEW',
                attempt_id=aid, object_id=oid, gt_id=pid, decision=decision,
                source_binding=scene['source_binding'], link_provenance=link.get('provenance'),
                review_revision=session.review.review_revision) if confirmed else None
            relation = bool(confirmed and pid in view['plates'] and link.get('status') == 'ASSOCIATED_FOR_GEOMETRY')
            unit['finals'].append(dict(attempt_id=aid, object_id=oid, index=index, role='FINAL',
                confidence=confidence, bbox=box, bbox_source_verified=coordinates,
                pose_quad=copy.deepcopy(c['mt_quad']), pose_source_verified=coordinates,
                source_pose_status=c.get('source_pose_status'), decision=decision,
                operator_confirmed=confirmed, gt_id=pid, relation_confirmed=relation,
                operator_review_evidence=evidence))
            if not confirmed or (decision == 'PLATE' and not relation):
                unit['review_tasks'].append(dict(object_id=oid, attempt_id=aid,
                    task='CONFIRM_MT_GTR_RELATION' if decision == 'PLATE' else 'RESOLVE_FINAL', decision=decision))
        unit['review_tasks'].extend(dict(gt_id=pid, task='CONFIRM_GTR_GEOMETRY') for pid in progress['pending_gtr_ids'])
    except (ValueError, KeyError, TypeError, OSError) as error:
        unit['unavailable_reasons'].append('SOURCE_OR_REVIEW_UNAVAILABLE: ' + str(error))
    return unit


def evaluate_unit(session, gid, *, policy=POLICY, cache=True):
    key = None
    try:
        stat = session.path.stat()
        key = (id(session.review), session.review.review_revision, session.review.source_archive_sha256,
               stat.st_size, stat.st_mtime_ns, gid)
    except OSError:
        pass
    if cache and key is not None and _approved(policy):
        with _lock:
            if key in _cache.get(session, {}):
                return copy.deepcopy(_cache[session][key])
    result = score_unit_v2(unit_from_session(session, gid), policy=policy)
    if cache and key is not None and _approved(policy):
        with _lock:
            old = _cache.setdefault(session, {})
            if old and next(iter(old))[:3] != key[:3]:
                old.clear()
            old[key] = copy.deepcopy(result)
    return result


def aggregate_units(units):
    unique, conflicts = {}, set()
    for u in units:
        key = u['unit_id']
        if key in unique and unique[key] != u:
            conflicts.add(key)
        else:
            unique[key] = copy.deepcopy(u)
    for key in conflicts:
        unique[key].update(status='NOT_EVALUABLE', n_qualified=0, tp=None, fp=None, fn=None,
            n_gt=None, n_final_mt=None, matched_pairs=[], final_outcomes=[], geometry_diagnostics=[],
            unavailable_reasons=['CONFLICTING_UNIT_ALIASES'], **rates(None, None, None))
    cohort = {(u.get('session_id'), u.get('source_archive_sha256'), u.get('review_revision'),
               u.get('matching_policy_id'), u.get('matching_policy_sha256'),
               (u.get('completion_eligibility') or {}).get('sha256','legacy_checked_v1')) for u in unique.values()}
    mixed = len(cohort) > 1 or any(u.get('matching_policy_id') != POLICY_ID or
        u.get('matching_policy_sha256') != POLICY_SHA256 for u in unique.values())
    measured = [u for u in unique.values() if u['status'] == 'MEASURED']
    consistent = all(all(type(u.get(k)) is int and u[k] >= 0 for k in ('tp','fp','fn','n_final_mt','n_gt'))
        and u['tp']+u['fp'] == u['n_final_mt'] and u['tp']+u['fn'] == u['n_gt'] for u in measured)
    if mixed or not consistent:
        measured = []
    result = dict(schema=SCHEMA, matching_policy_id=POLICY_ID, matching_policy_sha256=POLICY_SHA256,
        methodology_version='F4-EVAL-2.0', adopted_date='2026-11-01',
        chronology='RETROSPECTIVE_NOT_PREREGISTERED', gt_mode='ASSISTED',
        unit='mt_invocation_id+source_frame+ROI', n_qualified=len(measured), n_total=len(unique),
        units=sorted(unique.values(), key=lambda u: u['unit_id']), alias_conflicts=sorted(conflicts),
        status='COMPLETE' if measured and len(measured) == len(unique) else 'PARTIAL' if measured else 'NOT_EVALUABLE',
        unavailable_reasons=(['CONFLICTING_UNIT_ALIASES'] if conflicts else [])
            + (['MIXED_SESSION_OR_POLICY_COHORT'] if mixed else [])
            + ([] if consistent else ['INCONSISTENT_UNIT_COUNTS']))
    if unique and all('completion_eligibility' in u for u in unique.values()):
        from .mobile_frame_review_state import ELIGIBILITY,ELIGIBILITY_SHA256
        result['completion_eligibility']=dict(ELIGIBILITY,sha256=ELIGIBILITY_SHA256)
    for field in ('tp', 'fp', 'fn', 'n_gt', 'n_final_mt', 'n_raw_suppressed'):
        result[field] = sum(u[field] for u in measured) if measured else None
    result.update(rates(result['tp'], result['fp'], result['fn'], n_gt=result['n_gt'], n_final_mt=result['n_final_mt']))
    if not measured:
        result['unavailable_reasons'].append('NO_QUALIFIED_UNITS')
    pairs = [p for u in measured for p in u['matched_pairs']]
    valid = [p for p in pairs if p['polygon_iou_pct'] is not None and p['nmce_pct'] is not None]
    bbox = [p['bbox_iou'] for p in pairs if p['bbox_iou'] is not None]
    result['geometry_summary'] = dict(n_pairs=len(pairs), n_geometry=len(valid), n_bbox=len(bbox),
        status='FULL' if valid and len(valid) == len(pairs) else 'SUBSET' if valid else 'N/A',
        polygon_iou_mean_pct=sum(p['polygon_iou_pct'] for p in valid)/len(valid) if valid else None,
        nmce_mean_pct=sum(p['nmce_pct'] for p in valid)/len(valid) if valid else None,
        bbox_iou_mean=sum(bbox)/len(bbox) if bbox else None,
        unavailable_reasons=sorted({p['pose_reason'] for p in pairs if p['pose_reason']}))
    return result


def evaluate_session(session):
    identity = dict(session_id=session.session_id, source_archive_sha256=session.review.source_archive_sha256,
                    review_revision=session.review.review_revision)
    epoch = id(session.review)
    result = aggregate_units(evaluate_unit(session, gid) for gid in session.mt_invocations)
    if not session.mt_invocations and (getattr(session.bundle, 'scene_evidence', {}) or {}).get('available'):
        # An empty MT population is a current N/A result, not a stale revision.
        # The independent full-scene assessment remains outside MT denominators.
        from .mobile_frame_review_state import ELIGIBILITY, ELIGIBILITY_SHA256
        result['completion_eligibility'] = dict(ELIGIBILITY, sha256=ELIGIBILITY_SHA256)
    from .mobile_object_mz_quality import snapshot
    result['ocr_summary'] = snapshot(session, result)
    if (epoch != id(session.review) or any(getattr(session.review, k) != v for k, v in identity.items() if k != 'session_id')
            or any(any(u.get(k) != v for k, v in identity.items()) for u in result['units'])):
        result.update(status='STALE_REVIEW', n_qualified=0, tp=None, fp=None, fn=None, n_gt=None, n_final_mt=None)
        result['unavailable_reasons'].append('REVIEW_REVISION_CHANGED')
        result.update(rates(None, None, None))
    result.update(**identity, matching_policy=json.loads(_POLICY_JSON))
    return result


def reason_text(result):
    reasons = result.get('unavailable_reasons') or []
    first = reasons[0].split(':', 1)[0] if reasons else ''
    return {'GT_GEOMETRY_UNCONFIRMED': 'Potwierdź obrys GTR',
        'REVIEW_INCOMPLETE': 'Dokończ ocenę obrazu', 'FINAL_DECISION_UNRESOLVED': 'Rozstrzygnij ramki MT',
        'MT_GTR_RELATION_UNPROVEN': 'Potwierdź powiązanie MT ↔ GTR',
        'OPERATOR_DECISION_UNPROVEN': 'Brak zapisanej decyzji operatora',
        'GT_PARTIAL_ROI_COVERAGE': 'Nieustalona granica ROI', 'MT_NOT_COMPLETED': 'MT nie zostało zakończone',
        'SCENE_EXCLUDED': 'Obraz poza oceną', 'FINAL_POPULATION_UNPROVEN': 'Nieustalone końcowe wskazania MT',
        'SOURCE_OR_REVIEW_UNAVAILABLE': 'Brak zgodnego źródła lub referencji',
        'SOURCE_FINAL_ROLE_UNPROVEN': 'Nieudowodnione role FINAL/RAW · N/A',
        'NO_REFERENCE_EVIDENCE': 'Brak zapisanych referencji · N/A',
        'NO_QUALIFIED_UNITS': 'Brak ocenialnych wywołań',
        'FINAL_CONFIDENCE_UNAVAILABLE': 'Brak źródłowej pewności MT'}.get(first, 'Brak kompletnych danych MT')


def export(session, directory):
    from .mobile_mt_object_export import export as write_export
    return write_export(session, directory)
