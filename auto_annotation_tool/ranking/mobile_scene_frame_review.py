"""Scene-frame scope in the existing Canvas, separate from MT invocation review."""
from __future__ import annotations
import copy
import hashlib
import json
import uuid

SCHEMA = 'alpr.mobile.scene_frame_review.v1'
SCOPE = 'SCENE_FRAME'

def contains(session, key):
    return key in getattr(session, 'scene_frames', {})

def context(session, key, image_size=None):
    row = getattr(session, 'scene_frames', {}).get(key)
    if not row or row['write_status'] != 'WRITTEN' or not session.bundle.validation.ok:
        return None
    session.verify_source()
    size = (row['source_frame_width_px'], row['source_frame_height_px'])
    if image_size is not None and tuple(image_size) != size:
        return None
    return dict(scope=SCOPE, scene_event_id=key, entry=row['source_frame_entry'], size=size,
                roi=(0, 0, *size), source_stamp=copy.deepcopy(row['source_stamp']), records=(),
                source_sequence=str(row['source_stamp']['source_sequence']),
                source_timestamp_nanos=str(row['source_stamp']['source_timestamp_nanos']))

def _seed(session, key):
    ctx = context(session, key)
    if ctx is None:
        row = session.scene_frames.get(key, {})
        raise ValueError('SCENE_FRAME N/A: ' + str(row.get('missing_reason') or 'SOURCE_INTEGRITY_UNAVAILABLE'))
    binding = dict(scope=SCOPE, session_id=session.session_id, scene_event_id=key,
                   source_archive_sha256=session.review.source_archive_sha256,
                   image_entry=ctx['entry'], image_size=list(ctx['size']), source_stamp=ctx['source_stamp'])
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema=SCHEMA, **binding, source_binding=digest, edit_bounds=list(ctx['roi']),
                parents=[], objects=[], events=[], checked=False,
                exclusion_reason='', presence='UNRESOLVED', mt_decisions={}, provenance='ASSISTED_SCENE_FRAME',
                independent_gt=False, decision_contract='scene_frame_v1')

def _registry(s):
    associations = {o['object_id']: o['physical_gt_id'] for o in s['objects']}
    s['physical_gt_v1'] = dict(plates=[dict(gt_id=o['physical_gt_id'], reference_object_id=o['object_id'],
        quad=copy.deepcopy(o['quad']), scene_identity='DISTINCT') for o in s['objects']],
        associations=associations, confirmed=s['checked'], decision_source='operator_scene_frame',
        scope=dict(scene_event_id=s['scene_event_id'], source_binding=s['source_binding']))
    return s

def _validate(session, key, saved):
    from . import mobile_canvas_review as core
    seed = _seed(session, key)
    for field in ('schema', 'scope', 'session_id', 'scene_event_id', 'source_archive_sha256',
                  'image_entry', 'image_size', 'source_stamp', 'source_binding', 'edit_bounds', 'parents'):
        if saved.get(field) != seed[field]:
            raise ValueError('SCENE_FRAME source binding conflict: ' + field)
    if saved.get('mt_invocation_id') is not None:
        raise ValueError('SCENE_FRAME cannot manufacture an MT invocation')
    core._validate_objects(_view(saved))
    identifiers = [o.get('physical_gt_id') for o in saved['objects']]
    if any(not isinstance(v, str) or not v.startswith('PGT:') for v in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError('Invalid physical identity in scene')
    if saved.get('presence') not in {'UNRESOLVED', 'PRESENT', 'NO_PLATES'}:
        raise ValueError('Invalid scene presence decision')
    if saved['presence'] == 'NO_PLATES' and saved['objects']:
        raise ValueError('Empty scene cannot contain registered plates')
    if saved['checked'] and not saved['exclusion_reason']:
        if saved['presence'] == 'UNRESOLVED' or (saved['presence'] == 'PRESENT' and not saved['objects']):
            raise ValueError('Scene lacks explicit operator decision')
    for obj in saved['objects']:
        if type(obj.get('scene_confirmed')) is not bool:
            raise ValueError('Invalid physical-plate confirmation')
        if obj['quad'] is None and not obj.get('geometry_exclusion_reason'):
            raise ValueError('Presence-only needs an explicit geometry reason')
        for aid in obj.get('mt_attempt_links', []):
            role = next((r for r in session.mt_detection_roles.values()
                         if aid in r['final_attempt_ids'] and r['scene_event_id'] == key), None)
            if role is None:
                raise ValueError('Physical reference link lacks source FINAL evidence')
    by_id = {o['physical_gt_id']: o for o in saved['objects']}
    decisions = saved.get('mt_decisions', {})
    if not isinstance(decisions, dict):
        raise ValueError('Invalid per-FINAL decisions')
    for aid, decision in decisions.items():
        role = role_for_final(session, key, aid)
        if (decision.get('decision') not in {'PLATE','NOT_PLATE'}
                or decision.get('source_binding') != seed['source_binding']
                or decision.get('mt_invocation_id') != role['mt_invocation_id']):
            raise ValueError('FINAL decision has a foreign source')
        pid = decision.get('physical_gt_id')
        if (decision['decision']=='PLATE') != (pid in by_id):
            raise ValueError('FINAL decision lacks the physical reference')
    for pid,obj in by_id.items():
        links=obj.get('mt_attempt_links',[])
        expected={aid for aid,d in decisions.items() if d.get('physical_gt_id')==pid and d['decision']=='PLATE'}
        if len(links)!=len(set(links)) or set(links)!=expected:
            raise ValueError('Physical reference and explicit FINAL links disagree')

def validate_loaded(session, review):
    annotations = getattr(review, 'scene_annotations', {})
    if not isinstance(annotations, dict) or set(annotations) - set(getattr(session, 'scene_frames', {})):
        raise ValueError('Scene review refers to a foreign event')
    for key, saved in annotations.items():
        _validate(session, key, saved)

def scene(session, key, *, include_history=True):
    seed = _seed(session, key)
    saved = session.review.scene_annotations.get(key)
    if saved is None:
        return _registry(_view(seed))
    _validate(session, key, saved)
    return _registry(_view(copy.deepcopy(saved if include_history else {k:v for k,v in saved.items() if k != 'scene_history_v1'})))

def _view(value):
    # Existing geometry widgets consume roi as editing bounds. This transient
    # adapter never publishes a selected ROI or an MT identifier in the sidecar.
    return dict(value, roi=list(value['edit_bounds']), mt_invocation_id=None)

def store(session, key, value, action, oid='', *, restore_history=False, expected_sidecar_sha256=None):
    from .mobile_human_review import _utc_now_iso
    current = scene(session, key, include_history=False)
    out = copy.deepcopy({k:v for k,v in value.items() if k not in ('scene_history_v1','roi','mt_invocation_id')})
    if not restore_history:
        out['checked'] = bool(out['exclusion_reason'] or
            out['presence'] == 'NO_PLATES' or
            (out['presence'] == 'PRESENT' and out['objects'] and all(o['scene_confirmed'] for o in out['objects'])))
    out['events'] = list(current['events'])
    out['events'].append(dict(action=action, object_id=oid, saved_at=_utc_now_iso(),
        review_revision_before=session.review.review_revision,
        object_before=copy.deepcopy(next((o for o in current['objects'] if o['object_id'] == oid), None)),
        object_after=copy.deepcopy(next((o for o in out['objects'] if o['object_id'] == oid), None))))
    saved_history = session.review.scene_annotations.get(key,{}).get('scene_history_v1')
    history = dict(saved_history) if saved_history else dict(states=[_state(current)], cursor=0)
    if not restore_history:
        history['states'] = history['states'][:history['cursor']+1] + [_state(out)]
        history['cursor'] += 1
    else:
        history = dict(value['scene_history_v1'])
    out['scene_history_v1'] = history
    _registry(out)
    _validate(session, key, out)
    changed = copy.copy(session.review)
    changed.scene_annotations = dict(session.review.scene_annotations, **{key:out})
    if changed.review_mode == 'blinded_gt_v1' and not changed.review_mode_locked:
        changed.review_mode = 'assisted'
        changed.review_mode_origin = 'blinded_gt_v1'
    changed.review_status = 'IN_PROGRESS'
    opts = {} if expected_sidecar_sha256 is None else {'_expected_sidecar_sha256':expected_sidecar_sha256}
    session._save_change(changed, **opts)

def _state(value):
    return copy.deepcopy({k:v for k,v in value.items() if k not in ('scene_history_v1', 'events', 'roi', 'mt_invocation_id')})

def restore(session, key, *, redo=False):
    scene(session, key, include_history=False)
    source_history = session.review.scene_annotations.get(key,{}).get('scene_history_v1')
    history = dict(source_history) if source_history else None
    if not history:
        return False
    cursor = history['cursor'] + (1 if redo else -1)
    if not 0 <= cursor < len(history['states']):
        return False
    value = copy.deepcopy(history['states'][cursor])
    history['cursor'] = cursor
    value['scene_history_v1'] = history
    store(session, key, value, 'redo_scene_frame' if redo else 'undo_scene_frame', restore_history=True)
    return True

def add_plate(session, key, points=None, *, geometry_reason=''):
    from . import mobile_canvas_review as core
    from .mobile_manual_frame_guard import require_free_manual_area
    s = scene(session, key)
    quad = core.ordered_quad(points, s['image_size'], s['roi']) if points is not None else None
    if quad is not None:
        require_free_manual_area(s, quad, session=session, gid=key)
    elif not str(geometry_reason).strip():
        raise ValueError('State why reference corners cannot be evaluated')
    if len(s['objects']) >= core.MAX_OBJECTS:
        raise ValueError('Canvas object limit reached')
    oid = 'MANUAL:' + uuid.uuid4().hex
    obj = core._new_object(oid, quad=quad)
    obj.update(physical_gt_id='PGT:' + uuid.uuid4().hex, scene_confirmed=True,
        geometry_state='operator_corrected' if quad else 'unavailable',
        geometry_exclusion_reason=str(geometry_reason).strip(), mt_attempt_links=[])
    s['objects'].append(obj)
    s.update(presence='PRESENT', exclusion_reason='')
    store(session, key, s, 'add_scene_plate', oid)
    return oid

def object_id(s, reference):
    return next((o['object_id'] for o in s['objects']
        if reference in (o['object_id'], o['physical_gt_id'], o['physical_gt_id']+':reference')), reference)

def edit(session, key, reference, *, quad=None, text=None, reason=''):
    from . import mobile_canvas_review as core
    from .mobile_human_review import normalize_registration
    s = scene(session, key)
    obj = core._object(s, object_id(s, reference))
    if quad is not None:
        from .mobile_manual_frame_guard import require_free_manual_area
        points = core.ordered_quad(quad, s['image_size'], s['roi'])
        require_free_manual_area(s, points, exclude_plate_id=obj['physical_gt_id'], session=session, gid=key)
        obj.update(quad=points, geometry_state='operator_corrected', scene_confirmed=True, geometry_exclusion_reason='')
    if text is not None:
        text, reason = normalize_registration(text), ' '.join(str(reason).split())
        if (text and reason) or len(text)>256 or len(reason)>240:
            raise ValueError('Use GTZ or text_exclusion_reason, not both')
        obj.update(gt_text=text, text_exclusion_reason=reason, text_confirmed=bool(text or reason))
    s['exclusion_reason'] = ''
    store(session, key, s, 'edit_scene_plate', obj['object_id'])

def remove(session, key, reference):
    s = scene(session, key)
    oid = object_id(s, reference)
    if not any(o['object_id'] == oid for o in s['objects']):
        raise ValueError('Unknown physical plate')
    pid = next(o['physical_gt_id'] for o in s['objects'] if o['object_id']==oid)
    s['mt_decisions'] = {a:d for a,d in s.get('mt_decisions',{}).items() if d.get('physical_gt_id')!=pid}
    s['objects'] = [o for o in s['objects'] if o['object_id'] != oid]
    s.update(presence='PRESENT' if s['objects'] else 'UNRESOLVED', exclusion_reason='')
    store(session, key, s, 'remove_scene_plate', oid)

def confirm_empty(session, key):
    s = scene(session, key)
    if s['objects']:
        raise ValueError('Scene contains physical reference plates')
    if s['presence'] == 'NO_PLATES':
        return False
    s.update(presence='NO_PLATES', exclusion_reason='')
    store(session, key, s, 'operator_confirm_empty_full_scene')

def toggle(session, key, reference):
    from . import mobile_canvas_review as core
    s = scene(session, key)
    obj = core._object(s, object_id(s, reference))
    obj['scene_confirmed'] = not obj['scene_confirmed']
    store(session, key, s, 'operator_toggle_scene_plate', obj['object_id'])


def role_for_final(session, key, aid):
    row = session.scene_frames[key]
    role = next((r for r in session.mt_detection_roles.values() if r['scene_event_id']==key
                 and aid in r['final_attempt_ids']), None)
    if (role is None or role['role_proof_status']!='COMPLETE' or role['stale_or_cancelled']
            or role['execution_error'] or role['source_stamp']!=row['source_stamp']):
        raise ValueError('FINAL is not proven on this exact scene bitmap')
    return role

def decide_mt(session, key, aid, *, reference=None, is_plate=True):
    from . import mobile_canvas_review as core
    s = scene(session, key)
    role = role_for_final(session, key, aid)
    obj = core._object(s, object_id(s, reference)) if is_plate else None
    decision = dict(decision='PLATE' if is_plate else 'NOT_PLATE',
        physical_gt_id=obj['physical_gt_id'] if obj else None,
        source_binding=s['source_binding'], mt_invocation_id=role['mt_invocation_id'],
        review_revision_before=session.review.review_revision,
        provenance='OPERATOR_EXPLICIT_SCENE_FINAL')
    s.setdefault('mt_decisions', {})[aid] = decision
    for other in s['objects']:
        other['mt_attempt_links'] = [v for v in other.get('mt_attempt_links', []) if v!=aid]
    if obj:
        obj['mt_attempt_links'].append(aid)
    store(session, key, s, 'operator_scene_final_decision', obj['object_id'] if obj else '')

def unit_for_mt(session, gid):
    """Adapt explicitly reviewed full-scene references to the unchanged ROI scorer."""
    import math
    from .mobile_source_frame_roi import source_invocation_context, source_pose_quad
    from .mobile_mt_postdedup import source_roles_v2
    # Existing local decisions always take precedence; never migrate them.
    if session.review.invocation_annotations.get(gid):
        return None
    role = getattr(session, 'mt_detection_roles', {}).get(gid)
    if not role or role['scene_event_id'] not in session.review.scene_annotations:
        return None
    key = role['scene_event_id']
    s = scene(session, key, include_history=False)
    if role['source_stamp'] != s['source_stamp']:
        return None  # No geometry transport across different CameraX frames.
    group = session.mt_invocations.get(gid)
    ctx = source_invocation_context(group.records, gid, session.entry_names) if group else None
    if not ctx or list(ctx['size']) != s['image_size']:
        return None
    evidence = source_roles_v2(session, gid)
    binding = hashlib.sha256(json.dumps(dict(scene=s['source_binding'], role=role),
        sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    unit = dict(session_id=session.session_id, source_archive_sha256=session.review.source_archive_sha256,
        review_revision=session.review.review_revision, unit_id=session.review.source_archive_sha256+':'+gid,
        mt_invocation_id=gid, source_frame_id=session.review.source_archive_sha256+':'+ctx['entry'],
        source_binding=binding,image_entry=ctx['entry'],image_size=list(ctx['size']),roi=list(ctx['roi']),
        gt=[],finals=[],n_raw_suppressed=len(role['suppressed_attempt_ids']) if evidence['proof_status']=='PROVEN' else None,source_verified=True,
        execution_status='COMPLETED' if group.executed and not group.cancelled and not group.execution_failed else 'NOT_COMPLETED',
        complete=s['checked'] and not s['exclusion_reason'],gt_complete=s['checked'] and not s['exclusion_reason'],
        unavailable_reasons=[],review_tasks=[],source_role_evidence=evidence,
        scene_event_id=key,reference_scope='SCENE_FRAME_SAME_STAMP_TO_MT_ROI')
    if evidence['proof_status'] != 'PROVEN':
        unit['unavailable_reasons'].append('FINAL_POPULATION_UNPROVEN')
    if s['exclusion_reason']:
        unit['unavailable_reasons'].append('SCENE_EXCLUDED')
    for obj in s['objects']:
        unit['gt'].append(dict(gt_id=obj['physical_gt_id'],quad=copy.deepcopy(obj['quad']),
            confirmed=obj['scene_confirmed'],gt_text=obj['gt_text'],text_confirmed=obj['text_confirmed'],
            text_exclusion_reason=obj['text_exclusion_reason'],geometry_state=obj['geometry_state']))
    for aid in role['final_attempt_ids']:
        row = session.attempts[aid]
        decision = s.get('mt_decisions',{}).get(aid,{})
        pid = decision.get('physical_gt_id')
        try:
            confidence=float(row.get('mt_confidence') or row.get('plate_confidence'))
            if not math.isfinite(confidence):confidence=None
        except (TypeError,ValueError):confidence=None
        try:box=[float(row[k]) for k in ('plate_left','plate_top','plate_right','plate_bottom')]
        except (TypeError,ValueError,KeyError):box=None
        proof = dict(kind='PERSISTED_SCENE_FINAL_DECISION',attempt_id=aid,object_id=aid,
            gt_id=pid,decision=decision['decision'],source_binding=binding,scene_source_binding=s['source_binding'],
            review_revision=session.review.review_revision) if decision else None
        unit['finals'].append(dict(attempt_id=aid,object_id='GT:'+aid,index=int(row['mt_detection_index']),
            role='FINAL',confidence=confidence,bbox=box,bbox_source_verified=True,
            pose_quad=source_pose_quad(row,ctx['size']),pose_source_verified=True,source_pose_status=row.get('mt_pose_status'),
            decision=decision.get('decision','UNRESOLVED'),operator_confirmed=bool(decision),
            gt_id=pid,relation_confirmed=bool(pid),operator_review_evidence=proof))
    if not unit['gt'] and not unit['finals']:
        unit['unavailable_reasons'].append('NO_REFERENCE_EVIDENCE')
    return unit



def mt_review_issues(session, gid):
    unit=unit_for_mt(session,gid)
    if unit is None:return None
    if not unit['complete']:
        return ['SCENE_FRAME '+unit['scene_event_id']+': dokończ ocenę pełnej sceny']
    if unit['source_role_evidence']['proof_status']!='PROVEN':
        return []  # Technical N/A, not a request for invented source evidence.
    return ['Rozstrzygnij FINAL '+c['attempt_id']+' w pełnej scenie'
            for c in unit['finals'] if c['decision']=='UNRESOLVED']


def progress(s):
    objects = s['objects']
    pending = [o['physical_gt_id'] for o in objects if not o['scene_confirmed']]
    complete = bool(s['checked'])
    return dict(state='FRAME_READY_ASSISTED' if complete else 'FRAME_PENDING', complete=complete,
        metric_complete=False, decisions=len(pending) + int(not objects and not complete),
        pending_ids=[], pending_gtr_ids=pending, resolved=len(objects)-len(pending), total=len(objects),
        correct=len(objects)-len(pending), false_positive=0, manual_added=len(objects), plate_count=len(objects),
        empty_eligible=not objects, empty_confirmed=s['presence']=='NO_PLATES', source_unknown=False,
        object_decisions={o['object_id']: 'PLATE' if o['scene_confirmed'] else 'UNRESOLVED' for o in objects},
        confirmed_gt_ids=[o['physical_gt_id'] for o in objects if o['scene_confirmed']],
        legacy_checked=complete, reason='' if complete else 'SCENE_REVIEW_PENDING', scope=SCOPE)

def issues(session):
    result = []
    for key, row in getattr(session, 'scene_frames', {}).items():
        if row['write_status'] == 'WRITTEN' and not scene(session, key)['checked']:
            result.append('SCENE_FRAME ' + key + ': potwierdź obecność tablic albo świadomy brak tablic')
    return result

def coverage(session, key):
    s = scene(session, key)
    row = session.scene_frames[key]
    out = dict(scene_event_id=key, scope=SCOPE, gt_mode='ASSISTED',
               status='N/A', reason='SCENE_REVIEW_PENDING', n_physical_gt=None, rows=[],
               tv_e2e_status='N/A', tv_e2e_reason='SCENE_UNMAPPED', mt_fn=None)
    if not s['checked'] or s['exclusion_reason']:
        return out
    out['n_physical_gt'] = len(s['objects'])
    observations = [m for m in row['mp_observations']
                    if m['source_stamp'] == row['source_stamp'] and m['execution_status']=='EXECUTED']
    if not observations:
        out['reason'] = 'ROI_COVERAGE_UNPROVEN'; return out
    # Multiple observations on the exact same stamp must agree; never choose one.
    rois = [m['selected_rois'] for m in observations]
    if any(v != rois[0] for v in rois):
        out['reason'] = 'ROI_OBSERVATION_CONFLICT'; return out
    selected = rois[0]
    for obj in s['objects']:
        quad = obj['quad']
        status = 'OUTSIDE_ROI' if not selected else 'N/A'
        if selected and quad:
            boxes = [(r['left'],r['top'],r['right'],r['bottom']) for r in selected]
            if any(all(l<=x<=r and t<=y<=b for x,y in quad) for l,t,r,b in boxes):
                status = 'COVERED'
            else:
                # Conservative bounded boxes only distinguish proven disjointness.
                xs,ys=zip(*quad)
                if all(max(xs)<l or min(xs)>r or max(ys)<t or min(ys)>b for l,t,r,b in boxes):
                    status = 'OUTSIDE_ROI'
                else:status = 'PARTIAL_ROI'
        out['rows'].append(dict(physical_gt_id=obj['physical_gt_id'], coverage=status,
            text_exclusion_reason=obj['text_exclusion_reason'],
            geometry_status='AVAILABLE' if quad else 'N/A'))
    out.update(status='ASSESSED', reason='' if s['objects'] else 'ZERO_DENOMINATOR',
               n_covered=sum(r['coverage']=='COVERED' for r in out['rows']),
               n_outside=sum(r['coverage']=='OUTSIDE_ROI' for r in out['rows']))
    return out

def summary(session):
    extension = getattr(session.bundle, 'scene_evidence', {}) or {}
    rows=[]
    for key,row in getattr(session,'scene_frames',{}).items():
        if row['write_status']!='WRITTEN':
            rows.append(dict(scene_event_id=key,status='N/A',reason=row['missing_reason'],n_physical_gt=None,
                             tv_e2e_reason='SCENE_UNMAPPED',mt_fn=None));continue
        rows.append(coverage(session,key))
    return dict(schema='alpr.mobile.scene_frame_assessment.v1',session_id=session.session_id,
        source_archive_sha256=session.review.source_archive_sha256,review_revision=session.review.review_revision,
        evidence_status=extension.get('status', 'ABSENT'),
        scene_population_complete=bool(extension.get('scene_evidence_complete')
                                       and not extension.get('pending_loss_unknown', True)),
        pending_loss_unknown=extension.get('pending_loss_unknown', True),
        gt_mode='ASSISTED',n_scene_events=len(rows),n_assessed=sum(scene(session,k)['checked'] and not scene(session,k)['exclusion_reason']
            for k,r in getattr(session,'scene_frames',{}).items() if r['write_status']=='WRITTEN'),
        n_missing=sum(r['write_status']=='MISSING' for r in getattr(session,'scene_frames',{}).values()),
        n_mt_invocations=len(session.mt_invocations),n_attempts=len(session.attempts),scenes=rows,
        tv_e2e_status='N/A',tv_e2e_reason='SCENE_UNMAPPED')

def export(session, directory):
    from pathlib import Path
    from .mobile_human_review import _atomic_write
    if not (getattr(session.bundle, 'scene_evidence', {}) or {}).get('available'):
        return []
    path=Path(directory)/'scene_frame_assessment.json'
    payload=dict(summary(session),source_scene_records=list(session.scene_frames.values()),
                 source_mt_roles=list(session.mt_detection_roles.values()),scene_annotations=session.review.scene_annotations)
    _atomic_write(path,json.dumps(payload,ensure_ascii=False,indent=2))
    return [path]
