"""Audited, compare-and-swap cleanup of a proven false model-only reference."""
from __future__ import annotations
import copy,hashlib

ACTION='system_reconcile_false_model_reference'


def prepare(session,gid,pid):
    from . import mobile_reference_disposition as legacy,mobile_reference_lifecycle as life
    from . import mobile_canvas_partial_review as partial
    if session.review.review_status=='COMPLETED':
        raise ValueError('COMPLETED_REVIEW_PROTECTED')
    scene,state,target=legacy._context(session,gid,None)
    existing=life.archived_reference(state,pid)
    if existing:
        target.update(kind='RECONCILE_FALSE_MODEL_REFERENCE',plate_id=pid,status='NOOP_ALREADY_RECONCILED')
        event=next((e for e in reversed(scene.get('events',[])) if e.get('action')==ACTION and e.get('plate_id')==pid),None)
        if event:
            target['original_precondition']=dict(review_revision=event['review_revision_before'],sidecar_sha256=event['backup_sha256'])
        return target
    candidate=next((c for c in state['candidates'] if c['object_id']==state['plates'].get(pid,{}).get('origin_object_id')),None)
    if (not candidate or candidate['role']!='FINAL' or candidate['association_status']!='UNASSOCIATED_NO_PLATE'
            or partial.progress_decisions(scene).get(candidate['object_id'])!=partial.NOT_PLATE):
        raise ValueError('REJECTED_SOURCE_FINAL_NOT_PROVEN')
    evidence=life.dependency_evidence(scene,state,pid)
    legacy._history_proven(scene,state,pid)
    target.update(kind='RECONCILE_FALSE_MODEL_REFERENCE',plate_id=pid,status='READY',
        source_object_id=candidate['object_id'],source_attempt_id=candidate['attempt_id'],
        plate_sha256=legacy._digest(state['plates'][pid]),evidence=evidence)
    return target


def apply(session,target,*,backup_path):
    from pathlib import Path
    from . import mobile_reference_disposition as legacy,mobile_reference_lifecycle as life
    from . import mobile_canvas_review as core,mobile_candidate_stacks as stacks
    if target.get('kind')!='RECONCILE_FALSE_MODEL_REFERENCE':raise ValueError('INVALID_CLEANUP_TICKET')
    fresh=prepare(session,target['mt_invocation_id'],target['plate_id'])
    if fresh!=target:raise ValueError(legacy.STALE)
    if fresh['status']=='NOOP_ALREADY_RECONCILED':return dict(status='NOOP_ALREADY_RECONCILED',review_revision=session.review.review_revision)
    backup=Path(backup_path)
    if hashlib.sha256(backup.read_bytes()).hexdigest()!=target['sidecar_sha256']:
        raise ValueError('EXACT_BACKUP_REQUIRED')
    gid=target['mt_invocation_id'];pid=target['plate_id']
    scene=core._legacy_scene(session,gid);state=stacks._state(scene)
    record=life.retire(scene,state,target['source_object_id'])
    if not record or pid in state['plates']:raise ValueError('CLEANUP_EVIDENCE_CHANGED')
    scene[stacks.KEY]=state
    details=dict(actor='PROGRAM_RECONCILIATION',reason='PROVEN_FALSE_MODEL_DERIVED_REFERENCE',
        plate_id=pid,source_object_id=target['source_object_id'],source_attempt_id=target['source_attempt_id'],
        session_id=session.session_id,source_archive_sha256=scene['source_archive_sha256'],source_binding=scene['source_binding'],
        mt_invocation_id=gid,backup_sha256=target['sidecar_sha256'],backup_path=str(backup.resolve()),
        evidence=copy.deepcopy(target['evidence']),reference_after=None,
        checked_preserved=bool(scene['checked']))
    core._store(session,gid,stacks.project(session,gid,scene),ACTION,pid,
        object_before=record['plate'],event_details=details,expected_sidecar_sha256=target['sidecar_sha256'])
    return dict(status='RECONCILED',review_revision=session.review.review_revision,plate_id=pid,
        sidecar_sha256=hashlib.sha256(session.sidecar_path.read_bytes()).hexdigest())
