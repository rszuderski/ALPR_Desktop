"""Read-only operational completion of source-bound FINAL/GTR decisions.

ASSISTED operator references are not an exhaustive independent scene census.
The historical checked/empty flags are evidence, never a required new gesture.
"""
from __future__ import annotations
import hashlib,json,math

READY='FRAME_READY_ASSISTED'
PENDING='PENDING_FRAMES'
SOURCE_NA='SOURCE_NOT_EVALUABLE'
NO_REFERENCE='NO_REFERENCE_EVIDENCE'
BASIS='AUTO_FROM_FRAME_DECISIONS'
ELIGIBILITY=dict(id='f4_frame_decisions_assisted_v1',version=1,adopted_date='2026-11-02',
    completion_basis=BASIS,gt_scope='operator_reference_scope',fn_scope='saved_physical_GT_only',
    independent_exhaustive_scene_gt=False,chronology='RETROSPECTIVE_NOT_PREREGISTERED',
    zero_final_zero_gt='NO_REFERENCE_EVIDENCE',matching_policy='F4-EVAL-2.0_UNCHANGED')
ELIGIBILITY_SHA256=hashlib.sha256(json.dumps(ELIGIBILITY,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def from_scene(scene,*,editing=False,source_reason=None):
    """Pure predicate over a canonically source-validated Canvas projection."""
    if scene and scene.get('scope') == 'SCENE_FRAME' and not source_reason:
        from .mobile_scene_frame_review import progress
        return progress(scene)
    from . import mobile_canvas_partial_review as partial
    out=dict(state=SOURCE_NA,complete=True,metric_complete=False,decisions=0,pending_ids=[],pending_gtr_ids=[],
        resolved=0,total=0,correct=0,false_positive=0,manual_added=0,plate_count=0,empty_eligible=False,
        empty_confirmed=bool((scene or {}).get(partial.KEY,{}).get('empty_confirmed')),source_unknown=True,
        object_decisions={},confirmed_gt_ids=[],completion_basis=BASIS,eligibility=dict(ELIGIBILITY,sha256=ELIGIBILITY_SHA256),
        legacy_checked=bool((scene or {}).get('checked')),reason=source_reason or 'SOURCE_FRAME_UNAVAILABLE')
    if scene is None or source_reason:return out
    state,data=partial._state(scene)
    if scene.get('exclusion_reason'):
        out['reason']='SCENE_EXCLUDED';return out
    try:
        w,h=scene['image_size'];l,t,r,b=scene['roi']
        valid=bool(scene.get('source_binding') and state.get('source_binding')==scene['source_binding']
            and len(scene.get('source_archive_sha256',''))==64 and scene.get('mt_invocation_id') and scene.get('image_entry')
            and all(type(v) in (int,float) and math.isfinite(v) for v in (w,h,l,t,r,b)) and 0<=l<r<=w and 0<=t<b<=h)
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:out['reason']='SOURCE_FRAME_ROI_BINDING_UNAVAILABLE';return out
    from .mobile_canvas_review import manual_bad_objects,validate_mt_quad
    roles=data.get('source_roles',{});objects=partial._objects(scene)
    if scene.get('_post_nms_unknown') or any(role not in {'FINAL','SUPPRESSED'} for role in roles.values()):
        out['reason']='SOURCE_FINAL_ROLE_UNPROVEN';return out
    if (state.get('conflicts') or scene.get('_post_nms_unknown') or scene.get('_post_nms_pending')
            or scene.get('migration_warnings') or manual_bad_objects(scene)
            or any(role not in {'FINAL','SUPPRESSED'} for role in roles.values())
            or set(roles)!={oid for oid,o in objects.items() if o.get('parent_id')}):
        out['reason']='FINAL_POPULATION_OR_REFERENCE_IDENTITY_UNPROVEN';return out
    decisions=dict(data['decisions']);manual=partial.manual_plate_ids(scene,state);active=partial.active_plate_ids(scene,state)
    pending=[oid for oid,value in decisions.items() if value==partial.UNRESOLVED]
    pending_gt=set();confirmed=[];optional_geometry=[]
    flags=set(data.get('pending_gtr_ids',[]))
    for pid in sorted(active):
        p=state['plates'][pid]
        if p.get('scene_identity')=='UNRESOLVED_ACROSS_INVOCATIONS':
            pending_gt.add(pid);continue
        if not p.get('gt_quad') and p.get('confirmed') and p.get('geometry_state')!='operator_removed':
            bbox_identity=any(link.get('plate_id')==pid and link.get('identity_decision')
                and decisions.get(oid)==partial.PLATE for oid,link in state['links'].items())
            if bbox_identity:
                optional_geometry.append(pid);continue
        if not p.get('gt_quad') or not p.get('confirmed'):
            pending_gt.add(pid);continue
        try:validate_mt_quad(p['gt_quad'],w,h)
        except (ValueError,TypeError):out['reason']='GT_SOURCE_GEOMETRY_INVALID';return out
        # A completed D/drag is already a positive geometry decision. Preserve
        # deliberate Space-unconfirmation, including standalone manual GTRs.
        if pid in flags:
            actions=[e.get('action') for e in scene.get('events',[]) if e.get('object_id')==pid
                and e.get('action') in {'operator_unconfirm_associated_frames','operator_confirm_associated_frames','operator_confirm_gtr_geometry'}]
            if actions and actions[-1]=='operator_unconfirm_associated_frames':pending_gt.add(pid);continue
        confirmed.append(pid)
    units={('GTR',pid) for pid in pending_gt}
    for oid in pending:
        link=state['links'].get(oid,{})
        pid=link.get('plate_id') if link.get('status')=='ASSOCIATED_FOR_GEOMETRY' else None
        units.add(('GTR',pid) if pid in state['plates'] else ('MT',oid))
    unresolved=len(units)+int(editing)
    empty=not decisions and not active
    status=PENDING if unresolved else NO_REFERENCE if empty else READY
    out.update(state=status,complete=not unresolved,metric_complete=status==READY and not optional_geometry,decisions=unresolved,
        pending_ids=pending,pending_gtr_ids=sorted(pending_gt),confirmed_gt_ids=confirmed,
        resolved=sum(v!=partial.UNRESOLVED for v in decisions.values()),total=len(decisions),
        correct=sum(v==partial.PLATE for v in decisions.values()),false_positive=sum(v==partial.NOT_PLATE for v in decisions.values()),
        manual_added=len(manual),plate_count=len(active),empty_eligible=empty,source_unknown=False,
        object_decisions=decisions,optional_geometry_ids=optional_geometry,
        reason='UNRESOLVED_FRAME_DECISIONS' if unresolved else NO_REFERENCE if empty else 'GT_REFERENCE_GEOMETRY_UNAVAILABLE' if optional_geometry else '')
    return out


def for_session(session,gid,scene=None,*,editing=False):
    """Validate real source/ROI/roles before calling the same pure predicate."""
    group=session.mt_invocations.get(gid)
    if not group or not group.executed or group.cancelled or group.execution_failed:
        return from_scene(None,source_reason='MT_NOT_COMPLETED')
    try:
        from . import mobile_canvas_review as core,mobile_canvas_partial_review as partial
        session.verify_source()
        s=scene if scene is not None else core.scene(session,gid,include_history=False)
        from .mobile_candidate_stacks import roles
        if any(value=='UNKNOWN' for value in roles(session,gid,s).values()):
            return from_scene(s,source_reason='SOURCE_FINAL_ROLE_UNPROVEN')
        partial.validate(session,gid,s)
        from .mobile_gt_link_contract import validate
        validate(s)
        stamp=session.path.stat();key=(stamp.st_size,stamp.st_mtime_ns,gid,s['source_binding'])
        verified=getattr(session,'_frame_review_verified',set())
        if key not in verified:
            from .mobile_canvas_resolution import verify_frame
            verify_frame(session,gid);session._frame_review_verified=verified|{key}
        return from_scene(s,editing=editing)
    except (ValueError,TypeError,KeyError,OSError) as exc:
        return from_scene(None,source_reason='SOURCE_OR_REVIEW_UNAVAILABLE: '+str(exc))
