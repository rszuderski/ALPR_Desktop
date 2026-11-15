"""Additive, source-bound shared GT; legacy children remain immutable history.

Reading never migrates. Only explicit operator edits create this contract.
Candidate diagnostics are a separate population from the existing final score.
"""
from __future__ import annotations
import copy, csv, hashlib, io, json
from pathlib import Path

KEY='candidate_stacks_v1'
HISTORY='candidate_stack_history_v1'
SCHEMA='alpr.mobile.candidate_stacks.v1'
FIELDS=('quad','gt_text','text_exclusion_reason','text_confirmed','geometry_state')
DECISION_FIELDS=('identity_decision','identity_location','identity_unresolved','separate_from_mt')

def record_history(session,gid,s,action,*,diagnostic_only=False):
    """Append-only scene snapshots: undo/redo survives closing the window."""
    from . import mobile_canvas_review as core
    previous=session.invocation_annotation(gid).get(core.KEY,{})
    if action in ('undo_stack_history','redo_stack_history'):return
    if KEY not in s and HISTORY not in previous:return
    def snapshot_scene(value):return copy.deepcopy({k:v for k,v in value.items() if k not in ('events',HISTORY,'_stack_legacy')})
    history=copy.deepcopy(previous.get(HISTORY) or dict(entries=[],head=None,redo=[]))
    entry=dict(action=action,diagnostic_only=diagnostic_only,review_revision_before=session.review.review_revision,previous=history['head'],
               before=snapshot_scene(previous or core._legacy_scene(session,gid)),after=snapshot_scene(s))
    history['entries'].append(entry);history['head']=len(history['entries'])-1;history['redo']=[]
    s[HISTORY]=history

def restore_history(session,gid,*,redo=False):
    from . import mobile_canvas_review as core
    live=core._legacy_scene(session,gid);history=copy.deepcopy(live.get(HISTORY))
    if not history:return False
    if redo:
        if not history['redo']:return False
        index=history['redo'].pop();entry=history['entries'][index]
        target=copy.deepcopy(entry['after']);history['head']=index
    else:
        index=history['head']
        if index is None:return False
        entry=history['entries'][index];target=copy.deepcopy(entry['before'])
        history['head']=entry['previous'];history['redo'].append(index)
    if target.get('source_binding')!=live['source_binding'] or target.get('parents')!=live['parents']:
        raise ValueError('Historia wspólnej GT dotyczy innego źródła.')
    target[HISTORY]=history;target['events']=copy.deepcopy(live['events'])
    # Undoing the very first Canvas annotation restores an empty canonical
    # registry, not the old provisional MT-as-GT counter. This is an explicit
    # history action; passive legacy reads still remain byte-for-byte unchanged.
    if KEY not in target:target[KEY]=_state(target)
    if entry.get('diagnostic_only'):target['_stack_diag_only']=True
    core._store(session,gid,project(session,gid,target),'redo_stack_history' if redo else 'undo_stack_history')
    return True

def _id(s,oid):
    from .mobile_canvas_physical_gt import _gt_id
    return _gt_id(s,oid)

def roles(session,gid,s):
    from .mobile_mt_postdedup import canvas_post_nms_proof,final_ids,suppressed_ids
    from .mobile_canvas_physical_gt import require_final
    result={p['attempt_id']:'UNKNOWN' for p in s['parents']}
    proof=canvas_post_nms_proof(session,gid)
    if proof:
        for aid in final_ids(proof):result[aid]='FINAL'
        for aid in suppressed_ids(proof):result[aid]='SUPPRESSED'
    elif s['parents']:
        try:require_final(session,gid,[o for o in s['objects'] if o.get('parent_id')])
        except (ValueError,KeyError,TypeError):
            # The existing BBOX identity path proves final membership even in
            # older traces without raw counters. It requires a unique trace,
            # exact after-dedup count, epoch/transform, source frame and ROI.
            from .mobile_canvas_resolution import model_location
            for obj in s['objects']:
                parent=next((p for p in s['parents'] if p['parent_id']==obj.get('parent_id')),None)
                if parent and parent['quad'] is None:
                    try:model_location(session,gid,s,obj)
                    except (ValueError,KeyError,TypeError):continue
                    result[parent['attempt_id']]='FINAL'
        else:result={aid:'FINAL' for aid in result}
    return result

def _plate(s,obj,pid):
    return dict(plate_id=pid,origin_object_id=obj['object_id'],primary_object_id=obj['object_id'] if obj.get('parent_id') else None,
        gt_quad=copy.deepcopy(obj['quad']),gt_text=obj['gt_text'],text_exclusion_reason=obj['text_exclusion_reason'],
        text_confirmed=obj['text_confirmed'],geometry_state=obj['geometry_state'],
        confirmed=bool(s['checked'] or obj['geometry_state']!='unconfirmed' or obj.get('canvas_reference_decision')))

def _legacy(s):
    plates={};links={};conflicts=[];reg=s.get('physical_gt_v1') or {}
    objects={o['object_id']:o for o in s['objects']}
    for p in reg.get('plates',[]):
        obj=objects[p['reference_object_id']];plates[p['gt_id']]=_plate(s,obj,p['gt_id'])
        plates[p['gt_id']]['scene_identity']=p.get('scene_identity','DISTINCT')
    for o in s['objects']:
        oid=o['object_id'];pid=reg.get('associations',{}).get(oid)
        if o.get('post_nms_suppressed') or o['error']:
            links[oid]=dict(plate_id=None,status='UNASSOCIATED_NO_PLATE' if o['error'] else 'UNRESOLVED',provenance='legacy')
            continue
        if o.get('parent_id') and not o.get('quad') and not pid:
            links[oid]=dict(plate_id=None,status='UNRESOLVED',provenance='legacy_missing_reference')
            continue
        if o.get('parent_id') and not pid and not s['checked'] and o['geometry_state']=='unconfirmed' and not o.get('canvas_reference_decision'):
            links[oid]=dict(plate_id=None,status='UNRESOLVED',provenance='unconfirmed_candidate_not_gt')
            continue
        if not pid and o.get('duplicate_of'):
            target=objects.get(o['duplicate_of']);pid=_id(s,target['object_id']) if target else None
        if not pid:pid=_id(s,oid)
        if pid not in plates:plates[pid]=_plate(s,o,pid)
        elif plates[pid]['origin_object_id']!=oid:
            p=plates[pid]
            different_text=o['text_confirmed'] and p['text_confirmed'] and (o['gt_text'],o['text_exclusion_reason'])!=(p['gt_text'],p['text_exclusion_reason'])
            different_quad=o['geometry_state']!='unconfirmed' and p['geometry_state']!='unconfirmed' and o['quad']!=p['gt_quad']
            if different_text or different_quad:conflicts.append(dict(plate_id=pid,object_ids=[p['origin_object_id'],oid],reason='conflicting_legacy_reference'))
        links[oid]=dict(plate_id=pid,status='ASSOCIATED_FOR_GEOMETRY',provenance='legacy_reference')
    for oid,link in links.items():
        for key in DECISION_FIELDS:
            if key in objects[oid]:link[key]=copy.deepcopy(objects[oid][key])
    return dict(schema=SCHEMA,source_binding=s['source_binding'],plates=plates,links=links,conflicts=conflicts)

def snapshot(session,gid,s=None):
    from . import mobile_canvas_review as core
    s=s or core._legacy_scene(session,gid,include_history=False)
    state=copy.deepcopy(s.get(KEY) or _legacy(s))
    from .mobile_reference_disposition import withdrawn_ids
    withdrawn_ids(s,state,session_id=session.session_id)
    from .mobile_reference_lifecycle import validate_retired
    validate_retired(s,state)
    if state.get('schema')!=SCHEMA or state.get('source_binding')!=s['source_binding']:
        raise ValueError('Stos GT dotyczy innego źródła/wywołania.')
    states=roles(session,gid,s);candidates=[]
    from .mobile_canvas_partial_review import KEY as PARTIAL_KEY, active_plate_ids
    partial = s.get(PARTIAL_KEY)
    for parent in sorted(s['parents'],key=lambda p:(int(p['detection_index']),p['attempt_id'])):
        oid=next(o['object_id'] for o in s['objects'] if o.get('parent_id')==parent['parent_id'])
        link=state['links'].get(oid,{})
        if link.get('status') not in {'ASSOCIATED_FOR_GEOMETRY','UNASSOCIATED_NO_PLATE','UNRESOLVED'}:
            raise ValueError('Nieznany stan powiązania kandydata.')
        if bool(link.get('plate_id')) != (link['status']=='ASSOCIATED_FOR_GEOMETRY'):
            raise ValueError('Niespójna tożsamość kandydata i wspólnej GT.')
        if link.get('plate_id') and link['plate_id'] not in state['plates']:raise ValueError('Osierocone powiązanie kandydata.')
        row=session.attempts[parent['attempt_id']]
        candidates.append(dict(object_id=oid,attempt_id=parent['attempt_id'],mt_detection_index=int(parent['detection_index']),
            role=states[parent['attempt_id']],plate_id=link.get('plate_id'),association_status=link.get('status','UNRESOLVED'),
            mt_quad=copy.deepcopy(parent['quad']),mz_prediction=parent['mz_prediction'],mz_status=parent['mz_status'],mz_ran=parent['mz_ran'],
            score=row.get('mt_confidence') or row.get('plate_confidence'),source_pose_status=row.get('mt_pose_status'),source_mt_status=row.get('mt_status'),
            bbox={k:row.get(k) for k in ('plate_left','plate_top','plate_right','plate_bottom')}))
        if partial and oid in partial['decisions']:
            decision = partial['decisions'][oid]
            candidates[-1]['operator_decision'] = decision
            if decision != 'PLATE':
                candidates[-1].update(plate_id=None, association_status=(
                    'UNASSOCIATED_NO_PLATE' if decision == 'NOT_PLATE' else 'UNRESOLVED'))
    state['active_plate_ids'] = sorted(active_plate_ids(s, state))
    state.update(candidates=candidates,session_id=session.session_id,mt_invocation_id=gid,source_sha=session.review.source_archive_sha256,
                 image_entry=s['image_entry'],image_size=s['image_size'])
    if s.get('scope') == 'SCENE_FRAME':
        state.update(scope='SCENE_FRAME', scene_event_id=gid, mt_invocation_id=None)
    return state

def _state(s):
    if s.get('_post_nms_pending'):raise ValueError('Nierozstrzygnięty zapis legacy RAW; zachowano historyczną ocenę bez konwersji.')
    state=copy.deepcopy(s.get(KEY) or _legacy(s))
    if state['conflicts']:raise ValueError('Konflikt historycznych GT: zachowano dawną ocenę; nie wykonano konwersji.')
    return state


def adopt_canvas_write(session,gid,s):
    """First explicit edit only; preserve legacy evidence, never infer GT from checked."""
    from . import mobile_canvas_review as core
    if KEY in s:return project(session,gid,pack(s))
    previous=core._legacy_scene(session,gid)
    source=copy.deepcopy(s)
    # A newly set inspection flag is not evidence about any candidate.
    source['checked']=previous['checked']
    state=_state(source)
    s[KEY]=state
    return project(session,gid,s)


def record_scene_identity(session,gid,s):
    """Preserve the existing cross-call ambiguity gate, independently of count UI."""
    from .mobile_canvas_physical_gt import scope
    from .mobile_canvas_operator_view import eligible_physical_plate_ids
    from .mobile_reference_disposition import withdrawn_ids
    current=scope(session,gid)
    prior=any(other_gid!=gid and scope(session,other_gid)==current and
              (bool(eligible_physical_plate_ids(value['canvas_review_v1'],value['canvas_review_v1'][KEY])) if value.get('canvas_review_v1',{}).get(KEY) else
               bool((value.get('canvas_review_v1',{}).get('physical_gt_v1') or {}).get('plates')))
              for other_gid,value in session.review.invocation_annotations.items() if other_gid in session.mt_invocations)
    withdrawn=withdrawn_ids(s,s[KEY])
    for pid,plate in s[KEY]['plates'].items():
        if pid in withdrawn:continue
        plate.setdefault('scene_identity','UNRESOLVED_ACROSS_INVOCATIONS' if prior else 'DISTINCT')


def physical_count_view(s):
    """Known physical records, independent of image completion/count confirmation."""
    state=s.get(KEY) or _legacy(s)
    from .mobile_canvas_operator_view import eligible_physical_plate_ids
    n=len(eligible_physical_plate_ids(s,state))
    known=bool(n or (s.get('checked') and all(o['error'] or o.get('post_nms_suppressed') for o in s['objects'])))
    return n,known

def project(session,gid,s):
    """Effective view only for explicitly created additive state."""
    from . import mobile_canvas_review as core
    from .mobile_canvas_physical_gt import scope
    if KEY not in s:return s
    v=snapshot(session,gid,s);state=s[KEY]
    from .mobile_canvas_partial_review import KEY as PARTIAL_KEY, active_plate_ids
    active = active_plate_ids(s, state)
    registered = ({pid for pid in active if state['plates'][pid]['gt_quad']}
                  if s.get(PARTIAL_KEY) else active)
    partial = s.get(PARTIAL_KEY)
    if v['conflicts']:raise ValueError('Konflikt zapisanej wspólnej referencji.')
    for pid,p in state['plates'].items():
        if pid!=p.get('plate_id') or not pid.startswith('PGT:'):raise ValueError('Nieprawidłowe ID wspólnej GT.')
        if p['gt_quad'] is not None:core.validate_mt_quad(p['gt_quad'],*s['image_size'])
        if (not isinstance(p['gt_text'],str) or len(p['gt_text'])>256 or
            not isinstance(p['text_exclusion_reason'],str) or len(p['text_exclusion_reason'])>240 or
            (p['gt_text'] and p['text_exclusion_reason'])):raise ValueError('Niespójny tekst wspólnej GT.')
    out=copy.deepcopy(s)
    out['_stack_legacy']=dict(objects=copy.deepcopy(s['objects']),physical=copy.deepcopy(s.get('physical_gt_v1')))
    objects={o['object_id']:o for o in out['objects']};reg=dict(scope=scope(session,gid),plates=[],associations={},confirmed=bool(s['checked']),decision_source='canvas_objects')
    owned_manual={p['origin_object_id'] for p in state['plates'].values() if not objects.get(p['origin_object_id'],{}).get('parent_id')}
    out['objects']=[o for o in out['objects'] if o['object_id'] not in owned_manual and o['object_id'] not in state.get('removed_manual_ids',[])]
    for oid,link in state['links'].items():
        if oid not in objects:raise ValueError('Nieznany kandydat w powiązaniu GT.')
        obj=objects[oid]
        if not obj.get('parent_id'):continue
        for key in DECISION_FIELDS:
            if key in link:obj[key]=copy.deepcopy(link[key])
            elif key in obj:obj.pop(key)
        pid=link.get('plate_id')
        if link['status']=='UNASSOCIATED_NO_PLATE':
            obj['error']=True;obj.pop('identity_unresolved',None)
            previous=state['plates'].get((link.get('previous_link') or {}).get('plate_id'))
            if previous:
                for key in ('gt_text','text_exclusion_reason','text_confirmed'):obj[key]=previous[key]
        elif link['status']=='UNRESOLVED':obj['error']=False;obj.pop('duplicate_of',None)
        if pid:
            p=state['plates'][pid]
            obj.update(quad=copy.deepcopy(p['gt_quad']),gt_text=p['gt_text'],text_exclusion_reason=p['text_exclusion_reason'],
                       text_confirmed=p['text_confirmed'],geometry_state=p['geometry_state'],error=False)
            if link.get('identity_decision'):
                obj.update(identity_decision=copy.deepcopy(link['identity_decision']),identity_location=copy.deepcopy(link['identity_location']))
                obj.pop('identity_unresolved',None)
            if (not obj.get('post_nms_suppressed') and pid in registered
                    and (not partial or partial['decisions'].get(oid) == 'PLATE')):
                reg['associations'][oid]=pid
                primary=p.get('primary_object_id')
                primary_active=primary and state['links'].get(primary,{}).get('plate_id')==pid
                if not primary_active:
                    obj['error']=True;obj.pop('duplicate_of',None)
                elif primary!=oid:obj['duplicate_of']=primary
                else:obj.pop('duplicate_of',None)
            elif partial:
                obj.pop('duplicate_of', None)
        elif link['status']=='UNASSOCIATED_NO_PLATE':obj.pop('duplicate_of',None)
    for pid,p in state['plates'].items():
        if pid not in registered:
            continue
        primary=p.get('primary_object_id')
        if primary and state['links'].get(primary,{}).get('plate_id')==pid:
            ref=primary
        else:
            ref=p['origin_object_id'] if p['origin_object_id'] in owned_manual else pid+':reference'
            obj=core._new_object(ref,quad=p['gt_quad'])
            obj.update(gt_text=p['gt_text'],text_exclusion_reason=p['text_exclusion_reason'],text_confirmed=p['text_confirmed'],geometry_state=p['geometry_state'])
            for key in DECISION_FIELDS:
                if key in state['links'].get(p['origin_object_id'],{}):obj[key]=copy.deepcopy(state['links'][p['origin_object_id']][key])
            out['objects'].append(obj);reg['associations'][ref]=pid
        reg['plates'].append(dict(gt_id=pid,reference_object_id=ref,quad=copy.deepcopy(p['gt_quad']),scene_identity=p.get('scene_identity','DISTINCT')))
    out['physical_gt_v1']=reg
    return out

def pack(s):
    """Capture an explicit edit; restore legacy children before serialization."""
    if '_stack_legacy' not in s:return s
    state=s[KEY];objects={o['object_id']:o for o in s['objects']}
    from .mobile_reference_disposition import withdrawn_ids
    withdrawn=withdrawn_ids(s,state)
    from .mobile_canvas_partial_review import KEY as PARTIAL_KEY
    if s.get('roi_presence')=='NO_PLATES' and PARTIAL_KEY not in s:
        state['removed_manual_ids']=list(set(state.get('removed_manual_ids',[]))|{o['object_id'] for o in s['_stack_legacy']['objects'] if not o.get('parent_id')})
        state['plates']={}
        state['links']={oid:(link if objects.get(oid,{}).get('post_nms_suppressed') else
                            dict(plate_id=None,status='UNASSOCIATED_NO_PLATE',provenance='operator_explicit_negative_roi',previous_link=link))
                        for oid,link in state['links'].items() if objects.get(oid,{}).get('parent_id')}
    legacy_ids={o['object_id'] for o in s['_stack_legacy']['objects']}
    virtual={pid+':reference' for pid in state['plates']}
    for oid,obj in objects.items():
        if oid in state['links']:
            for key in DECISION_FIELDS:
                if key in obj:state['links'][oid][key]=copy.deepcopy(obj[key])
                else:state['links'][oid].pop(key,None)
        if oid not in legacy_ids and oid not in virtual and not obj.get('parent_id'):
            pid=_id(s,oid);state['plates'][pid]=_plate(s,obj,pid)
            state['links'][oid]=dict(plate_id=pid,status='ASSOCIATED_FOR_GEOMETRY',provenance='operator_manual')
            s['_stack_legacy']['objects'].append(copy.deepcopy(obj))
    retained={p['gt_id'] for p in s['physical_gt_v1']['plates']}
    for pid,p in list(state['plates'].items()):
        if pid in withdrawn:continue
        if pid not in retained and p['origin_object_id'] not in objects:
            state.setdefault('removed_manual_ids',[]).append(p['origin_object_id'])
            state['links'].pop(p['origin_object_id'],None);state['plates'].pop(pid)
    for pid,p in state['plates'].items():
        ref=next((r['reference_object_id'] for r in s['physical_gt_v1']['plates'] if r['gt_id']==pid),None)
        obj=objects.get(ref)
        if obj:
            p.update(gt_quad=copy.deepcopy(obj['quad']),gt_text=obj['gt_text'],text_exclusion_reason=obj['text_exclusion_reason'],text_confirmed=obj['text_confirmed'],geometry_state=obj['geometry_state'],confirmed=p['confirmed'] or obj['geometry_state']!='unconfirmed')
    legacy=s.pop('_stack_legacy');s['objects']=legacy['objects']
    if legacy['physical'] is None:s.pop('physical_gt_v1',None)
    else:s['physical_gt_v1']=legacy['physical']
    return s

def _write(session,gid,s,state,action,oid,*,keep_inspected=False,object_before=None):
    from . import mobile_canvas_review as core
    from .mobile_canvas_resolution import verify_frame
    verify_frame(session,gid)
    obj=next((o for o in s['objects'] if o['object_id']==oid),{})
    diagnostic_only=(action in {'operator_associate_candidate_shared_gt','operator_candidate_no_plate'}
                     and roles(session,gid,s).get(obj.get('attempt_id'))=='SUPPRESSED')
    s=copy.deepcopy(s);s[KEY]=state
    if diagnostic_only:s['_stack_diag_only']=True
    else:s.update(checked=s['checked'] if keep_inspected else False,exclusion_reason='')
    if any(link.get('identity_decision') for link in state['links'].values()):s['decision_contract']=core.DECISION_CONTRACT
    if not diagnostic_only:s.pop('roi_presence',None)
    core._store(session,gid,project(session,gid,s),action,oid,object_before=object_before)

def associate(session,gid,oid,target_id,*,keep_inspected=False,manual_replacement=None):
    from . import mobile_canvas_review as core
    from .mobile_canvas_resolution import model_location,verify_frame,identity_evidence
    s=core._legacy_scene(session,gid);state=_state(s);objects={o['object_id']:o for o in s['objects']}
    pid=target_id if target_id in state['plates'] else state['links'].get(target_id,{}).get('plate_id')
    if not pid:raise ValueError('Wskaż istniejącą referencję GT.')
    if state['links'].get(oid,{}).get('plate_id')==pid:return False
    from .mobile_gt_link_contract import require_unoccupied
    require_unoccupied(s,oid,pid)
    p=state['plates'][pid]
    if not p['confirmed'] or not p['gt_quad']:raise ValueError('Najpierw potwierdź lub popraw referencję GT.')
    candidate=objects.get(oid)
    if not candidate or not candidate.get('parent_id'):raise ValueError('Wybierz źródłowy kandydat MT.')
    verify_frame(session,gid)
    parent=next(x for x in s['parents'] if x['parent_id']==candidate['parent_id'])
    role=roles(session,gid,s)[candidate['attempt_id']]
    if role=='UNKNOWN':raise ValueError('Nie udowodniono roli RAW/FINAL; zachowano dotychczasową ocenę.')
    if parent['quad']:
        core.validate_mt_quad(parent['quad'],*s['image_size'])
        location=dict(kind='POSE',quad=copy.deepcopy(parent['quad']))
    else:location=model_location(session,gid,s,candidate)
    previous=state['links'].get(oid,{}).get('plate_id')
    if previous and previous!=pid:
        old=state['plates'][previous]
        if old['confirmed'] and manual_replacement is None:
            raise ValueError('Konflikt: kandydat ma już potwierdzoną inną GT; zachowano obie referencje.')
        if manual_replacement is not None:
            # Existing explicit "same plate" action replaces the geometry of
            # the named reference. Previous text/quad remain in audit history.
            if old['text_confirmed']:
                for key in ('gt_text','text_exclusion_reason','text_confirmed'):p[key]=old[key]
            for linked in state['links'].values():
                if linked.get('plate_id')==previous:linked['plate_id']=pid
        state['plates'].pop(previous)
    link=dict(plate_id=pid,status='ASSOCIATED_FOR_GEOMETRY',provenance='operator_canvas_association')
    if not next(x for x in s['parents'] if x['parent_id']==candidate['parent_id'])['quad']:
        link.update(identity_decision=identity_evidence(session,gid,s,candidate),identity_location=location)
    state['links'][oid]=link
    if not p.get('primary_object_id') and roles(session,gid,s)[candidate['attempt_id']]=='FINAL':p['primary_object_id']=oid
    _write(session,gid,s,state,'operator_link_manual_to_mt' if manual_replacement is not None else 'operator_associate_candidate_shared_gt',oid,
           keep_inspected=keep_inspected,object_before=manual_replacement)

def accept_reference(session,gid,oid):
    """Explicit Canvas acceptance of an unchanged final Pose, never on view."""
    from . import mobile_canvas_review as core
    s=core._legacy_scene(session,gid);state=_state(s);obj=core._object(s,oid)
    if obj.get('error') or not obj.get('parent_id') or roles(session,gid,s).get(obj['attempt_id'])!='FINAL':
        raise ValueError('Referencja wymaga jawnie wybranego końcowego MT z poprawnym Pose; dla brakującej tablicy użyj D.')
    core.validate_mt_quad(obj['quad'],*s['image_size'])
    parent=next(p for p in s['parents'] if p['parent_id']==obj['parent_id'])
    if obj['quad']!=parent['quad']:raise ValueError('To nie jest niezmieniony wynik MT; zachowano wcześniejszą korektę GT.')
    if state['links'].get(oid,{}).get('plate_id'):return
    from .mobile_manual_frame_guard import require_free_manual_area
    require_free_manual_area(s,parent['quad'],session=session,gid=gid)
    pid=_id(s,oid);p=_plate(s,obj,pid);p.update(confirmed=True,geometry_state='operator_confirmed_unchanged')
    from .mobile_reference_lifecycle import mark_model
    mark_model(s,p,oid)
    state['plates'][pid]=p;state['links'][oid]=dict(plate_id=pid,status='ASSOCIATED_FOR_GEOMETRY',provenance='operator_accept_unchanged_final_reference')
    _write(session,gid,s,state,'operator_accept_shared_gt_reference',oid)

def complete_bbox_reference(session,gid,oid,points,*,expected_plate_id=None,operator_draw=False):
    """Operator drawing on a selected BBOX fills that same physical GT once.

    Revalidate the selected identity at save time. The operator's four points
    are a reference only; neither BBOX nor absent source Pose is rewritten.
    """
    from . import mobile_canvas_review as core, mobile_canvas_partial_review as partial
    from .mobile_canvas_resolution import model_location
    s=core._legacy_scene(session,gid);state=_state(s);obj=core._object(s,oid)
    parent=next((p for p in s['parents'] if p['parent_id']==obj.get('parent_id')),None)
    link=state['links'].get(oid,{})
    pid=link.get('plate_id');plate=state['plates'].get(pid)
    removed=bool(plate and plate['geometry_state']=='operator_removed')
    if (parent is None or (parent['quad'] is not None and not removed) or obj.get('error')
            or roles(session,gid,s).get(obj.get('attempt_id'))!='FINAL'
            or (partial.progress(s)['object_decisions'].get(oid)!=partial.PLATE and not operator_draw)
            or link.get('status')!='ASSOCIATED_FOR_GEOMETRY' or not plate
            or not plate['confirmed'] or plate['gt_quad'] is not None
            or (expected_plate_id is not None and pid!=expected_plate_id)):
        raise ValueError('Wskazanie zmieniło stan lub ma już obrys GT; zachowano zapisane dane. Wybierz MT bez obrysu operatora i użyj D.')
    if model_location(session,gid,s,obj)['kind']!='BBOX' and not removed:
        raise ValueError('Brak wiarygodnego źródłowego BBOX MT.')
    quad=core.ordered_quad(points,s['image_size'],s['roi'])
    from .mobile_manual_frame_guard import require_free_manual_area
    require_free_manual_area(s,quad,session=session,gid=gid)
    edit_gt(session,gid,oid,quad=quad)
    return oid


def redraw_reference(session,gid,target,points):
    """One geometry edit of the explicitly selected, unchanged physical GT."""
    from . import mobile_canvas_review as core
    from .mobile_canvas_operator_view import saved_reference_ids
    from .mobile_manual_frame_guard import require_free_manual_area
    from .mobile_canvas_resolution import verify_frame
    s=core._legacy_scene(session,gid);state=_state(s)
    pid=target['plate_id'];p=state['plates'].get(pid);oid=target['oid']
    if (target['gid']!=gid or target['binding']!=s['source_binding']
            or target['revision']!=session.review.review_revision or not p
            or pid not in saved_reference_ids(s,state) or p['gt_quad']!=target['quad']
            or p['gt_quad'] is None):
        raise ValueError('Wybrana GTR lub źródło zmieniły się. Anuluj szkic; zachowano zapisany obrys.')
    live_pid=state['links'].get(oid,{}).get('plate_id')
    if live_pid!=pid and oid not in (p['origin_object_id'],pid+':reference'):
        raise ValueError('Zmieniono powiązanie wybranej GTR; niczego nie zapisano.')
    verify_frame(session,gid)
    quad=core.ordered_quad(points,s['image_size'],s['roi'])
    require_free_manual_area(s,quad,exclude_plate_id=pid,session=session,gid=gid)
    edit_gt(session,gid,oid,quad=quad)
    return oid


def edit_gt(session,gid,oid,*,quad=None,text=None,reason=''):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import edit
        return edit(session, gid, oid, quad=quad, text=text, reason=reason)
    from . import mobile_canvas_review as core
    from .mobile_human_review import normalize_registration
    s=core._legacy_scene(session,gid);state=_state(s)
    from .mobile_reference_disposition import require_active
    require_active(s,plate_id=oid if oid in state['plates'] else None,object_id=oid)
    pid=state['links'].get(oid,{}).get('plate_id') or next((pid for pid,p in state['plates'].items() if oid in (p['origin_object_id'],pid+':reference')),None)
    if quad is not None:
        from .mobile_manual_frame_guard import require_free_manual_area
        require_free_manual_area(s,quad,exclude_plate_id=pid,session=session,gid=gid)
    if not pid:
        obj=core._object(s,oid)
        if obj.get('error') or obj.get('post_nms_suppressed') or s.get('_post_nms_unknown'):
            raise ValueError('Nie można ustanowić GT z FP, odrzuconego RAW ani nieustalonego wyniku MT.')
        reference=quad if quad is not None else obj.get('quad')
        core.validate_mt_quad(reference,*s['image_size'])
        if quad is None and not (text or reason):raise ValueError('Brak jawnej referencji GT.')
        pid=_id(s,oid);p=_plate(s,obj,pid)
        p.update(gt_quad=copy.deepcopy(reference),confirmed=True,geometry_state='unconfirmed')
        state['plates'][pid]=p
        state['links'][oid]=dict(plate_id=pid,status='ASSOCIATED_FOR_GEOMETRY',provenance='operator_canvas_reference')
        if obj.get('parent_id') and not next(p for p in s['parents'] if p['parent_id']==obj['parent_id'])['quad']:
            from .mobile_canvas_resolution import model_location,identity_evidence
            state['links'][oid].update(identity_location=model_location(session,gid,s,obj),identity_decision=identity_evidence(session,gid,s,obj))
    p=state['plates'][pid]
    if quad is not None:
        p['gt_quad']=[list(v) for v in core.validate_mt_quad(quad,*s['image_size'])];p['geometry_state']='operator_corrected';p['confirmed']=True
        from .mobile_reference_lifecycle import mark_independent
        mark_independent(p,'OPERATOR_CORNERS')
        obj=next((o for o in s['objects'] if o['object_id']==oid),None)
        if obj and not obj.get('parent_id'):
            l,t,r,b=s['roi']
            if any(not(l<=x<=r and t<=y<=b) for x,y in p['gt_quad']):raise ValueError('Ręczna tablica poza analizowanym ROI')
    if text is not None:
        text=normalize_registration(text);reason=' '.join(str(reason).split())
        if len(text)>256 or len(reason)>240 or (text and reason):raise ValueError('Wpisz numer albo przyczynę nieczytelności.')
        p.update(gt_text=text,text_exclusion_reason=reason,text_confirmed=bool(text or reason))
        if text or reason:
            from .mobile_reference_lifecycle import mark_independent
            mark_independent(p,'OPERATOR_GTZ')
    _write(session,gid,s,state,'operator_edit_shared_gt',oid)

def toggle_unassociated(session,gid,oid,*,partial_review=False):
    from . import mobile_canvas_review as core
    core.require_source_final(session,gid,oid)
    s=core._legacy_scene(session,gid);state=_state(s);link=state['links'].get(oid)
    from .mobile_canvas_partial_review import enable
    enable(session, gid, s, state)
    if not link:raise ValueError('Nieznany kandydat.')
    obj=core._object(s,oid)
    parent=next(p for p in s['parents'] if p['parent_id']==obj['parent_id'])
    if link['status']!='UNASSOCIATED_NO_PLATE' and not parent.get('quad'):
        from .mobile_canvas_resolution import model_location
        model_location(session,gid,s,obj)  # no spatial FP without source location
    if link['status']=='UNASSOCIATED_NO_PLATE':
        from .mobile_reference_lifecycle import restore
        restore(state,link)
        if link.get('previous_link'):state['links'][oid]=link['previous_link']
        else:state['links'][oid]=dict(plate_id=None,status='UNRESOLVED',provenance='operator_canvas')
    else:
        state['links'][oid]=dict(plate_id=None,status='UNASSOCIATED_NO_PLATE',provenance='operator_canvas',previous_link=copy.deepcopy(link))
        from .mobile_reference_lifecycle import retire
        retire(s,state,oid)
    _write(session,gid,s,state,'operator_candidate_no_plate',oid)


def remove_linked_geometry(session,gid,oid):
    """Remove operator corners, retaining identity/GTZ and immutable MT source."""
    from . import mobile_canvas_review as core,mobile_canvas_partial_review as partial
    from .mobile_canvas_operator_view import reference_numbers
    s=core._legacy_scene(session,gid);state=_state(s);pid=state['links'].get(oid,{}).get('plate_id')
    p=state['plates'].get(pid)
    linked=[key for key,link in state['links'].items() if link.get('plate_id')==pid]
    if not p or p['origin_object_id']!=oid or len(linked)!=1:
        raise ValueError('R nie usuwa ani nie zastępuje wspólnej GT; użyj korekty lub Undo.')
    core.require_source_final(session,gid,oid)
    if p['gt_quad'] is None:return False
    before=copy.deepcopy(p)
    state['reference_numbers']=reference_numbers(state)
    partial.enable(session,gid,s,state)
    p.update(gt_quad=None,geometry_state='operator_removed')
    _write(session,gid,s,state,'remove_operator_reference_geometry',oid,object_before=before)
    return 'remove_operator_reference_geometry'

def confirm_negative(session,gid):
    from . import mobile_canvas_review as core
    from .mobile_canvas_resolution import verify_frame,identity_evidence
    verify_frame(session,gid);effective=core.scene(session,gid)
    if any(not o.get('parent_id') and not o['error'] for o in effective['objects']):
        raise ValueError('Pozostaje ręczna tablica; najpierw rozstrzygnij jej obecność.')
    for obj in effective['objects']:
        if obj.get('parent_id') and not obj.get('post_nms_suppressed'):identity_evidence(session,gid,effective,obj)
    s=core._legacy_scene(session,gid);state=_state(s)
    state['plates']={}
    objects={o['object_id']:o for o in s['objects']}
    state['links']={oid:(link if objects[oid].get('post_nms_suppressed') else
                        dict(plate_id=None,status='UNASSOCIATED_NO_PLATE',provenance='operator_explicit_negative_roi',previous_link=link))
                    for oid,link in state['links'].items()}
    s[KEY]=state;s.update(checked=True,exclusion_reason='',roi_presence='NO_PLATES',decision_contract=core.DECISION_CONTRACT)
    core._store(session,gid,project(session,gid,s),'operator_confirm_no_plates_in_roi')

def diagnostics(session, invocation_ids=None):
    from . import mobile_canvas_review as core
    from .mobile_pose_metrics import compare_mt_pose_to_gt
    from .mobile_human_review import align_plate_text
    rows=[];plates=set();calls=set();conflicts=[];unavailable=[]
    for gid in (session.mt_invocations if invocation_ids is None else invocation_ids):
        try:
            s=core.scene(session,gid,include_history=False);raw=core._legacy_scene(session,gid,include_history=False);v=snapshot(session,gid,raw)
        except (ValueError,KeyError,TypeError) as error:
            unavailable.append(dict(mt_invocation_id=gid,reason=str(error),attempt_ids=[r.get('id') or r.get('attempt_id') for r in session.mt_invocations[gid].records]))
            continue
        conflicts.extend(dict(mt_invocation_id=gid,**c) for c in v['conflicts'])
        for pid in v['plates']:plates.add((gid,pid))
        calls.add(gid)
        for c in v['candidates']:
            p=v['plates'].get(c['plate_id']);conflict=any(x['plate_id']==c['plate_id'] for x in v['conflicts'])
            reason='UNASSOCIATED' if not p else 'LEGACY_CONFLICT' if conflict else 'GT_GEOMETRY_REMOVED' if p['geometry_state']=='operator_removed' else 'GT_UNCONFIRMED' if not p['confirmed'] or p['geometry_state']=='unconfirmed' else 'MT_POSE_NOT_AVAILABLE' if not c['mt_quad'] else ''
            if reason=='MT_POSE_NOT_AVAILABLE':
                reason='MT_POSE_INVALID' if c['source_pose_status']=='INVALID_QUAD' or c['source_mt_status']=='DETECTION_INVALID_QUAD' else 'MT_POSE_EVIDENCE_UNAVAILABLE'
            if s.get('exclusion_reason'):reason='SCENE_EXCLUDED'
            if conflict:p=None  # No arbitrary canonical text/quad in new diagnostics.
            legacy=next(o for o in raw['objects'] if o['object_id']==c['object_id'])
            row=dict(session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,review_revision=session.review.review_revision,
                source_image=s['image_entry'],mt_invocation_id=gid,plate_id=c['plate_id'],shared_gt_id=c['plate_id'],**{k:v for k,v in c.items() if k!='plate_id'},
                coordinate_space='source_frame_pixels',source_image_size=list(s['image_size']),roi=list(s['roi']),source_binding=s['source_binding'],
                gt_reference_provenance='canonical_physical_gt' if KEY in raw else 'legacy_reference',
                legacy_child_correction=copy.deepcopy(legacy['quad']),legacy_gt_text=legacy['gt_text'],legacy_geometry_state=legacy['geometry_state'],
                gt_quad=copy.deepcopy(p['gt_quad']) if p else None,gt_text=p['gt_text'] if p else '',text_exclusion_reason=p['text_exclusion_reason'] if p else '',
                polygon_iou_pct=None,nmce_pct=None,cer=None,not_measurable_reason=reason)
            if not reason:
                try:row.update(compare_mt_pose_to_gt(c['mt_quad'],p['gt_quad'],*s['image_size']))
                except (ValueError,TypeError):row['not_measurable_reason']='INVALID_GEOMETRY'
            if p and not s.get('exclusion_reason') and p['text_confirmed'] and p['gt_text'] and c['mz_ran'] and c['mz_status']!='AMBIGUOUS_SAMPLES':row['cer']=align_plate_text(p['gt_text'],c['mz_prediction']).cer
            rows.append(row)
    return dict(schema='alpr.mobile.candidate_diagnostics.v1',provenance='ASSISTED_SHARED_GT_CANDIDATE_DIAGNOSTICS',session_id=session.session_id,
        source_archive_sha256=session.review.source_archive_sha256,review_revision=session.review.review_revision,n_unique_plates=len({pid for gid,pid in plates}),
        n_plate_invocations=len(plates),plate_unit='explicit_physical_gt_id',n_invocations=len(calls),n_source_invocations=len(session.mt_invocations),
        invocation_unit='source_bound_readable_invocations',unavailable_invocations=unavailable,n_candidates=len(rows),n_measurable=sum(not r['not_measurable_reason'] for r in rows),rows=rows,conflicts=conflicts)

def export(session,directory):
    from .mobile_gt_link_contract import validate_review
    validate_review(session,session.review)
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    for n in range(1,10000):
        dest=root/f'mt_candidates_v1_rev{session.review.review_revision}_{n:03d}'
        try:dest.mkdir()
        except FileExistsError:continue
        break
    else:raise ValueError('Brak wolnej wersji eksportu diagnostycznego.')
    data=diagnostics(session)
    (dest/'mt_candidates.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=data['rows'];out=io.StringIO(newline='');fields=list(dict.fromkeys(k for r in rows for k in r)) or ['attempt_id']
    w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
    for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    (dest/'mt_candidates.csv').write_text('\ufeff'+out.getvalue(),encoding='utf-8')
    (dest/'report.md').write_text(f"# Diagnostyka kandydatów MT — ASSISTED\n\nNie jest oceną końcową MT+MZ.\n\nUnikatowe jawne GT IDs: {data['n_unique_plates']}; obserwacje GT w wywołaniach: {data['n_plate_invocations']}.\n\nWywołania ze zweryfikowanym źródłem: {data['n_invocations']}/{data['n_source_invocations']}; kandydaci: {data['n_candidates']}; mierzalni: {data['n_measurable']}.\n\nNiedostępne wywołania: {len(data['unavailable_invocations'])}; konflikty legacy: {len(data['conflicts'])}. Przyczyny, oryginalne ID i geometrie są w JSON/CSV. Odrzucone RAW nie należą do końcowej populacji TP/FP/FN.\n",encoding='utf-8')
    manifest=dict(schema='alpr.mobile.candidate_manifest.v1',source_archive_sha256=data['source_archive_sha256'],review_revision=data['review_revision'],
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return dest
