"""Explicit retirement of a historical manual duplicate; never auto-merge by IoU."""
from __future__ import annotations
import copy
from . import mobile_candidate_stacks as stacks


def _similar(a,b):
    from .mobile_pose_metrics import _signed_area,_clip_convex
    if not a or not b:return False
    common=abs(_signed_area(_clip_convex(a,b)))
    union=abs(_signed_area(a))+abs(_signed_area(b))-common
    return bool(union>0 and common/union>=.85)


def suggestions(scene,state):
    """Geometry only proposes a choice; an operator must establish identity."""
    from .mobile_canvas_operator_view import reference_numbers,saved_reference_ids
    objects={o['object_id']:o for o in [*(scene.get('_stack_legacy') or {}).get('objects',[]),*scene['objects']]}
    saved=saved_reference_ids(scene,state);numbers=reference_numbers(state);result=[]
    finals={c['object_id'] for c in state.get('candidates',[]) if c['role']=='FINAL'}
    targets={link['plate_id'] for oid,link in state['links'].items() if oid in finals
             and link.get('status')=='ASSOCIATED_FOR_GEOMETRY' and link.get('plate_id') in saved}
    for pid in saved-targets:
        p=state['plates'][pid];oid=p['origin_object_id'];obj=objects.get(oid,{})
        if obj.get('parent_id') or obj.get('kind')!='manual' or obj.get('error'):continue
        if any(o!=oid and link.get('plate_id')==pid for o,link in state['links'].items()):continue
        separate=set(obj.get('separate_from_mt',[]))|set(state['links'].get(oid,{}).get('separate_from_mt',[]))
        matches=[]
        for target in targets:
            q=state['plates'][target]
            linked={o for o,l in state['links'].items() if l.get('plate_id')==target}
            if separate&linked:continue
            if p['text_confirmed'] and (p['gt_text'],p['text_exclusion_reason'])!=(q['gt_text'],q['text_exclusion_reason']):continue
            if _similar(p['gt_quad'],q['gt_quad']):matches.append(target)
        # Ambiguous proposals must never choose one nearby physical plate.
        if len(matches)==1:
            target=matches[0]
            result.append(dict(duplicate_id=pid,canonical_id=target,duplicate_number=numbers[pid],canonical_number=numbers[target]))
    return sorted(result,key=lambda r:r['duplicate_number'])


def merge_manual_duplicate(session,gid,duplicate_id,canonical_id,*,expected_revision):
    """Operator says 'same plate': retain canonical GT and archive the extra GTR."""
    from . import mobile_canvas_review as core
    from .mobile_canvas_operator_view import reference_numbers
    if session.review.review_revision!=expected_revision:
        raise ValueError('Ocena zmieniła się. Wybierz ponownie aktualne referencje.')
    scene=core._legacy_scene(session,gid);state=stacks._state(scene)
    allowed=suggestions(scene,stacks.snapshot(session,gid,scene))
    if not any(p['duplicate_id']==duplicate_id and p['canonical_id']==canonical_id for p in allowed):
        raise ValueError('Nie można bezpiecznie uzgodnić tych GTR. Zachowano obie referencje i oceny.')
    old=copy.deepcopy(state['plates'][duplicate_id]);oid=old['origin_object_id']
    state['reference_numbers']=reference_numbers(state)
    state.setdefault('retired_references',{})[duplicate_id]=dict(plate=old,merged_into=canonical_id,
        links={o:copy.deepcopy(l) for o,l in state['links'].items() if l.get('plate_id')==duplicate_id},
        reason='operator_same_physical_plate',review_revision_before=expected_revision)
    state['plates'].pop(duplicate_id);state['links'].pop(oid,None)
    state['removed_manual_ids']=sorted(set(state.get('removed_manual_ids',[]))|{oid})
    # Keep canonical ID/geometry/text/approval and every source decision intact.
    # The obsolete manual object's complete record remains in scene/history.
    stacks._write(session,gid,scene,state,'operator_merge_duplicate_gtr',state['plates'][canonical_id]['origin_object_id'],
                  keep_inspected=True,object_before=dict(retired_plate_id=duplicate_id,retired_plate=old,canonical_plate_id=canonical_id))
    return True
