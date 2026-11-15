"""Spatial FINAL/GTR presentation, independent of physical identity and scoring.

The 2026-10-31 operator contract supersedes explicit-link-only presentation.
No view produced here is ever passed to the scientific matcher or saved.
"""
from __future__ import annotations
import copy
from ..ranking.mobile_pose_metrics import _clip_convex,_signed_area,validate_mt_quad

POLICY='alpr.spatial_final_gtr.v1'

def area(q):return abs(_signed_area(q)) if q else 0.0
def rect(b):
    l,t,r,b=b;return [[l,t],[r,t],[r,b],[l,b]]
def bounds(q):return [min(p[0] for p in q),min(p[1] for p in q),max(p[0] for p in q),max(p[1] for p in q)]
def overlap(a,b):return area(_clip_convex(a,b)) if a and b else 0.0
def iou(a,b):
    common=overlap(a,b);union=area(a)+area(b)-common
    return common/union if union>0 else 0.0

def rank(sources,references):
    """Pure deterministic geometry; positive area is NOT the TP threshold."""
    result={}
    for source in sources:
        oid=source['object_id'];bbox=rect(source['bbox']) if source.get('bbox') else None;pose=source.get('pose')
        out=dict(plate_id=None,candidates=[],conflict=False,reason='',policy=POLICY)
        if source['role']!='FINAL':out['reason']='NOT_FINAL'
        elif not bbox and not pose:out['reason']='SOURCE_GEOMETRY_UNAVAILABLE'
        else:
            for ref in references:
                q=ref['quad'];a=overlap(q,bbox);b=overlap(q,pose)
                if a>0 or b>0:
                    out['candidates'].append(dict(plate_id=ref['plate_id'],bbox_overlap_area=a,pose_overlap_area=b,
                        bbox_iou=iou(bbox,rect(bounds(q))) if bbox else None,pose_iou=iou(pose,q) if pose else None,
                        scene_identity=ref.get('scene_identity'),pose_only_overlap=bool(b>0 and a==0)))
            out['candidates'].sort(key=lambda c:(-(c['bbox_iou'] if c['bbox_iou'] is not None else -1),
                -(c['pose_iou'] if c['pose_iou'] is not None else -1),c['plate_id']))
            if out['candidates']:
                out['plate_id']=out['candidates'][0]['plate_id'];out['conflict']=len(out['candidates'])>1
                out['reason']='MAX_BBOX_IOU_THEN_POSE_IOU_THEN_STABLE_ID' if out['conflict'] else 'POSITIVE_SOURCE_OVERLAP'
            else:out['reason']='NO_POSITIVE_OVERLAP'
        result[oid]=out
    return result

def relations(session,gid,scene=None,state=None):
    """Validate source Bitmap/ROI/FINAL evidence before geometry calculation."""
    from ..ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
    from ..ranking.mobile_canvas_resolution import verify_frame,identity_evidence
    from ..ranking.mobile_canvas_operator_view import saved_reference_ids
    from ..ranking.mobile_mt_geometry_layers import source_bbox
    result=dict(policy=POLICY,status='UNAVAILABLE',reason='',relations={},reference_errors=[])
    try:
        s=scene if scene is not None else core.scene(session,gid,include_history=False)
        expected=core.scene(session,gid,include_history=False)
        for key in ('source_archive_sha256','mt_invocation_id','image_entry','image_size','roi','source_binding','parents'):
            if s.get(key)!=expected.get(key):raise ValueError('SOURCE_FRAME_ROI_BINDING_MISMATCH: '+key)
        v=state if state is not None else stacks.snapshot(session,gid,s)
        if v.get('source_binding')!=s['source_binding']:raise ValueError('SOURCE_FRAME_ROI_BINDING_MISMATCH: state')
        stat=session.path.stat();key=(str(session.path),stat.st_size,stat.st_mtime_ns,gid,s['source_binding'])
        if getattr(session,'_spatial_verified_frame',None)!=key:
            verify_frame(session,gid);session._spatial_verified_frame=key
        actual=stacks.snapshot(session,gid,s)
        sources=[]
        for c in actual['candidates']:
            source=dict(object_id=c['object_id'],role=c['role'],bbox=None,pose=None)
            if c['role']=='FINAL':
                identity_evidence(session,gid,s,core._object(s,c['object_id']))
                source['bbox']=source_bbox(session,s,c)
                if c['mt_quad']:
                    try:source['pose']=[list(p) for p in validate_mt_quad(c['mt_quad'],*s['image_size'])]
                    except (ValueError,TypeError):pass
            sources.append(source)
        refs=[]
        for pid in saved_reference_ids(s,actual):
            p=actual['plates'][pid]
            try:q=[list(x) for x in validate_mt_quad(p['gt_quad'],*s['image_size'])]
            except (ValueError,TypeError):result['reference_errors'].append(dict(plate_id=pid,reason='INVALID_GTR'));continue
            refs.append(dict(plate_id=pid,quad=q,scene_identity=p.get('scene_identity')))
        result.update(status='VERIFIED',source_binding=s['source_binding'],image_entry=s['image_entry'],roi=s['roi'],
            sources=sources,references=refs,relations=rank(sources,refs))
    except (OSError,ValueError,KeyError,TypeError) as exc:result['reason']=str(exc)
    return result

def project(session,gid,scene,state):
    """Return a disposable view; canonical link/identity/BAD stay auditable."""
    evidence=relations(session,gid,scene,state);v=copy.deepcopy(state);v['spatial_reference']=evidence
    for c in v['candidates']:
        if c['role']!='FINAL':continue
        canonical=copy.deepcopy(state['links'].get(c['object_id'],{}))
        c['canonical_link']=canonical;c['operator_bad']=canonical.get('status')=='UNASSOCIATED_NO_PLATE'
        rel=evidence['relations'].get(c['object_id'],dict(plate_id=None,candidates=[],conflict=False,reason=evidence['reason']))
        c['spatial_relation']=rel;pid=rel['plate_id'];c['plate_id']=pid
        c['association_status']='ASSOCIATED_FOR_GEOMETRY' if pid else 'UNASSOCIATED_NO_PLATE' if c['operator_bad'] else 'UNRESOLVED'
        v['links'][c['object_id']]=dict(plate_id=pid,status=c['association_status'],provenance='READ_ONLY_SPATIAL_PROJECTION')
    return v

def view(ed,scene,state):
    try:stat=ed.h.session.path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
    except OSError:stamp=None
    key=(id(ed.h.session),ed.h.current_invocation,ed.h.session.review.review_revision,scene['source_binding'],stamp)
    if getattr(ed,'_spatial_view_key',None)!=key:
        ed._spatial_view=project(ed.h.session,ed.h.current_invocation,scene,state);ed._spatial_view_key=key
    return ed._spatial_view

def diagnostic(row):
    c=row.get('candidate') if row else None;rel=c.get('spatial_relation') if c else None
    if not rel:return ''
    text='Relacja przestrzenna GTR: '+(str(row['n']) if rel['plate_id'] else 'brak')+'; nie jest oceną TP.'
    if rel.get('conflict'):
        text+=' Konflikt przecięć: '+', '.join(p['plate_id']+' (BBOX IoU '+str(round(p['bbox_iou'] or 0,6))+')' for p in rel['candidates'])
        text+='. Wybór: największe BBOX IoU, potem Pose IoU, potem stabilne ID. Zachowano wszystkie GT.'
    elif rel.get('candidates') and rel['candidates'][0].get('pose_only_overlap'):
        text+=' Przecięcie tylko z Pose; BBOX i Pose wskazują odmienne obszary.'
    if c.get('canonical_link',{}).get('plate_id') not in (None,rel['plate_id']):
        text+=' Zapisana relacja fizyczna pozostaje bez zmian.'
    if not rel['plate_id']:text+=' '+rel.get('reason','')
    return text


def canonical_action(session,gid,oid,pid,*,operation,quad=None,text=None,reason=''):
    """One explicit operator write using existing canonical APIs and history.

    Preparation happens on a detached session. Intermediate association is
    never published; the real _store validates and writes once, or not at all.
    A stored different physical relation is never reassigned by this adapter.
    """
    from ..ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
    s=core.scene(session,gid);state=stacks.snapshot(session,gid,s)
    relation=relations(session,gid,s,state)['relations'].get(oid,{})
    if relation.get('plate_id')!=pid:raise ValueError('Relacja przestrzenna zmieniła się; wybierz aktualną GTR.')
    if operation=='edit' and quad is None:
        # A GTZ edit belongs to this physical reference. It does not approve
        # corners or a source decision merely to satisfy associate()'s guard.
        return stacks.edit_gt(session,gid,pid+':reference',text=text,reason=reason)
    old_pid=state['links'].get(oid,{}).get('plate_id')
    plate=state['plates'][pid]
    reference=next((o for o in s['objects'] if o['object_id']==plate['origin_object_id']),{})
    distinct=oid in reference.get('separate_from_mt',())
    bad=state['links'].get(oid,{}).get('status')=='UNASSOCIATED_NO_PLATE'
    if distinct or (bad and operation=='edit'):
        if operation=='edit':return stacks.edit_gt(session,gid,pid+':reference',quad=quad,text=text,reason=reason)
        return partial.toggle_stack(session,gid,plate_id=pid)
    if old_pid and old_pid!=pid:
        if operation=='edit':return stacks.edit_gt(session,gid,pid+':reference',quad=quad,text=text,reason=reason)
        return partial.toggle_stack(session,gid,plate_id=pid)
    if old_pid==pid:
        if operation=='edit':return stacks.edit_gt(session,gid,oid,quad=quad,text=text,reason=reason)
        return partial.toggle_stack(session,gid,plate_id=pid)
    staged=copy.copy(session);staged.review=copy.deepcopy(session.review)
    def capture(changed,**kwargs):
        changed.review_revision=staged.review.review_revision+1;staged.review=changed
    staged._save_change=capture
    if not state['plates'][pid]['confirmed']:partial.confirm_gtr(staged,gid,pid)
    stacks.associate(staged,gid,oid,pid)
    if operation=='edit':stacks.edit_gt(staged,gid,oid,quad=quad,text=text,reason=reason)
    # This is the first explicit association, not a request to undo it.
    # associate() already supplies a positive decision; toggling it again
    # would immediately turn this single Space into UNRESOLVED.
    else:partial.confirm_stack(staged,gid,oid=oid)
    final=core.scene(staged,gid)
    final['events']=copy.deepcopy(s['events'])
    if stacks.HISTORY in s:final[stacks.HISTORY]=copy.deepcopy(s[stacks.HISTORY])
    else:final.pop(stacks.HISTORY,None)
    final[stacks.KEY]['links'][oid]['spatial_policy']=POLICY
    action='operator_edit_shared_gt' if operation=='edit' else 'operator_confirm_associated_frames'
    core._store(session,gid,final,action,pid if operation!='edit' else oid)
    return True


def selected_relation(ed,oid=None):
    from .mobile_canvas_clean_view import selected,records
    requested=oid or ed.selected
    chosen=selected(ed)
    matches=lambda r:requested in (r['oid'],r.get('reference_id'),(r.get('plate') or {}).get('origin_object_id'))
    row=chosen if chosen and chosen.get('candidate') and matches(chosen) else next((r for r in records(ed) if matches(r) and r.get('candidate')),None)
    c=row.get('candidate') if row else None
    if row and row['gt'] and row['pid'] and c and c.get('spatial_relation',{}).get('plate_id')==row['pid']:
        return c['object_id'],row['pid']
    return None


def entry_reference(ed,oid,scene=None):
    """Resolve inline GTZ to the saved physical ID, never the source alias.

    The legacy text policy requires a persisted MT parent. A spatial view can
    already expose a valid manual reference before that link is saved. Both
    opening the field and its permission guard must use the same target.
    """
    if not hasattr(ed,'clean_bar'):return None
    from ..ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
    relation=selected_relation(ed,oid)
    if not relation:return None
    source,pid=relation;s=scene if scene is not None else core.scene(ed.h.session,ed.h.current_invocation)
    evidence=relations(ed.h.session,ed.h.current_invocation,s)
    if evidence['relations'].get(source,{}).get('plate_id')!=pid:return None
    state=stacks.snapshot(ed.h.session,ed.h.current_invocation,s)
    p=state['plates'].get(pid)
    if not p or not p['gt_quad']:return None
    ref=next((r['reference_object_id'] for r in s.get('physical_gt_v1',{}).get('plates',[]) if r['gt_id']==pid),p['origin_object_id'])
    obj=next((o for o in s['objects'] if o['object_id']==ref),None)
    if not obj or obj.get('error'):return None
    validate_mt_quad(obj['quad'],*s['image_size'])
    return dict(source_oid=source,plate_id=pid,reference_id=ref)
