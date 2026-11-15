"""Read-only BBOX/Pose/GTR projection for ONE proven source FINAL."""
from __future__ import annotations
import copy,math
from . import mobile_canvas_review as core,mobile_candidate_stacks as stacks

def source_bbox(session,scene,candidate):
    row=session.attempts[candidate['attempt_id']]
    try:
        box=[float(row[k]) for k in ('plate_left','plate_top','plate_right','plate_bottom')]
        w,h=scene['image_size'];l,t,r,b=box
        if (not all(math.isfinite(v) for v in box) or not (0<=l<r<=w and 0<=t<b<=h)
                or row.get('source_frame_coordinate_space')!='mt_source_bitmap_xy'
                or row.get('source_frame_origin')!='pipeline_bitmap_before_roi'
                or [float(row[k]) for k in ('source_frame_width_px','source_frame_height_px')]!=list(scene['image_size'])):return None
        return box
    except (KeyError,ValueError,TypeError,OverflowError):return None

def snapshot(session,gid,oid,*,scene=None,state=None):
    from .mobile_gt_link_contract import validate
    from .mobile_canvas_operator_view import linked_references
    s=scene or core.scene(session,gid);validate(s);v=state or stacks.snapshot(session,gid)
    c=next((c for c in v['candidates'] if c['object_id']==oid),None)
    if not c or c['role']!='FINAL':raise ValueError('Warstwy źródłowe wymagają potwierdzonego MT FINAL.')
    rel=linked_references(s,v,c)
    if rel['count'] is None:raise ValueError('Nie ustalono pojedynczej relacji GTR dla MT.')
    layers=[];bbox=source_bbox(session,s,c)
    if bbox:layers.append(dict(kind='BBOX',role='MT BBOX (źródło)',bbox=bbox,quad=None))
    if c['mt_quad']:
        core.validate_mt_quad(c['mt_quad'],*s['image_size'])
        layers.append(dict(kind='POSE',role='MT POSE (źródło)',quad=copy.deepcopy(c['mt_quad'])))
    if rel['ids']:
        pid=rel['ids'][0];p=v['plates'][pid]
        layers.append(dict(kind='GTR',role='GTR (operator)',plate_id=pid,quad=copy.deepcopy(p['gt_quad'])))
    return dict(source_key=[v['source_sha'],s['source_binding'],gid,c['attempt_id'],c['mt_detection_index']],
        object_id=oid,layers=layers,gtr_count=rel['count'],gtr_id=rel['ids'][0] if rel['ids'] else None,
        bbox_unavailable=bbox is None,pose_unavailable=not bool(c['mt_quad']),bad=c['association_status']=='UNASSOCIATED_NO_PLATE')

def create_reference(session,gid,oid,points,*,expected_binding=None,expected_revision=None):
    """D on an explicitly selected unlinked FINAL: one atomic GTR + link."""
    from .mobile_gt_link_contract import require_unoccupied
    from .mobile_canvas_resolution import verify_frame,identity_evidence,model_location
    from .mobile_manual_frame_guard import require_free_manual_area
    from . import mobile_canvas_partial_review as partial
    s=core._legacy_scene(session,gid);require_unoccupied(s,oid);core.require_source_final(session,gid,oid)
    if ((expected_binding is not None and expected_binding!=s['source_binding']) or
            (expected_revision is not None and expected_revision!=session.review.review_revision)):
        raise ValueError('Stan MT/GTR zmienił się podczas rysowania; nie zapisano referencji.')
    obj=core._object(s,oid)
    if obj['error']:raise ValueError('Błędne MT nie może otrzymać nowej GTR.')
    verify_frame(session,gid);location=model_location(session,gid,s,obj)
    q=core.ordered_quad(points,s['image_size'],s['roi']);require_free_manual_area(s,q,session=session,gid=gid)
    state=stacks._state(s);decisions=partial.enable(session,gid,s,state)
    pid=stacks._id(s,oid);p=stacks._plate(s,obj,pid)
    p.update(gt_quad=q,geometry_state='operator_corrected',confirmed=True)
    state['plates'][pid]=p
    link=dict(plate_id=pid,status='ASSOCIATED_FOR_GEOMETRY',provenance='operator_draw_selected_final')
    if location['kind']=='BBOX':link.update(identity_decision=identity_evidence(session,gid,s,obj),identity_location=location)
    state['links'][oid]=link;decisions['decisions'][oid]=partial.PLATE
    stacks._write(session,gid,s,state,'operator_draw_selected_final',oid)
    return oid

def overlapping_references(scene,state):
    """Advisory only: nearly coincident independent quads, never an identity."""
    result={};items=[(pid,p) for pid,p in state['plates'].items() if p['gt_quad']]
    def box(q):return (min(x for x,y in q),min(y for x,y in q),max(x for x,y in q),max(y for x,y in q))
    for i,(pid,p) in enumerate(items):
        for j,(other,op) in enumerate(items[:i]):
            a,b=box(p['gt_quad']),box(op['gt_quad'])
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if union>0 and intersection/union>=.8:
                result.setdefault(pid,[]).append(j+1);result.setdefault(other,[]).append(i+1)
    return result
