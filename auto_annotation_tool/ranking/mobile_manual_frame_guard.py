"""Active GTR collision lookup; never infer or merge physical identities."""
from __future__ import annotations

import copy,math
from .mobile_pose_metrics import _cross, _signed_area, _clip_convex


def reference_records(scene,exclude_plate_id=None):
    """Saved active references, not raw model contours or history snapshots."""
    if scene.get('scope') in ('SOURCE_PHOTOGRAPH','F5A_TV_EXPOSURE_CAMERAX','F5A_SOURCE_INSPECTION'):
        for number, obj in enumerate(scene['objects'], 1):
            if obj['quad'] is not None and obj['object_id'] != exclude_plate_id:
                yield dict(plate_id=obj['object_id'],number=number,quad=obj['quad'])
        return
    from . import mobile_candidate_stacks as stacks
    from .mobile_canvas_operator_view import collision_reference_ids,reference_numbers
    state=scene.get(stacks.KEY) or stacks._legacy(scene)
    active=collision_reference_ids(scene,state)
    numbers=reference_numbers(state)
    for pid,p in state['plates'].items():
        if pid in active and pid!=exclude_plate_id:
            yield dict(plate_id=pid,number=numbers[pid],quad=p['gt_quad'])


def manual_references(scene):
    for reference in reference_records(scene):
        yield reference['number'],reference['quad']


def conflict_message(number):
    return f'GTR {number} już istnieje w tym miejscu.'


def edge_distance(point,quad):
    distances=[]
    for a,b in zip(quad,quad[1:]+quad[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=dx*dx+dy*dy
        t=max(0,min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length)) if length else 0
        distances.append(math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy))
    return min(distances)


def contains(quad,point,tolerance=0):
    turns=[_cross(quad[i],quad[(i+1)%4],point) for i in range(4)]
    return (all(v>=-1e-9 for v in turns) or all(v<=1e-9 for v in turns)
            or (tolerance>0 and edge_distance(point,quad)<=tolerance))


def overlaps(a,b):
    """Positive common area, without an IoU/quality or identity decision."""
    return abs(_signed_area(_clip_convex(a,b)))>1e-7


def reservation_records(scene,*,session=None,gid=None,exclude_plate_id=None):
    """Saved GT owns its area plus intersecting, proven FINAL BBOX/Pose.

    This projection never creates links, changes GT identity or enters scoring.
    No saved operator reference means model geometry is entirely unreserved.
    """
    refs=[dict(r,contours=[dict(kind='GTR',quad=copy.deepcopy(r['quad']))]) for r in reference_records(scene)]
    if refs and session is not None and gid is not None:
        from . import mobile_candidate_stacks as stacks
        from .mobile_mt_geometry_layers import source_bbox
        state=stacks.snapshot(session,gid,scene)
        for candidate in state['candidates']:
            if candidate['role']!='FINAL':continue
            contours=[];bbox=source_bbox(session,scene,candidate)
            if bbox:
                l,t,r,b=bbox;contours.append(dict(kind='BBOX',quad=[[l,t],[r,t],[r,b],[l,b]],object_id=candidate['object_id']))
            if candidate['mt_quad']:
                contours.append(dict(kind='POSE',quad=copy.deepcopy(candidate['mt_quad']),object_id=candidate['object_id']))
            owners=[ref for ref in refs if any(overlaps(ref['quad'],c['quad']) for c in contours)]
            # A deliberate edit of an existing GT can use its own source stack;
            # the actual areas of other GT remain protected below.
            if any(ref['plate_id']==exclude_plate_id for ref in owners):continue
            for ref in owners:ref['contours'].extend(copy.deepcopy(contours))
    return [ref for ref in refs if ref['plate_id']!=exclude_plate_id]


def find_point_collision(scene,point,*,exclude_plate_id=None,tolerance=0,session=None,gid=None):
    refs=reservation_records(scene,session=session,gid=gid,exclude_plate_id=exclude_plate_id)
    hits=[r for r in refs if any(contains(c['quad'],point,tolerance) for c in r['contours'])]
    if not hits:return None
    hits.sort(key=lambda r:(not contains(r['quad'],point,tolerance),edge_distance(point,r['quad']),r['plate_id']))
    result=copy.deepcopy(hits[0])
    if session is not None and len(hits)>1:result['choices']=copy.deepcopy(hits)
    return result


def frame_containing_point(scene,point):
    reference=find_point_collision(scene,point)
    return reference['number'] if reference else None


class GTRCollisionError(ValueError):
    def __init__(self,reference):
        self.reference=reference
        super().__init__(conflict_message(reference['number']))


def require_free_manual_area(scene,quad,*,exclude_plate_id=None,session=None,gid=None):
    """A new GT cannot overlap any reserved area, including crossing edges."""
    refs=reservation_records(scene,session=session,gid=gid,exclude_plate_id=exclude_plate_id)
    hits=[ref for ref in refs if any(overlaps(quad,c['quad'])
          or any(contains(c['quad'],p) for p in quad)
          or any(contains(quad,p) for p in c['quad']) for c in ref['contours'])]
    if hits:
        hits.sort(key=lambda r:(abs(_signed_area(r['quad'])),r['plate_id']))
        result=copy.deepcopy(hits[0])
        if session is not None and len(hits)>1:result['choices']=copy.deepcopy(hits)
        raise GTRCollisionError(result)
