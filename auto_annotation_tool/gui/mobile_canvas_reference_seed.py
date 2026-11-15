"""Unsaved operator starting corners. Never a scientific GT or a stored ID."""
import copy


def project(session,gid,scene,state,candidate):
    from ..ranking.mobile_manual_frame_guard import reservation_records
    from ..ranking.mobile_mt_geometry_layers import source_bbox
    from ..ranking.mobile_pose_metrics import validate_mt_quad
    c=candidate
    if not c or c['role']!='FINAL' or c['association_status']!='UNRESOLVED' or c.get('plate_id'):return None
    # A removed/detached operator reference must never regenerate from MT.
    if any(c['object_id'] in (p['origin_object_id'],p.get('primary_object_id')) for p in state['plates'].values()):return None
    refs=reservation_records(scene,session=session,gid=gid)
    if any(any(q.get('object_id')==c['object_id'] for q in ref['contours']) for ref in refs):return None
    if c['mt_quad']:
        try:q=validate_mt_quad(c['mt_quad'],*scene['image_size'])
        except (ValueError,TypeError):return None
        return dict(quad=[list(p) for p in q],origin='Pose')
    box=source_bbox(session,scene,c)
    if box:
        l,t,r,b=box
        return dict(quad=[[l,t],[r,t],[r,b],[l,b]],origin='BBOX')
    return None


def focus(ed,oid):
    from .mobile_canvas_navigator import select_layer
    if not select_layer(ed,'GTR',oid):return False
    ed.gt_focus_id=oid;ed.sync();ed.request_overlay()
    ed.message('Szkic GTR z modelu — przeciągnij narożniki. Zapis nastąpi po puszczeniu narożnika; Esc kończy edycję.')
    return True


def for_editor(ed,scene,state):
    from .mobile_canvas_visibility import gt_only
    if gt_only(ed):return {}
    key=(id(ed.h.session),ed.h.current_invocation,ed.h.session.review.review_revision,scene['source_binding'])
    if getattr(ed,'_reference_seeds_key',None)!=key:
        ed._reference_seeds={c['object_id']:project(ed.h.session,ed.h.current_invocation,scene,state,c) for c in state['candidates']}
        ed._reference_seeds_key=key
    return ed._reference_seeds


def right_target(ed,x,y):
    """An unsaved seed is a view of its FINAL, never a persisted reference.

    An explicit active source owns overlap; saved GTRs keep their BAD guard.
    Without an active source, only a unique seed outside saved GTRs qualifies.
    """
    if ed.mode=='draw' or ed.drag:return None
    from . import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers
    from .mobile_canvas_selection import contains
    from ..ranking.mobile_mt_geometry_layers import source_bbox
    from ..ranking.mobile_canvas_review import require_source_final
    from .mobile_canvas_visibility import enabled
    row=clean.selected(ed);value=layers.model(ed);scene=ed._read()
    if row and row['plate'] and (ed.gt_focus_id or not value or value['active']=='GTR'):return None
    def hit(r):
        shapes=[r['gt']] if r.get('seed') and enabled(ed,'ref') else []
        if enabled(ed,'inf'):
            shapes.append(r['mt'])
            b=source_bbox(ed.h.session,scene,r['candidate'])
            if b:
                l,t,rr,bottom=b;shapes.append([(l,t),(rr,t),(rr,bottom),(l,bottom)])
        return any(q and contains(ed,q,x,y) for q in shapes)
    if row and row['candidate'] and row['role']=='FINAL':
        explicit=getattr(ed,'_bad_mt_selection',None)==(id(ed.h.session),ed.h.current_invocation,row['candidate']['object_id'])
        if (row.get('seed') or (explicit and value and value['active'] in ('POSE','BBOX'))) and hit(row):
            oid=row['candidate']['object_id'];require_source_final(ed.h.session,ed.h.current_invocation,oid);return oid
    rows=clean.records(ed)
    if any(r['plate'] and r['gt'] and contains(ed,r['gt'],x,y) for r in rows):return None
    hits=[r for r in rows if r.get('seed') and hit(r)]
    if len(hits)>1:raise ValueError('Nakładające się wskazania MT. Wybierz właściwe źródło przez S lub nawigator.')
    if len(hits)==1:
        oid=hits[0]['candidate']['object_id'];require_source_final(ed.h.session,ed.h.current_invocation,oid);return oid
    return None
