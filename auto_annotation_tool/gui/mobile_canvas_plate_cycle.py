"""A visits physical plates once, choosing GTR, then Pose, then BBOX."""


def targets(ed):
    if not hasattr(ed,'clean_bar'):
        return [dict(key=o['object_id'],oid=o['object_id'],kind='GTR',quad=ed._current_quad(o),pid=None)
                for o in (ed._read() or {}).get('objects',[])
                if not o.get('post_nms_suppressed') and ed._current_quad(o)]
    from . import mobile_canvas_clean_view as clean
    result=[];seen=set()
    for row in clean.records(ed):
        if row['role'] not in ('GT','FINAL') or row['conflict']:continue
        q=row['gt'] or row['mt'];kind='GTR' if row['gt'] else 'POSE'
        if not q:
            location=clean.source_location(ed,row)
            if location and location['kind']=='BBOX':
                l,t,r,b=location['bbox'];q=[(l,t),(r,t),(r,b),(l,b)];kind='BBOX'
        if not q:continue
        key=('GTR',row['pid']) if row['gt'] and row['pid'] else ('MT',row['oid'])
        if key in seen:continue
        seen.add(key);result.append(dict(key=key,oid=row['oid'],kind=kind,quad=q,pid=row['pid']))
    return result


def next_target(ed,choices):
    if not choices:return None
    current=None
    if hasattr(ed,'clean_bar'):
        from .mobile_canvas_clean_view import selected
        row=selected(ed)
        if row:current=('GTR',row['pid']) if row['gt'] and row['pid'] else ('MT',row['oid'])
    else:current=ed.selected
    index=next((i for i,r in enumerate(choices) if r['key']==current),-1)
    return choices[(index+1)%len(choices)]


def select(ed,target):
    if not hasattr(ed,'clean_bar'):
        return ed._choose_plate(target['oid'])
    from . import mobile_canvas_geometry_layers as layers,mobile_canvas_reservation as reservation
    ed.collision_warning.clear();ed.corner_pulse.clear();ed._reserved_edit=None;ed.gt_focus_id=''
    if not ed._choose_plate(target['oid'],source=target['kind']!='GTR'):return False
    value=layers.model(ed)
    if value and any(l['kind']==target['kind'] for l in value['layers']):
        ed._geometry_layer_state=(value['identity'],target['kind'])
    if target['kind']=='GTR':
        from .mobile_canvas_clean_view import selected
        row=selected(ed)
        if not row or row['pid']!=target['pid']:
            ed.gt_focus_id=target['oid']
            ed._reserved_edit=dict(context=reservation.key(ed),plate_id=target['pid'],oid=target['oid'])
        ed._bad_mt_selection=None
    ed.sync();ed.request_overlay()
    if target['kind']=='GTR':ed.corner_pulse.trigger()
    return True
