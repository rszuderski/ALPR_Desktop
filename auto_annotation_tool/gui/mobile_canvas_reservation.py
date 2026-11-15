"""View-only redirection from an occupied drawing area to its existing GT."""
from . import mobile_canvas_clean_view as clean
from ..ranking import mobile_manual_frame_guard as guard


def key(ed):
    s=ed._read()
    return (id(ed.h.session),ed.h.current_invocation,s.get('source_binding') if s else None)


def focused_row(ed,rows):
    focus=getattr(ed,'_reserved_edit',None)
    if (not focus or focus['context']!=key(ed) or ed.mode!='edit'
            or ed.selected!=focus['oid'] or ed.gt_focus_id!=focus['oid']):return None
    return next((r for r in rows if r['pid']==focus['plate_id'] and r['gt'] and not r['conflict']),None)


def preview(ed,pid):
    if not getattr(ed,'_reserved_edit',None):return False
    row=focused_row(ed,clean.records(ed))
    return bool(row and row['pid']==pid)


def for_source(ed,oid):
    refs=guard.reservation_records(ed._read(),session=ed.h.session,gid=ed.h.current_invocation)
    hits=[r for r in refs if any(c.get('object_id')==oid for c in r['contours'])]
    if not hits:return None
    result=dict(hits[0])
    if len(hits)>1:result['choices']=hits
    return result


def handle(ed,reference):
    """An explicit redraw keeps its target; a NEW drawing edits occupied GT."""
    target=getattr(ed,'draw_target',None)
    if target and target['kind']=='replace':
        ed.collision_warning.show(reference)
        return
    choices=reference.get('choices') or [reference]
    ed.points=[];ed.pointer=None;ed.draw_target=None;ed.draw_reference_target=None
    ed.drag=None;ed._click_candidate=None;ed._bad_mt_selection=None
    ed.mode='select';ed.gt_focus_id='';ed.selected=ed.hover=''
    if len(choices)>1:
        ed.sync();ed.request_overlay()
        ed.collision_warning.show(reference,editing=True)
        ed.message('Miejsce zarezerwowane przez GTR '+', '.join(str(r['number']) for r in choices)+'. Wybierz istniejącą GTR do edycji.')
        return
    rows=clean.records(ed);row=next((r for r in rows if r['pid']==reference['plate_id'] and r['gt'] and not r['conflict']),None)
    if not row:
        ed.sync();ed.request_overlay();ed.message('Zapisana GTR zmieniła stan. Wybierz istniejącą referencję ponownie.')
        return
    objects={o['object_id'] for o in ed._read()['objects']}
    oid=row['reference_id'] if row.get('reference_id') in objects else row['oid']
    if not ed._choose_plate(oid):return
    oid=ed.selected # A spatially attached manual reference uses its MT view alias.
    ed.gt_focus_id=oid;ed._reserved_edit=dict(context=key(ed),plate_id=row['pid'],oid=oid)
    ed.sync();ed.request_overlay();ed.c.focus_set()
    ed.collision_warning.show(reference,editing=True)
    ed.corner_pulse.trigger()
    ed.message(f"GTR {reference['number']} już rezerwuje to miejsce — edycja istniejącej GT. Esc kończy edycję.")
