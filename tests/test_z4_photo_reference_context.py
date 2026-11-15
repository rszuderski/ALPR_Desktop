"""Photo-only context actions. All writes target private fixtures/copies."""
import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from auto_annotation_tool.ranking import photo_reference as model
from test_z4_photo_reference import reference, gui, quad, material


REASON='Ucięta krawędzią obrazu'


def obj(s,k,oid):
    return next(o for o in s.state(k)['objects'] if o['object_id']==oid)


def geometry(kind):
    if kind=='EXACT_QUAD':return dict(quad=quad())
    return dict(geometry_reason=REASON,**({'approximate_region':[10,20,130,75]} if kind=='APPROXIMATE_REGION' else {'approximate_point':[630,475]}))


@pytest.mark.parametrize('geom',['EXACT_QUAD','APPROXIMATE_REGION','APPROXIMATE_POINT'])
@pytest.mark.parametrize('text',['FULL_TEXT','PARTIAL_TEXT','UNREADABLE_TEXT'])
def test_independent_decisions_reopen_history_no_fake_gt(reference,geom,text):
    s=reference;k=s.material.order[0];oid=s.add(k,**geometry(geom))
    s.edit(k,oid,text_state=text,text='GWE' if text!='UNREADABLE_TEXT' else '',reason='Rozmyty numer')
    o=obj(s,k,oid)
    assert o['geometry_kind']==geom and o['text_state']==text and s.status(k)=='Gotowe'
    assert o['quad'] is None if geom!='EXACT_QUAD' else o['quad']==quad()
    assert o['gt_text']==('GWE' if text=='FULL_TEXT' else '')
    assert o['partial_text']==('GWE' if text=='PARTIAL_TEXT' else '')
    assert 'bbox' not in o and 'cer' not in o and 'iou' not in o
    assert model.PhotoReference(s.material,s.path).data==s.data
    s.restore(k);assert s.status(k)=='Wymaga uzupełnienia'
    s.restore(k,redo=True);assert obj(s,k,oid)==o
    from auto_annotation_tool.ranking.mobile_manual_frame_guard import reference_records
    assert len(list(reference_records(s.scene(k))))==(1 if geom=='EXACT_QUAD' else 0)


def test_all_transitions_keep_id_versions_and_no_negative_by_deletion(reference):
    s=reference;k=s.material.order[0];oid=s.add(k,quad())
    s.edit(k,oid,text='GWE')
    for kind in ('APPROXIMATE_POINT','APPROXIMATE_REGION','EXACT_QUAD','APPROXIMATE_REGION','APPROXIMATE_POINT','EXACT_QUAD'):
        before=obj(s,k,oid);s.edit(k,oid,**geometry(kind));after=obj(s,k,oid)
        assert after['object_id']==oid and after['gt_text']=='GWE'
        s.restore(k);assert obj(s,k,oid)==before
        s.restore(k,redo=True);assert obj(s,k,oid)==after
    for kind in ('PARTIAL_TEXT','FULL_TEXT','UNREADABLE_TEXT','PARTIAL_TEXT','UNREADABLE_TEXT','FULL_TEXT'):
        versions=copy.deepcopy(s.data['reviews'][k]['versions'])
        s.edit(k,oid,text_state=kind,text='' if kind=='UNREADABLE_TEXT' else 'GWE',reason='Rozmyty numer')
        assert obj(s,k,oid)['object_id']==oid
        assert all(s.data['reviews'][k]['versions'][v]==state for v,state in versions.items())
    with pytest.raises(ValueError):s.empty(k)
    s.remove(k,oid);assert s.state(k)['presence']=='UNRESOLVED'
    s.restore(k);assert obj(s,k,oid)['object_id']==oid


def legacy_copy(s):
    """Make a genuinely v1 synthetic file, never downgrade a real reference."""
    data=copy.deepcopy(s.data);data['schema']=model.SCHEMA
    for review in data['reviews'].values():
        for version in review['versions'].values():
            for o in version['state']['objects']:
                for field in ('decision_version','text_state','partial_text','geometry_kind','approximate_region','approximate_point','localization_quad','geometry_status'):
                    o.pop(field,None)
    path=s.path.parent/'legacy'/'reference_v1.json';path.parent.mkdir()
    path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    return model.PhotoReference(s.material,path)


def test_legacy_40_untouched_until_edit_backup_exact_bytes(reference):
    s=reference
    for k in s.material.order:s.empty(k)
    k=s.material.order[-1];s.restore(k)
    for i in range(4):
        oid=s.add(k,quad(i) if i<3 else None,geometry_reason='' if i<3 else REASON)
        s.edit(k,oid,text='GWE' if i==3 else 'SYN'+str(i))
    s=legacy_copy(s);before=s.path.read_bytes();history=copy.deepcopy(s.data['reviews'][k]['versions'])
    assert all(s.status(key)=='Gotowe' for key in s.material.order)
    assert model.text_kind(obj(s,k,oid))=='FULL_TEXT' and obj(s,k,oid)['quad'] is None
    assert s.path.read_bytes()==before and not list(s.path.parent.glob('*.pre-v3-*'))
    s.edit(k,oid,approximate_point=[638,478])
    assert s.data['schema']==model.SCHEMA_V3
    backups=list(s.path.parent.glob('*.pre-v3-*'));assert len(backups)==1 and backups[0].read_bytes()==before
    s.edit(k,oid,text_state='PARTIAL_TEXT')
    assert obj(s,k,oid)['gt_text']=='' and obj(s,k,oid)['partial_text']=='GWE'
    assert all(s.data['reviews'][k]['versions'][v]==state for v,state in history.items())
    assert all(s.status(key)=='Gotowe' for key in s.material.order)
    assert len(list(s.path.parent.glob('*.pre-v3-*')))==1


@pytest.mark.parametrize('location',[
    {'approximate_point':[-1,20]}, {'approximate_point':[641,20]},
    {'approximate_point':[0,float('nan')]}, {'approximate_point':[True,20]},
    {'approximate_region':[20,20,10,40]}, {'approximate_region':[10,20,10,40]},
    {'approximate_region':[0,0,640,481]}, {'approximate_region':[1,2]},
    {'quad':quad(),'approximate_point':[10,10]},
    {'approximate_point':[10,10],'approximate_region':[1,1,20,20]},
])
def test_invalid_location_is_atomic(reference,location):
    s=reference;k=s.material.order[0]
    with pytest.raises(ValueError):s.add(k,geometry_reason=REASON,**location)
    assert not s.path.exists() and not s.state(k)['objects']


def test_partial_empty_is_not_complete_and_unreadable_requires_reason(reference):
    s=reference;k=s.material.order[0];oid=s.add(k,quad())
    s.edit(k,oid,text_state='PARTIAL_TEXT')
    assert s.status(k)=='Wymaga uzupełnienia'
    before=s.path.read_bytes()
    with pytest.raises(ValueError):s.edit(k,oid,text_state='UNREADABLE_TEXT')
    assert s.path.read_bytes()==before
    s.edit(k,oid,text='GWE');assert s.status(k)=='Gotowe'
    assert obj(s,k,oid)['partial_text']=='GWE' and not obj(s,k,oid)['gt_text']


def test_migration_backup_failure_and_stale_writer_preserve_file(reference,monkeypatch):
    s=reference;k=s.material.order[0];oid=s.add(k,quad());s.edit(k,oid,text='GWE');s=legacy_copy(s)
    other=model.PhotoReference(s.material,s.path);before=s.path.read_bytes();original=Path.open
    def blocked(path,*args,**kwargs):
        if '.pre-v2-' in path.name:raise OSError('backup unavailable')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',blocked)
    with pytest.raises(OSError):s.edit(k,oid,text_state='PARTIAL_TEXT')
    assert s.path.read_bytes()==before and s.data['schema']==model.SCHEMA
    monkeypatch.setattr(Path,'open',original)
    s.edit(k,oid,text_state='PARTIAL_TEXT');after=s.path.read_bytes()
    with pytest.raises(ValueError,match='innym oknie'):other.edit(k,oid,text='OTHER')
    assert s.path.read_bytes()==after


def event(ed,point,widget=None):
    x,y=map(round,ed.to_canvas(point));c=ed.c
    return SimpleNamespace(x=x,y=y,x_root=c.winfo_rootx()+x,y_root=c.winfo_rooty()+y,widget=widget or c)


def submenu(menu,label):
    for i in range(menu.index('end')+1):
        if menu.type(i)=='cascade' and menu.entrycget(i,'label')==label:
            return menu.nametowidget(menu.entrycget(i,'menu'))
    raise AssertionError(label)


def labels(menu):
    return [menu.entrycget(i,'label') for i in range(menu.index('end')+1) if menu.type(i)!='separator']


def drag(h,ed,a,b):
    e=event(ed,a);f=event(ed,b)
    h.canvas.event_generate('<ButtonPress-1>',x=e.x,y=e.y)
    h.canvas.event_generate('<B1-Motion>',x=f.x,y=f.y)
    h.canvas.event_generate('<ButtonRelease-1>',x=f.x,y=f.y)
    h.window.update()


def test_native_ppm_background_exact_badge_inline_and_null(gui,monkeypatch):
    import tkinter as tk
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    menu=src.context_menu(event(ed,(300,300)),show=False)
    assert labels(menu)==['Dodaj tablicę','Na zdjęciu nie ma tablic']
    assert labels(submenu(menu,'Dodaj tablicę'))==['Narysuj obrys (D)','Nie mogę wskazać obszaru — punkt']
    submenu(menu,'Dodaj tablicę').invoke(0)
    for p in quad():native.click(h,ed,p)
    oid=s.state(k)['objects'][0]['object_id'];ed.entry_var.set('GWE');ed.entry_enter();root.update()
    menu=src.context_menu(event(ed,(75,125)),show=False)
    assert labels(menu)==['Numer','Geometria','Usuń tablicę']
    # A background context must not edit the previous selection.
    menu=src.context_menu(event(ed,(450,380)),show=False)
    assert labels(menu)[0]=='Dodaj tablicę' and menu.entrycget(1,'state')=='disabled'
    l,t,r,b=ed.badges[oid];e=SimpleNamespace(x=(l+r)/2,y=(t+b)/2,widget=h.canvas)
    assert labels(src.context_menu(e,show=False))[0]=='Numer'
    ed.start_entry(oid);root.update();seen=[]
    monkeypatch.setattr(tk.Menu,'tk_popup',lambda m,x,y:seen.append((m,x,y)))
    ed.entry.event_generate('<Button-3>',x=8,y=8);root.update()
    assert seen and labels(seen[-1][0])==['Numer','Geometria','Usuń tablicę']
    src.menu_open=False;ed.entry_enter()
    src.geometry_na(oid,REASON);root.update()
    l,t,r,b=ed.badges[oid]
    assert labels(src.context_menu(SimpleNamespace(x=l+10,y=t+10,widget=h.canvas),show=False))[0]=='Numer'
    assert not errors


def test_native_region_point_conversions_drag_escape_and_history(gui):
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    src.begin_geometry('APPROXIMATE_REGION',reason=REASON)
    drag(h,ed,(150,230),(270,290))
    oid=s.state(k)['objects'][0]['object_id'];ed.entry_var.set('GWE');ed.entry_enter();root.update()
    before=obj(s,k,oid);assert before['quad'] is None
    assert labels(src.context_menu(event(ed,(210,260)),show=False))==['Numer','Geometria','Usuń tablicę']
    drag(h,ed,(210,260),(240,310))
    after=obj(s,k,oid);assert after['approximate_region']!=before['approximate_region']
    ed.undo();assert obj(s,k,oid)==before
    ed.undo(redo=True);assert obj(s,k,oid)==after
    src.begin_geometry('APPROXIMATE_POINT',oid,reason=REASON,point=[500,350]);root.update()
    assert obj(s,k,oid)['approximate_point']==[500,350] and obj(s,k,oid)['quad'] is None
    assert labels(src.context_menu(event(ed,(500,350)),show=False))[0]=='Numer'
    frozen=s.path.read_bytes();e=event(ed,(500,350));f=event(ed,(550,400))
    ed.press(e);ed.motion(f);assert s.path.read_bytes()==frozen
    ed.escape();root.update();assert s.path.read_bytes()==frozen
    drag(h,ed,(500,350),(550,400));assert obj(s,k,oid)['approximate_point']!=[500,350]
    src.begin_geometry('EXACT_QUAD',oid)
    import test_mobile_p2b2_canvas_gui as native
    for p in quad():native.click(h,ed,p)
    assert len(s.state(k)['objects'])==1 and obj(s,k,oid)['quad'] is not None
    assert obj(s,k,oid)['approximate_point'] is None


def test_native_partial_autosave_ctrl_s_unreadable_close_resume(gui):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui.z4_photo_reference import PhotoReferenceWindow
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    original_windows=[w for w in root.winfo_children() if w.winfo_class()=='Toplevel']
    src.begin_geometry('APPROXIMATE_POINT',reason=REASON,point=[500,350]);root.update()
    oid=s.state(k)['objects'][0]['object_id'];ed.entry_var.set('GWE');ed.entry_enter()
    src.set_text_kind(oid,'PARTIAL_TEXT');root.update()
    assert ed.entry_var.get()=='GWE' and obj(s,k,oid)['gt_text']==''
    ed.entry_var.set('GW');ed.entry_typed()
    native.pump(root,lambda:obj(s,k,oid)['partial_text']=='GW')
    ed.entry_var.set('GWE');ed.entry.focus_force();native.pump(root,lambda:root.focus_get() is ed.entry)
    ed.entry.event_generate('<Control-s>');root.update()
    assert ed.entry is None and obj(s,k,oid)['partial_text']=='GWE'
    assert src.text_caption(obj(s,k,oid))=='Numer: fragment GWE'
    assert s.status(k)=='Gotowe'
    src.set_text_kind(oid,'UNREADABLE_TEXT');assert obj(s,k,oid)['text_state']=='UNREADABLE_TEXT'
    ed.undo();assert obj(s,k,oid)['partial_text']=='GWE'
    src.set_text_kind(oid,'FULL_TEXT');ed.entry_enter();assert obj(s,k,oid)['gt_text']=='GWE'
    src.set_text_kind(oid,'PARTIAL_TEXT');ed.entry_enter()
    assert [w for w in root.winfo_children() if w.winfo_class()=='Toplevel']==original_windows
    h.close();root.update()
    reopened=PhotoReferenceWindow(root,s.material,output=s.path);root.update()
    try:
        assert reopened.current_invocation==k and reopened.progress.get()=='Gotowe: 1/40'
        assert obj(reopened.session,k,oid)['partial_text']=='GWE'
    finally:reopened.close();root.update()


def test_native_point_ppm_uses_photo_position_margins_zoom_pan_edges(gui):
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    menu=src.context_menu(event(ed,(620,460)),show=False)
    submenu(menu,'Dodaj tablicę').invoke(1);ed.entry_var.set('GWE');ed.entry_enter();root.update()
    oid=s.state(k)['objects'][0]['object_id'];p=obj(s,k,oid)['approximate_point']
    assert p==pytest.approx([620,460],abs=1)
    h._view_zoom=2;h._view_center=(400,300);h._paint_image();root.update()
    drag(h,ed,p,(600,430));assert obj(s,k,oid)['approximate_point']==pytest.approx([600,430],abs=1)
    h.fit_image();root.update();h.window.geometry('1100x650');root.update()
    frozen=s.path.read_bytes()
    assert src.context_menu(SimpleNamespace(x=-10,y=-10,widget=h.canvas),show=False)=='break'
    assert s.path.read_bytes()==frozen
    src.begin_geometry('APPROXIMATE_POINT',oid,reason=REASON)
    ed.press(SimpleNamespace(x=-10,y=-10));ed.release();assert s.path.read_bytes()==frozen
    assert not h.select(s.material.order[1]);ed.escape()
    p=obj(s,k,oid)['approximate_point'];e=event(ed,p)
    ed.press(e);ed.motion(SimpleNamespace(x=100000,y=100000));ed.release();root.update()
    assert obj(s,k,oid)['approximate_point']==[640,480]
    l,t,r,b=ed.badges[oid];iw,ih=ed.to_canvas((640,480))
    assert r<=min(iw,h.canvas.winfo_width())+1 and b<=min(ih,h.canvas.winfo_height())+1


def test_private_copy_real_40_photo40_plate4(tmp_path):
    path=os.environ.get('ALPR_PPM_REFERENCE_COPY');root=os.environ.get('ALPR_A1_REAL_MATERIAL')
    if not path or not root:pytest.skip('Private reference snapshot and read-only source not supplied')
    source=Path(path);before=source.read_bytes();output=tmp_path/'reference_v1.json';shutil.copyfile(source,output)
    s=model.PhotoReference(model.Material(root),output)
    assert all(s.status(k)=='Gotowe' for k in s.material.order)
    matches=[(key,o) for key in s.material.order for o in s.state(key)['objects'] if o['quad'] is None and o['gt_text']=='GWE']
    assert len(matches)==1
    k,plate=matches[0];oid=plate['object_id']
    assert s.material.order.index(k)==38  # actual TV order: 39, not 40 in the handoff
    assert s.state(k)['objects'][3]['object_id']==oid
    assert plate['quad'] is None and plate['gt_text']=='GWE' and model.text_kind(plate)=='FULL_TEXT'
    # Read-only visual inspection of this 1520x1528 source: visible plate
    # fragment at the right edge. Approximate test point ONLY on this copy;
    # it is neither an exact corner nor a production operator decision.
    assert (s.material.images[k]['width'],s.material.images[k]['height'])==(1520,1528)
    s.edit(k,oid,approximate_point=[1505,954]);s.edit(k,oid,text_state='PARTIAL_TEXT')
    assert obj(s,k,oid)['quad'] is None and obj(s,k,oid)['partial_text']=='GWE'
    assert all(s.status(key)=='Gotowe' for key in s.material.order)
    assert source.read_bytes()==before
    assert model.PhotoReference(s.material,output).data==s.data


def test_native_location_failed_save_keeps_draft_blocks_navigation_and_retries(gui,monkeypatch):
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    src.begin_geometry('APPROXIMATE_POINT',reason=REASON,point=[400,300])
    oid=s.state(k)['objects'][0]['object_id'];ed.entry_var.set('GWE');ed.entry_enter();root.update()
    before=s.path.read_bytes();writer=model._atomic_write_json
    monkeypatch.setattr(model,'_atomic_write_json',lambda *a:(_ for _ in ()).throw(OSError('disk unavailable')))
    drag(h,ed,(400,300),(450,350))
    assert s.path.read_bytes()==before and src.location_draft and src.pending_action
    assert not h.select(s.material.order[1]) and h.current_invocation==k
    assert h.progress.get()=='Gotowe: 0/40'
    from auto_annotation_tool.gui import z4_photo_reference as ui
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:False)
    h.close();assert not h._closed
    monkeypatch.setattr(model,'_atomic_write_json',writer)
    src.retry();root.update()
    assert not src.location_draft and not src.pending_action and len(s.state(k)['objects'])==1
    assert obj(s,k,oid)['approximate_point']==pytest.approx([450,350],abs=1)
    assert h.progress.get()=='Gotowe: 1/40'


def test_native_context_dispatch_entry_and_no_plates(gui,monkeypatch):
    import tkinter as tk
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    # Windows native popup owns a modal OS loop. Exercise the real Tk binding
    # and menu commands without waiting for a physical operator to dismiss it.
    posted=[]
    monkeypatch.setattr(tk.Menu,'tk_popup',lambda menu,x,y:posted.append(menu))
    native.focus_canvas(h);e=event(ed,(300,350))
    h.canvas.event_generate('<Button-3>',x=e.x,y=e.y);root.update()
    menu=src._context_menu;assert posted[-1] is menu and src.menu_open
    menu.invoke(1);src._menu_closed();root.update()
    assert s.state(k)['presence']=='NO_PLATES'
    ed.undo();assert s.state(k)['presence']=='UNRESOLVED'
    src.begin_geometry('APPROXIMATE_POINT',reason=REASON,point=[400,300]);root.update()
    oid=s.state(k)['objects'][0]['object_id'];ed.entry_var.set('GWE')
    ed.entry.focus_force();native.pump(root,lambda:root.focus_get() is ed.entry)
    ed.entry.event_generate('<Button-3>',x=5,y=5);root.update()
    menu=src._context_menu;assert posted[-1] is menu and ed.entry is not None
    submenu(menu,'Numer').invoke(1);src._menu_closed();root.update()
    assert ed.entry is not None and ed.entry_var.get()=='GWE'
    assert obj(s,k,oid)['partial_text']=='GWE' and obj(s,k,oid)['gt_text']==''
    ed.entry_enter()


def test_native_exif_transposed_coordinates_fit_zoom_pan_resize(tmp_path):
    import tkinter as tk
    from PIL import Image
    from auto_annotation_tool.gui.z4_photo_reference import PhotoReferenceWindow
    m=material(tmp_path/'material');key=m.order[0];p=m.root/m.images[key]['relative_path']
    with Image.open(p) as image:
        image.load();exif=Image.Exif();exif[274]=6;image.save(p,exif=exif)
    sha=model.digest(p)
    sorting=model.read_json(m.sort_path);sorting['records'][0]['candidate_sha256']=sha
    tv=model.read_json(m.tv_path);tv['presentation_order'][0]['sha256']=sha
    m.sort_path.write_text(json.dumps(sorting),encoding='utf-8');m.tv_path.write_text(json.dumps(tv),encoding='utf-8')
    m=model.Material(m.root);key=m.order[0];assert (m.images[key]['width'],m.images[key]['height'])==(480,640)
    protected={p:model.digest(p) for p in [p,m.sort_path,m.tv_path]}
    root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *a:errors.append(a)
    h=PhotoReferenceWindow(root,m,output=tmp_path/'reference.json');root.update();ed=h._canvas_editor
    try:
        assert h._image.size==(480,640)
        h.source.begin_geometry('APPROXIMATE_REGION',reason=REASON)
        drag(h,ed,(100,280),(200,340));oid=h.session.state(key)['objects'][0]['object_id']
        ed.entry_var.set('GWE');ed.entry_enter();root.update()
        assert obj(h.session,key,oid)['approximate_region']==pytest.approx([100,280,200,340],abs=1)
        h._view_zoom=2;h._view_center=(200,350);h._paint_image();root.update()
        drag(h,ed,(150,310),(200,360))
        assert obj(h.session,key,oid)['approximate_region']==pytest.approx([150,330,250,390],abs=2)
        before=h.session.path.read_bytes();h.window.geometry('1200x700');root.update();h.fit_image();root.update()
        assert h.session.path.read_bytes()==before
        assert m.images[key]['coordinate_space']=='exif_transposed_source_pixels'
        assert all(model.digest(p)==sha for p,sha in protected.items())
        assert not errors
    finally:ed.escape();h.close();root.update();root.destroy()
