"""Regression: a context status keeps the drawn object, never arms drawing."""
import copy
import json
import os
from pathlib import Path
import shutil

import pytest

from auto_annotation_tool.ranking import photo_reference as model
from test_z4_photo_reference import reference, gui, quad
from test_z4_photo_reference_context import obj, event, submenu, labels, legacy_copy


@pytest.mark.parametrize('status',list(model.GEOMETRY_REASONS))
@pytest.mark.parametrize('text',['FULL_TEXT','PARTIAL_TEXT','UNREADABLE_TEXT'])
def test_status_keeps_drawn_location_identity_text_history_and_na(reference,status,text):
    s=reference;k=s.material.order[0];oid=s.add(k,quad())
    s.edit(k,oid,text='GWE' if text!='UNREADABLE_TEXT' else '',text_state=text,reason='Nieczytelny')
    before=obj(s,k,oid);versions=copy.deepcopy(s.data['reviews'][k]['versions'])
    assert before['geometry_status']=='EXACT'
    s.set_geometry_status(k,oid,status);after=obj(s,k,oid)
    assert len(s.state(k)['objects'])==1 and after['object_id']==oid
    assert after['quad'] is None and after['localization_quad']==before['quad']==quad()
    assert after['geometry_kind']=='APPROXIMATE_QUAD' and after['geometry_status']==status
    for f in ('gt_text','partial_text','text_state','text_exclusion_reason'):assert after[f]==before[f]
    assert all(s.data['reviews'][k]['versions'][v]==state for v,state in versions.items())
    from auto_annotation_tool.ranking.mobile_manual_frame_guard import reference_records
    assert not list(reference_records(s.scene(k)))
    assert s.scene(k)['objects'][0]['geometry_state']=='unavailable'
    assert s.status(k)=='Gotowe' and model.PhotoReference(s.material,s.path).data==s.data
    s.restore(k);assert obj(s,k,oid)==before
    s.restore(k,redo=True);assert obj(s,k,oid)==after
    s.set_geometry_status(k,oid,'EXACT');exact=obj(s,k,oid)
    assert exact['quad']==quad() and exact['localization_quad']==quad()
    assert not exact['geometry_exclusion_reason'] and len(list(reference_records(s.scene(k))))==1


def test_text_and_corner_edit_do_not_promote_geometry(reference):
    s=reference;k=s.material.order[0];oid=s.add(k,quad());s.edit(k,oid,text='GWE')
    s.set_geometry_status(k,oid,'CUT_OFF');before=obj(s,k,oid)
    s.edit(k,oid,text_state='PARTIAL_TEXT')
    after=obj(s,k,oid)
    for f in ('quad','localization_quad','geometry_kind','geometry_status','geometry_exclusion_reason'):assert before[f]==after[f]
    moved=quad();moved[0]=[30,100];s.edit(k,oid,localization_quad=moved)
    assert obj(s,k,oid)['quad'] is None and obj(s,k,oid)['geometry_status']=='CUT_OFF'
    assert obj(s,k,oid)['partial_text']=='GWE'


@pytest.mark.parametrize('legacy_version',[1,2])
def test_v1_v2_read_only_until_status_and_exact_backup(reference,legacy_version):
    s=reference;k=s.material.order[0];oid=s.add(k,quad());s.edit(k,oid,text='GWE')
    for key in s.material.order[1:]:s.empty(key)
    if legacy_version==1:s=legacy_copy(s)
    else:
        data=copy.deepcopy(s.data);data['schema']=model.SCHEMA_V2
        for review in data['reviews'].values():
            for version in review['versions'].values():
                for o in version['state']['objects']:
                    o['decision_version']=2;o.pop('geometry_status',None);o.pop('localization_quad',None)
        s.path.write_text(json.dumps(data),encoding='utf-8');s=model.PhotoReference(s.material,s.path)
    before=s.path.read_bytes();old=copy.deepcopy(s.data)
    assert all(s.status(key)=='Gotowe' for key in s.material.order)
    assert s.path.read_bytes()==before
    s.set_geometry_status(k,oid,'CUT_OFF')
    assert s.data['schema']==model.SCHEMA_V3
    assert all(s.status(key)=='Gotowe' for key in s.material.order)
    backup=list(s.path.parent.glob('*.pre-v3-*'));assert len(backup)==1 and backup[0].read_bytes()==before
    for key,review in old['reviews'].items():
        assert all(s.data['reviews'][key]['versions'][v]==value for v,value in review['versions'].items())
    assert s.data['images']==old['images']


def test_legacy_no_quad_and_old_region_are_not_invented_pose(reference):
    s=reference;k=s.material.order[0]
    oid=s.add(k,geometry_reason='Narożniki zasłonięte');s.edit(k,oid,text='GWE');s=legacy_copy(s)
    s.set_geometry_status(k,oid,'CUT_OFF')
    assert obj(s,k,oid)['quad'] is None and obj(s,k,oid)['localization_quad'] is None
    assert s.status(k)=='Gotowe'
    with pytest.raises(ValueError):s.set_geometry_status(k,oid,'EXACT')
    s.edit(k,oid,approximate_region=[10,20,100,70]);before=obj(s,k,oid)
    s.set_geometry_status(k,oid,'BLURRED')
    assert obj(s,k,oid)['approximate_region']==before['approximate_region']
    with pytest.raises(ValueError):s.set_geometry_status(k,oid,'EXACT')


def test_status_backup_failure_and_parallel_writer_leave_reference_intact(reference,monkeypatch):
    s=reference;k=s.material.order[0];oid=s.add(k,quad());s.edit(k,oid,text='GWE');s=legacy_copy(s)
    before=s.path.read_bytes();other=model.PhotoReference(s.material,s.path);original=Path.open
    def fail_backup(path,*args,**kwargs):
        if '.pre-v3-' in path.name:raise OSError('cannot create backup')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_backup)
    with pytest.raises(OSError):s.set_geometry_status(k,oid,'CUT_OFF')
    assert s.path.read_bytes()==before and obj(s,k,oid)['quad']==quad()
    monkeypatch.setattr(Path,'open',original);s.set_geometry_status(k,oid,'CUT_OFF');saved=s.path.read_bytes()
    with pytest.raises(ValueError,match='innym oknie'):other.set_geometry_status(k,oid,'BLURRED')
    assert s.path.read_bytes()==saved


@pytest.mark.parametrize('corruption',[
    {'quad':quad(),'geometry_status':'CUT_OFF'},
    {'quad':None,'geometry_status':'EXACT'},
    {'localization_quad':None},
    {'localization_quad':[[0,0]]*4},
    {'localization_quad':quad(1)},
    {'geometry_status':'UNKNOWN'},
])
def test_inconsistent_accuracy_or_coordinates_rejected(reference,corruption):
    s=reference;k=s.material.order[0];s.add(k,quad())
    state=s.state(k);state['objects'][0].update(corruption)
    with pytest.raises(ValueError):s._validate_state(k,state)


def _polygon(ed,oid):
    shapes=[i for i in ed.c.find_withtag(oid) if ed.c.type(i)=='polygon']
    assert len(shapes)==1,shapes
    return shapes[0]


def test_native_ppm_status_retains_frame_id_style_and_next_click_is_not_add(gui):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    native.key(h,'d')
    for p in quad():native.click(h,ed,p)
    oid=s.state(k)['objects'][0]['object_id'];ed.entry_var.set('GWE');ed.entry_enter();root.update()
    original=copy.deepcopy(obj(s,k,oid));coords=ed.c.coords(_polygon(ed,oid))
    exact_menu=submenu(src.context_menu(event(ed,(75,125)),show=False),'Geometria')
    assert labels(exact_menu)==list(model.GEOMETRY_REASONS.values())
    colors={ed.c.itemcget(_polygon(ed,oid),'outline')}
    for index,status in enumerate(model.GEOMETRY_REASONS):
        menu=src.context_menu(event(ed,(75,125)),show=False);submenu(menu,'Geometria').invoke(index);root.update()
        o=obj(s,k,oid);shape=_polygon(ed,oid)
        assert o['object_id']==oid and len(s.state(k)['objects'])==1 and o['localization_quad']==original['quad']
        assert ed.c.coords(shape)==coords and ed.c.itemcget(shape,'dash')
        assert 'photo_approximate_outline' in ed.c.gettags(shape)
        colors.add(ed.c.itemcget(shape,'outline'))
        assert ed.selected==oid and ed.mode=='edit' and not ed.points and not src.location_draft
        assert ed.draw_target is None and src.exact_target is None and src.entry_after_draw is None
        assert ed.hit(*ed.to_canvas((75,125)))[0]==oid
        caption='Tablica 1 · '+src.geometry_style(o)['label']
        assert caption in [ed.c.itemcget(i,'text') for i in ed.c.find_withtag(oid) if ed.c.type(i)=='text']
        rev=s.data['revision'];native.click(h,ed,(500,400));root.update()
        assert s.data['revision']==rev and len(s.state(k)['objects'])==1 and not ed.points
        ed._choose_plate(oid);root.update()
    assert len(colors)==5
    menu=submenu(src.context_menu(event(ed,(75,125)),show=False),'Geometria')
    assert labels(menu)[-1]=='Przywróć dokładną geometrię'
    assert not any('punkt' in label.lower() or 'obszar' in label.lower() for label in labels(menu))
    menu.invoke(menu.index('end'));root.update()
    assert obj(s,k,oid)['quad']==original['quad'] and not ed.c.itemcget(_polygon(ed,oid),'dash')
    assert ed.c.itemcget(_polygon(ed,oid),'outline')=='#38E4D3'


def test_native_status_cancels_armed_draw_and_ghost_callbacks(gui):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    oid=s.add(k,quad());s.edit(k,oid,text='GWE');ed.sync();h._paint_image();root.update()
    native.key(h,'d');native.click(h,ed,(350,300));assert ed.points
    menu=src.context_menu(event(ed,(75,125)),show=False)
    submenu(menu,'Geometria').invoke(0);root.update()
    assert ed.selected==oid and ed.mode=='edit' and not ed.points and not src.location_draft
    src.set_text_kind(oid,'PARTIAL_TEXT');ed.entry_enter();root.update()
    assert obj(s,k,oid)['partial_text']=='GWE' and obj(s,k,oid)['quad'] is None
    before=copy.deepcopy(obj(s,k,oid));native.click(h,ed,(500,400));root.update()
    assert s.state(k)['objects']==[before] and not ed.c.find_withtag('photo_location_draft')
    # An armed fallback is also cancelled by an existing object's status.
    src.begin_geometry('APPROXIMATE_POINT',reason='Nie można ustalić pełnego obrysu')
    assert src.location_draft
    menu=src.context_menu(event(ed,(75,125)),show=False);submenu(menu,'Geometria').invoke(1);root.update()
    assert not src.location_draft and ed.mode=='edit'
    native.click(h,ed,(500,400));assert len(s.state(k)['objects'])==1


def test_native_drag_uncertain_corner_zoom_pan_undo_and_reopen(gui):
    from auto_annotation_tool.gui.z4_photo_reference import PhotoReferenceWindow
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    oid=s.add(k,quad());s.edit(k,oid,text='GWE');ed.sync();h._paint_image();root.update()
    src.set_geometry_status(oid,'CUT_OFF');src.set_text_kind(oid,'PARTIAL_TEXT');ed.entry_enter();root.update()
    before=obj(s,k,oid);h._view_zoom=2;h._view_center=(120,130);h._paint_image();root.update()
    e=event(ed,before['localization_quad'][0]);dest=event(ed,(30,103))
    ed.press(e);ed.motion(dest);assert obj(s,k,oid)==before
    ed.release(dest);root.update();after=obj(s,k,oid)
    assert after['localization_quad']!=before['localization_quad'] and after['quad'] is None
    assert after['geometry_status']=='CUT_OFF' and after['partial_text']=='GWE'
    ed.undo();assert obj(s,k,oid)==before
    ed.undo(redo=True);assert obj(s,k,oid)==after
    h.window.geometry('1000x650');root.update();h.fit_image();root.update()
    assert obj(s,k,oid)==after and ed.c.itemcget(_polygon(ed,oid),'outline')=='#FF9F43'
    h.close();root.update();new=PhotoReferenceWindow(root,s.material,output=s.path);root.update()
    try:
        assert obj(new.session,k,oid)==after and new.progress.get()=='Gotowe: 1/40'
        assert new.canvas.itemcget(_polygon(new._canvas_editor,oid),'outline')=='#FF9F43'
    finally:new.close();root.update()


def test_native_existing_na_reason_status_does_not_arm_point(gui):
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    src.without_outline('Narożniki zasłonięte');ed.entry_var.set('GWE');ed.entry_enter();root.update()
    oid=s.state(k)['objects'][0]['object_id'];l,t,r,b=ed.badges[oid]
    from types import SimpleNamespace
    menu=src.context_menu(SimpleNamespace(x=l+10,y=t+10,widget=ed.c),show=False)
    geom=submenu(menu,'Geometria');assert 'Wskaż położenie punktem' in labels(geom)
    geom.invoke(0);root.update()
    assert obj(s,k,oid)['quad'] is None and obj(s,k,oid)['localization_quad'] is None
    assert not src.location_draft and len(s.state(k)['objects'])==1 and s.status(k)=='Gotowe'


def test_native_status_failed_save_and_retry_never_arm_drawing(gui,monkeypatch):
    root,h,ed,s,errors=gui;k=h.current_invocation;src=h.source
    oid=s.add(k,quad());s.edit(k,oid,text='GWE');ed.sync();h._paint_image();root.update()
    before=s.path.read_bytes();writer=model._atomic_write_json
    monkeypatch.setattr(model,'_atomic_write_json',lambda *a:(_ for _ in ()).throw(OSError('disk unavailable')))
    src.set_geometry_status(oid,'CUT_OFF');root.update()
    assert src.pending_action and s.path.read_bytes()==before and ed.mode=='edit'
    assert not src.location_draft and not ed.points and ed.draw_target is None
    assert not h.select(s.material.order[1])
    monkeypatch.setattr(model,'_atomic_write_json',writer);src.retry();root.update()
    assert not src.pending_action and obj(s,k,oid)['geometry_status']=='CUT_OFF'
    assert ed.entry is None and not src.location_draft and not ed.points
    assert obj(s,k,oid)['localization_quad']==quad() and obj(s,k,oid)['gt_text']=='GWE'


def test_private_current_reference_is_read_only_and_removed_gwe_is_in_history(tmp_path):
    path=os.environ.get('ALPR_GEOMETRY_CURRENT_REFERENCE_COPY');material=os.environ.get('ALPR_A1_REAL_MATERIAL')
    if not path or not material:pytest.skip('Private current snapshot and read-only material not supplied')
    output=tmp_path/'reference.json';shutil.copyfile(path,output)
    before=output.read_bytes();s=model.PhotoReference(model.Material(material),output)
    k=s.material.order[38];oid='PGT:f87de896a38641f7ad531d71fb8a3321'
    assert oid not in {o['object_id'] for o in s.state(k)['objects']}
    historical=[o for v in s.data['reviews'][k]['versions'].values() for o in v['state']['objects'] if o['object_id']==oid and o['quad'] is not None]
    assert historical and any(o['gt_text']=='GWE' for o in historical)
    assert len(s.state(k)['objects'])==4 and all(s.status(key)=='Gotowe' for key in s.material.order)
    assert output.read_bytes()==before
