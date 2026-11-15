"""A1: synthetic/private material only; no inference or production writes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pytest
from PIL import Image

from auto_annotation_tool.ranking import photo_reference as model


def material(root, count=40):
    root.mkdir(parents=True)
    tv=root.parent/'03_POKAZ_TV';tv.mkdir()
    order=[]
    for i in range(count):
        relative=f'bucket{i//5}/photo{i%5}.jpg';path=root/relative;path.parent.mkdir(exist_ok=True)
        Image.new('RGB',(640,480),(5*i,80,100)).save(path)
        order.append(dict(sha256=model.digest(path),sciezka_zrodla=relative))
    (root/'sorting_manifest.json').write_text(json.dumps(dict(independence={'status':'INCOMPLETE','ok':False},records=[{'candidate_sha256':r['sha256']} for r in order])),encoding='utf-8')
    (tv/'manifest_blokow.json').write_text(json.dumps({'presentation_order':order}),encoding='utf-8')
    return model.Material(root)


@pytest.fixture
def reference(tmp_path):
    return model.PhotoReference(material(tmp_path/'material'),tmp_path/'reference/reference.json')


def quad(i=0):
    x=25+i*145
    return [[x,100],[x+100,100],[x+100,155],[x,155]]


def test_recursive_40_identity_basename_duplicates_no_assumed_negatives(reference):
    s=reference;m=s.material
    assert len(m.order)==len(m.images)==40
    assert len({Path(r['relative_path']).name for r in m.images.values()})==5
    assert not s.path.exists()
    assert all(s.status(k)=='Do oznaczenia' and s.state(k)['presence']=='UNRESOLVED' for k in m.order)
    assert all(k=='img-sha256-'+m.images[k]['source_file_sha256'] for k in m.order)
    assert s.data['dataset_independence']=={'status':'INCOMPLETE','ok':False}


@pytest.mark.parametrize('count',[0,39,41])
def test_manifest_wrong_population_is_blocked(tmp_path,count):
    with pytest.raises(ValueError):material(tmp_path/'material',count)


def test_negative_one_action_undo_redo_resume(reference):
    s=reference;k=s.material.order[0];s.empty(k)
    assert s.status(k)=='Gotowe' and s.state(k)==dict(presence='NO_PLATES',objects=[])
    rev=s.data['revision'];s.empty(k);assert s.data['revision']==rev
    s.restore(k);assert s.state(k)['presence']=='UNRESOLVED' and s.status(k)!='Gotowe'
    s.restore(k,redo=True)
    opened=model.PhotoReference(s.material,s.path)
    assert opened.data==s.data


@pytest.mark.parametrize('count',[1,2,4])
def test_multiple_plates_independent_text_geometry_and_complete(reference,count):
    s=reference;k=s.material.order[1]
    ids=[s.add(k,quad(i)) for i in range(count)]
    assert len(set(ids))==count and s.status(k)=='Wymaga uzupełnienia'
    for i,oid in enumerate(ids):s.edit(k,oid,text=f'SYN{i}')
    assert s.status(k)=='Gotowe'
    s.edit(k,ids[-1],text='')
    assert s.status(k)=='Wymaga uzupełnienia'
    s.edit(k,ids[-1],text='',reason='Numer nieczytelny na fotografii')
    assert s.status(k)=='Gotowe'
    assert all(not o.get('attempt_id') for o in s.scene(k)['objects'])


def test_geometry_na_is_a_real_plate_without_fake_corners(reference):
    s=reference;k=s.material.order[0]
    oid=s.add(k,geometry_reason='Narożniki zasłonięte');s.edit(k,oid,text='SYN01')
    assert s.status(k)=='Gotowe' and s.state(k)['objects'][0]['quad'] is None
    s.edit(k,oid,quad=quad())
    assert s.state(k)['objects'][0]['object_id']==oid and not s.state(k)['objects'][0]['geometry_exclusion_reason']
    s.restore(k);assert s.state(k)['objects'][0]['quad'] is None


def test_removal_does_not_imply_negative_and_history_is_retained(reference):
    s=reference;k=s.material.order[0];oid=s.add(k,quad());s.edit(k,oid,text='SYN')
    s.remove(k,oid);assert s.state(k)['presence']=='UNRESOLVED'
    s.restore(k);assert s.state(k)['objects'][0]['gt_text']=='SYN'
    versions=set(s.data['reviews'][k]['versions'])
    s.edit(k,oid,text='NEW')
    assert versions <= set(s.data['reviews'][k]['versions'])
    assert s.state(k)['objects'][0]['object_id']==oid


def test_resume_last_and_next_pending(reference):
    s=reference;one,two,three=s.material.order[:3]
    s.empty(one);s.empty(two);s.remember(two)
    opened=model.PhotoReference(s.material,s.path)
    assert opened.data['last_image_id']==two and opened.next_pending(two)==three
    assert opened.next_pending(one)==three


def test_changed_or_missing_source_blocks_save(reference):
    s=reference;k=s.material.order[0];p=s.material.root/s.material.images[k]['relative_path']
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):s.empty(k)
    assert not s.path.exists() and s.state(k)['presence']=='UNRESOLVED'
    p.unlink()
    with pytest.raises(OSError):s.empty(k)


def test_save_error_is_atomic_and_retryable(reference,monkeypatch):
    s=reference;k=s.material.order[0];s.empty(k);before=s.path.read_bytes();data=copy.deepcopy(s.data)
    writer=model._atomic_write_json
    monkeypatch.setattr(model,'_atomic_write_json',lambda *a:(_ for _ in ()).throw(OSError('synthetic disk failure')))
    with pytest.raises(OSError):s.restore(k)
    assert s.path.read_bytes()==before and s.data==data
    monkeypatch.setattr(model,'_atomic_write_json',writer);s.restore(k)
    assert s.state(k)['presence']=='UNRESOLVED'


def test_concurrent_writer_never_overwrites_existing_decisions(reference):
    s=reference;other=model.PhotoReference(s.material,s.path)
    s.empty(s.material.order[0]);before=s.path.read_bytes()
    with pytest.raises(ValueError,match='innym oknie'):other.empty(s.material.order[1])
    assert s.path.read_bytes()==before


def test_duplicate_sha_is_an_alias_not_a_second_task(reference):
    m=reference.material;row=m.images[m.order[0]]
    shutil.copyfile(m.root/row['relative_path'],m.root/'duplicate.jpg')
    reopened=model.Material(m.root)
    assert len(reopened.images)==40 and 'duplicate.jpg' in reopened.images[m.order[0]]['aliases']


def test_no_repeated_photo_hash_on_unchanged_switches(reference,monkeypatch):
    s=reference;original=model.digest;hashed=[]
    def count(p):
        if Path(p).suffix=='.jpg':hashed.append(str(p))
        return original(p)
    monkeypatch.setattr(model,'digest',count)
    for k in s.material.order[:5]:s.material.image(k);s.remember(k)
    assert not hashed


def test_no_writes_to_original_material_and_manifests(reference):
    s=reference;paths=[p for p in s.material.root.parent.rglob('*') if p.is_file()]
    before={str(p):p.read_bytes() for p in paths}
    s.empty(s.material.order[0]);s.add(s.material.order[1],quad())
    assert all(p.read_bytes()==before[str(p)] for p in paths)


def test_invalid_geometry_and_inconsistent_decisions_rejected(reference):
    s=reference;k=s.material.order[0]
    with pytest.raises(ValueError):s.add(k)
    with pytest.raises(ValueError):s.add(k,[[0,0]]*4)
    oid=s.add(k,quad())
    with pytest.raises(ValueError):s.edit(k,oid,text='X',reason='Nieczytelny')
    with pytest.raises(ValueError):s.empty(k)


def test_changed_manifest_and_foreign_store_are_not_overwritten(reference):
    s=reference;k=s.material.order[0];s.empty(k);saved=s.path.read_bytes()
    p=s.material.sort_path;p.write_text(p.read_text(encoding='utf-8')+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='Manifest'):s.remember(s.material.order[1])
    assert s.path.read_bytes()==saved
    s.path.write_text('{"schema":"foreign"}',encoding='utf-8')
    with pytest.raises(ValueError,match='kontrakt'):model.PhotoReference(s.material,s.path)
    assert s.path.read_text(encoding='utf-8')=='{"schema":"foreign"}'


def test_junction_is_not_traversed(reference,tmp_path):
    import subprocess
    if os.name!='nt':pytest.skip('Windows junction test')
    outside=tmp_path/'outside';outside.mkdir();Image.new('RGB',(10,10)).save(outside/'foreign.jpg')
    link=reference.material.root/'junction'
    result=subprocess.run(['cmd','/c','mklink','/J',str(link),str(outside)],capture_output=True)
    assert result.returncode==0,result.stderr
    try:
        assert not any('foreign.jpg' in str(p) for p in model._walk(reference.material.root))
        assert len(model.Material(reference.material.root).images)==40
    finally:
        os.rmdir(link)  # remove the junction only, never its external target
    assert (outside/'foreign.jpg').is_file()


def test_read_only_real_f5a_material(tmp_path):
    path=os.environ.get('ALPR_A1_REAL_MATERIAL')
    if not path:pytest.skip('Real F5A read-only source not supplied')
    material=model.Material(path)
    s=model.PhotoReference(material,tmp_path/'reference/reference.json')
    assert len(material.images)==40 and not s.path.exists()
    assert s.data['dataset_independence']['status']=='INCOMPLETE'


@pytest.fixture
def gui(reference):
    import tkinter as tk
    from auto_annotation_tool.gui.z4_photo_reference import PhotoReferenceWindow
    root=tk.Tk();root.withdraw();errors=[]
    import traceback
    root.report_callback_exception=lambda *a:errors.append(''.join(traceback.format_exception(*a)))
    h=PhotoReferenceWindow(root,reference.material,output=reference.path)
    root.update()
    yield root,h,h._canvas_editor,h.session,errors
    if not h._closed:h._canvas_editor.escape();h.close()
    root.update();root.destroy()
    assert not errors,errors


def test_native_existing_canvas_draw_inline_autosave_corners_and_resume(gui):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui.mobile_canvas_editor import CanvasReviewEditor
    root,h,ed,s,errors=gui;k=h.current_invocation
    assert type(ed) is CanvasReviewEditor and not hasattr(ed,'clean_bar') and not hasattr(s,'mt_invocations')
    assert len(h.subject_tree.get_children())==40 and h.canvas.winfo_width()>h.subject_tree.winfo_width()*2
    native.key(h,'d')
    for p in quad():native.click(h,ed,p)
    assert len(s.state(k)['objects'])==1 and ed.entry is not None
    oid=s.state(k)['objects'][0]['object_id']
    ed.entry.delete(0,'end');ed.entry.insert(0,'SYN123');ed.entry_typed()
    native.pump(root,lambda:s.state(k)['objects'][0]['gt_text']=='SYN123')
    ed.entry_enter();assert s.status(k)=='Gotowe'
    before=copy.deepcopy(s.state(k)['objects'][0]['quad']);ed._choose_plate(oid);root.update()
    x,y=map(round,ed.to_canvas(before[0]));h.canvas.event_generate('<ButtonPress-1>',x=x,y=y)
    assert ed.drag
    h.canvas.event_generate('<B1-Motion>',x=x+5,y=y+4);h.canvas.event_generate('<ButtonRelease-1>',x=x+5,y=y+4);root.update()
    assert s.state(k)['objects'][0]['quad']!=before
    ed.undo();assert s.state(k)['objects'][0]['quad']==before
    ed.undo(redo=True)
    h.select(s.material.order[1]);h.source.empty();assert h.progress.get()=='Gotowe: 2/40'
    assert s.data['last_image_id']==s.material.order[1]
    assert model.PhotoReference(s.material,s.path).data==s.data
    assert not errors


def test_native_na_incomplete_navigation_and_save_failure(gui,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s,errors=gui;k=h.current_invocation
    h.source.without_outline('Narożniki zasłonięte');root.update()
    assert ed.entry is not None and s.status(k)=='Wymaga uzupełnienia'
    h.source.unreadable();root.update();assert s.status(k)=='Gotowe'
    ed.undo();assert s.status(k)=='Wymaga uzupełnienia'
    oid=s.state(k)['objects'][0]['object_id'];ed.start_entry(oid)
    ed.entry.delete(0,'end');ed.entry.insert(0,'PENDING')
    writer=model._atomic_write_json
    monkeypatch.setattr(model,'_atomic_write_json',lambda *a:(_ for _ in ()).throw(OSError('synthetic disk failure')))
    assert not h.select(s.material.order[1]) and h.current_invocation==k
    assert ed.entry is not None and ed.entry.get()=='PENDING' and 'NIE ZAPISANO' in h.status_var.get()
    monkeypatch.setattr(model,'_atomic_write_json',writer)
    ed.entry_enter();assert s.state(k)['objects'][0]['gt_text']=='PENDING'
    native.key(h,'d');native.click(h,ed,(30,300))
    assert not h.select(s.material.order[1])
    ed.escape();assert h.select(s.material.order[1])
    assert not errors


def test_native_multiple_text_ctrl_s_delete_pan_zoom_and_reopen(gui):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui.z4_photo_reference import PhotoReferenceWindow
    root,h,ed,s,errors=gui;k=h.current_invocation
    for i in range(4):
        native.key(h,'d')
        for point in quad(i):native.click(h,ed,point)
        assert ed.entry is not None
        ed.entry.delete(0,'end');ed.entry.insert(0,f'SYN{i}')
        ed.entry.event_generate('<Control-s>');root.update()
        assert ed.entry is None
    assert len(s.state(k)['objects'])==4 and s.status(k)=='Gotowe'
    frozen=s.path.read_bytes();h.zoom_image(2);root.update()
    ed._choose_plate(s.state(k)['objects'][0]['object_id'])
    # Drag background while selected: the real pan engine has no Android dependency.
    x,y=map(round,ed.to_canvas((300,300)))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y)
    h.canvas.event_generate('<B1-Motion>',x=x+40,y=y+10)
    h.canvas.event_generate('<ButtonRelease-1>',x=x+40,y=y+10);root.update()
    assert s.path.read_bytes()==frozen and h._view_zoom==2
    ed.fit();root.update()
    ed._choose_plate(s.state(k)['objects'][0]['object_id']);ed.remove_selected();root.update()
    assert len(s.state(k)['objects'])==3
    ed.undo();assert len(s.state(k)['objects'])==4
    h.select(s.material.order[3]);assert h.current_invocation==s.material.order[3]
    h.close();root.update()
    reopened=PhotoReferenceWindow(root,s.material,output=s.path);root.update()
    try:
        assert reopened.current_invocation==s.material.order[3] and reopened.progress.get()=='Gotowe: 1/40'
        reopened.next_pending();assert reopened.current_invocation==s.material.order[4]
        reopened.window.geometry('1000x700');root.update()
        assert reopened.canvas.winfo_width()>650 and reopened.canvas.winfo_height()>500
    finally:reopened.close();root.update()


def test_native_failed_drawing_retains_draft_retry_once(gui,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s,errors=gui;k=h.current_invocation;writer=model._atomic_write_json
    monkeypatch.setattr(model,'_atomic_write_json',lambda *a:(_ for _ in ()).throw(OSError('disk')))
    native.key(h,'d')
    for p in quad():native.click(h,ed,p)
    assert len(ed.points)==4 and not s.state(k)['objects']
    assert not h.select(s.material.order[1])
    monkeypatch.setattr(model,'_atomic_write_json',writer);h.source.retry();root.update()
    assert len(s.state(k)['objects'])==1 and not ed.points and ed.entry is not None
    ed.entry.delete(0,'end');ed.entry.insert(0,'SYN');ed.entry_enter()
    assert s.status(k)=='Gotowe',(s.state(k),h.status_var.get(),ed.entry_target)
