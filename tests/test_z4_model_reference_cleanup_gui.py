"""Windows Tk acceptance for the simplified reference lifecycle, private data."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,time,os
import pytest,tkinter as tk
from PIL import ImageGrab
from windows_dpi import enable_windows_dpi_awareness
DPI=enable_windows_dpi_awareness()
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_ppm_seed_active_click import settled
from test_z4_bad_only_mt_final import source_badge
from test_z4_model_reference_cleanup import actual,OUT,SID,G,GOOD,BAD,P1,P2,state,frozen,cleaned,repair,core,stacks,partial,mt
from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
from auto_annotation_tool.gui import mobile_canvas_editor as editor,mobile_canvas_navigator as nav,mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers

def artifacts(s):
    result=Path(os.environ.get('FRAME_COMPLETION_OUTPUT',str(s.sidecar_path.parent/'artifacts')))
    result.mkdir(parents=True,exist_ok=True);return result

def retro_auto(s):
    while state(s)[0][stacks.HISTORY]['head']>6:assert stacks.restore_history(s,G)
    assert state(s)[1]['plates'][P2]['gt_quad']
    core.set_scene_review_state(s,G,checked=False)
    return s

@pytest.fixture
def gui(tmp_path,monkeypatch,actual,request):
    theme,width,height,mode=getattr(request,'param',('light',1649,896,'clean'))
    if mode=='auto':retro_auto(actual)
    elif mode!='before':cleaned(actual)
    if mode=='partial':core.set_scene_review_state(actual,G,checked=False)
    old=tk.Tk
    def make_root(*a,**kw):
        root=old(*a,**kw);root.tk.call('tk','scaling',120/72);return root
    monkeypatch.setattr(native.tk,'Tk',make_root)
    original=editor.palette
    monkeypatch.setattr(editor,'palette',lambda w,b:original(w,'#FFFFFF' if theme=='light' else '#17232D'))
    def namespace(**kw):
        if 'palette' in kw:kw['palette']=dict(panel='#F5F5F2',fg='#213547',muted='#5E7283',accent='#14628B') if theme=='light' else dict(panel='#17232D',fg='#EBF2F6',muted='#B5C4CE',accent='#77CBF0')
        return NS(**kw)
    monkeypatch.setattr(native,'SimpleNamespace',namespace)
    monkeypatch.setattr(native,'source_session',lambda _:actual)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    h.window.state('normal');h.window.geometry(f'{width}x{height}+10+10');h.window.deiconify();h.window.title('P0 GTR QA - '+theme+' '+str(width));root.update()
    iid=next(i for i,k in h._subject_ids.items() if G in subject_groups(s,k));h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry']);settled(root,h,ed)
    try:yield root,h,ed,s
    finally:
        if h._closed:monkeypatch.setattr(ed,'escape',lambda *a,**kw:None)
        try:next(gen)
        except StopIteration:pass

def widgets(root):
    result=[]
    for w in root.winfo_children():result.append(w);result.extend(widgets(w))
    return result

def screenshot(h,path):
    h.window.lift();h.window.update();time.sleep(.15);h.window.update()
    box=(h.window.winfo_rootx(),h.window.winfo_rooty(),h.window.winfo_rootx()+h.window.winfo_width(),h.window.winfo_rooty()+h.window.winfo_height())
    im=ImageGrab.grab(bbox=box,all_screens=True);im.save(path);return im.size

@pytest.mark.parametrize('gui',[(t,w,h,'before') for t in ('light','dark') for w,h in ((1649,896),(1180,760))],indirect=True)
def test_T45_four_native_configurations_before_after_cleanup(gui,request):
    root,h,ed,s=gui;theme,width,height,_=request.node.callspec.params['gui'];out=artifacts(s)/'native';out.mkdir(parents=True,exist_ok=True)
    nav.select_layer(ed,'GTR',GOOD);ready(root,h,ed)
    assert screenshot(h,out/f'{theme}_{width}x{height}_before.png')==(width,height)
    if P2 in state(s)[1]['plates']:
        backup=s.sidecar_path.parent/'exact.before.json';backup.write_bytes(s.sidecar_path.read_bytes())
        assert ed._commit(lambda:repair.apply(s,repair.prepare(s,G,P2),backup_path=backup),'QA cleanup')
    settled(root,h,ed);assert screenshot(h,out/f'{theme}_{width}x{height}_after.png')==(width,height)
    captions=[]
    for w in widgets(h.window):
        try:captions.append(str(w.cget('text')))
        except tk.TclError:pass
    assert not any('Wycofaj' in s or 'Wyjaśnij GTR' in s for s in captions)
    assert not hasattr(ed,'reference_resolution')
    assert [ed.contexts.counters[k].cget('text') for k in ('gt','tp','fp','fn')]==['1','1','1','0']
    assert set(state(s)[1]['plates'])=={P1}
    (out/f'{theme}_{width}x{height}.json').write_text(json.dumps(dict(theme=theme,size=[width,height],source=SID,review_revision=s.review.review_revision,counts=dict(gt=1,tp=1,fp=1,fn=0),dpi=DPI,input='native_Tk_event_generate_and_own_widget_API'),indent=2),encoding='utf-8')

@pytest.mark.parametrize('gui',[('light',1649,896,'partial')],indirect=True)
def test_T17_T18_T20_frames_complete_without_inline_check_or_modal(gui):
    root,h,ed,s=gui;line=ed.contexts.fields['image']
    revision=s.review.review_revision
    assert not hasattr(line,'_inspection_target') and not line.button.winfo_ismapped()
    assert s.review.review_revision==revision and not state(s)[0]['checked']
    assert [ed.contexts.counters[k].cget('text') for k in ('gt','tp','fp','fn')]==['1','1','1','0']
    assert not any(isinstance(w,tk.Toplevel) for w in h.window.winfo_children())

@pytest.mark.parametrize('gui',[('light',1649,896,'auto')],indirect=True)
def test_T46_T47_single_PPM_on_real_MT3_removes_only_working_GTR(gui):
    root,h,ed,s=gui;assert nav.select_layer(ed,'POSE',BAD);ready(root,h,ed)
    before=copy.deepcopy(state(s)[1]['plates'][P1]);parents=copy.deepcopy(state(s)[0]['parents']);revision=s.review.review_revision
    x,y=source_badge(h,ed,BAD)
    ed.c.event_generate('<ButtonPress-3>',x=round(x),y=round(y));ed.c.event_generate('<ButtonRelease-3>',x=round(x),y=round(y));settled(root,h,ed)
    sc,v=state(s)
    assert s.review.review_revision==revision+1 and set(v['plates'])=={P1} and v['plates'][P1]==before
    assert sc['parents']==parents and partial.progress(sc)['object_decisions'][BAD]==partial.NOT_PLATE
    assert not partial.progress(sc)['pending_gtr_ids'] and not sc['checked']
    assert ed.contexts.counters['gt'].cget('text')=='1' and ed.contexts.counters['fp'].cget('text')=='1'
    assert 'GTR 2' not in ed.contexts.fields['image'].cget('text')
    assert ed.c.find_withtag('fp_diagonal')
    ed.undo();settled(root,h,ed);assert P2 in state(s)[1]['plates']
    ed.undo(redo=True);settled(root,h,ed);assert P2 not in state(s)[1]['plates']
    assert nav.select_layer(ed,'POSE',BAD);ready(root,h,ed)
    assert ed.clean_gtr_confirm_button.cget('state')=='normal'
    native.key(h,'space');settled(root,h,ed)
    assert set(state(s)[1]['plates'])=={P1,P2} and partial.progress(state(s)[0])['object_decisions'][BAD]==partial.PLATE
    assert not partial.progress(state(s)[0])['pending_gtr_ids']
    x,y=source_badge(h,ed,BAD)
    ed.c.event_generate('<ButtonPress-3>',x=round(x),y=round(y));ed.c.event_generate('<ButtonRelease-3>',x=round(x),y=round(y));settled(root,h,ed)
    assert set(state(s)[1]['plates'])=={P1}

@pytest.mark.parametrize('gui',[('light',1649,896,'auto')],indirect=True)
def test_T30_ZX_and_Space_have_one_shared_decision(gui):
    root,h,ed,s=gui;ids=set(state(s)[1]['plates'])
    for kind in ('BBOX','POSE','GTR'):
        assert nav.select_layer(ed,kind,BAD);ready(root,h,ed)
        before=partial.progress(state(s)[0])['object_decisions'][BAD]
        native.key(h,'z');native.key(h,'x');ready(root,h,ed)
        native.key(h,'space');settled(root,h,ed)
        after=partial.progress(state(s)[0])['object_decisions'][BAD];assert after!=before
        native.key(h,'space');settled(root,h,ed)
        assert partial.progress(state(s)[0])['object_decisions'][BAD]==before and set(state(s)[1]['plates'])==ids

def test_T31_view_and_D_start_do_not_persist(gui):
    root,h,ed,s=gui;assert nav.select_layer(ed,'GTR',GOOD);ready(root,h,ed);before=frozen(s)
    for key in ('Shift_L','z','x','a'):
        native.key(h,key);ready(root,h,ed)
    h._view_center=(h._view_center[0]+2,h._view_center[1]+1);h._view_zoom*=1.01;h._paint_image();ready(root,h,ed)
    nav.select_layer(ed,'GTR',GOOD);native.key(h,'d');ready(root,h,ed);ed.escape();ready(root,h,ed)
    assert frozen(s)==before

def test_frame_completion_is_available_while_GTZ_is_missing(tmp_path,monkeypatch):
    from test_z4_mt_geometry_layers import source_session
    s=source_session(tmp_path/'data',mz=True);g='mt-one';partial.confirm_stack(s,g,oid='GT:a0')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        h.window.state('normal');h.window.geometry('1649x896+10+10');root.update();settled(root,h,ed)
        line=ed.contexts.fields['image'];assert not hasattr(line,'_inspection_target') and not line.button.winfo_ismapped()
        assert 'GTZ' in line.cget('text') and 'Sprawdź cały obraz' not in line.cget('text')
        assert not core.scene(s,g)['checked'] and mt.evaluate_unit(s,g)['tp']==1
        assert mt.evaluate_session(s)['ocr_summary']['n_reads']==0
        assert ed.contexts.counters['tp'].cget('text')=='1'
        folder=artifacts(s)/'native';folder.mkdir(parents=True,exist_ok=True)
        screenshot(h,folder/'GTZ_missing_frames_done.png')
    finally:
        try:next(gen)
        except StopIteration:pass
