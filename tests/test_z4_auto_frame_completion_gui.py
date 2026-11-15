"""Real Windows Tk completion/navigation/viewport checks on private sessions."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,time
import pytest,tkinter as tk
from PIL import ImageGrab
from windows_dpi import enable_windows_dpi_awareness
DPI=enable_windows_dpi_awareness()
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready as painted
from test_z4_ppm_seed_active_click import settled
from test_z4_bad_only_mt_final import source_badge
from test_z4_auto_frame_completion import OUT,ROOT,SID,G7,G5,G,OID,MANUAL,actual,red,source_session,core,partial,stacks,mt,frame,frozen,build_inventory
from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_editor as editor,mobile_canvas_clean_view as clean

@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    theme,width,height,mode=getattr(request,'param',('light',1649,896,'red'))
    data=tmp_path/'data';data.mkdir()
    if mode=='annex':
        from test_z4_nms_multiple_source_roles import actual06,G6
        s=actual06.__wrapped__(data);gid=G6
    elif mode=='proven_nms':
        from test_z4_nms_multiple_source_roles import synthetic,finish
        s=synthetic(data);finish(s);gid=G
    elif mode in ('red','nav'):s=red.__wrapped__(data);gid=G7
    elif mode=='actual':s=actual.__wrapped__(data);gid=G5
    else:s=source_session(data,n=0 if mode=='empty' else 2,mz=False);gid=G
    if mode=='nav':
        other=next(g for g in s.mt_invocations if g!=G7 and partial.progress(core.scene(s,g))['correct'])
        oid=next(k for k,v in partial.progress(core.scene(s,other))['object_decisions'].items() if v==partial.PLATE)
        partial.toggle_stack(s,other,oid=oid);s._test_navigation_target=other
    old=tk.Tk
    def make_root(*a,**kw):
        root=old(*a,**kw);root.tk.call('tk','scaling',120/72);return root
    monkeypatch.setattr(native.tk,'Tk',make_root)
    original=editor.palette
    monkeypatch.setattr(editor,'palette',lambda w,b:original(w,'#FFFFFF' if theme=='light' else '#17232D'))
    def namespace(**kw):
        if 'palette' in kw:kw['palette']=dict(panel='#F5F5F2',fg='#213547',muted='#5E7283',accent='#14628B') if theme=='light' else dict(panel='#17232D',fg='#EBF2F6',muted='#B5C4CE',accent='#77CBF0')
        return NS(**kw)
    monkeypatch.setattr(native,'SimpleNamespace',namespace);monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    h.window.state('normal');h.window.geometry(f'{width}x{height}+10+10');h.window.title('P0 Auto Frame QA');root.update()
    iid=next(i for i,k in h._subject_ids.items() if gid in subject_groups(s,k));h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry']);settled(root,h,ed)
    try:yield root,h,ed,s
    finally:
        if h._closed:monkeypatch.setattr(ed,'escape',lambda *a,**k:None)
        try:next(gen)
        except StopIteration:pass

@pytest.mark.parametrize('gui',[('light',1649,896,'annex')],indirect=True)
def test_N22_N24_N25_N26_actual_UNKNOWN_native_diagnostics(gui):
    from test_z4_nms_multiple_source_roles import G6
    root,h,ed,s=gui;before=frozen(s);task=ed.inventory()['groups'][G6]['review_tasks']
    assert not task['needs_review'] and task['units']==task['steps']==[] and 'N/A' in task['next_action']
    assert all(c['role']=='UNKNOWN' for c in stacks.snapshot(s,G6)['candidates'])
    assert ed.navigator.result.cget('text')=='Nieudowodnione role MT: 4'
    assert ed.contexts.counters['tp'].cget('text')==ed.contexts.counters['fp'].cget('text')==ed.contexts.counters['fn'].cget('text')=='—'
    assert not core.scene(s,G6)['checked'];no_image_action(h,ed)
    assert ed.c.find_withtag('image_na') and not ed.c.find_withtag('image_verified')
    ed.c.focus_force();ed.c.event_generate('<KeyPress-Shift_L>');painted(root,h,ed)
    assert ed.shift_preview.active
    assert len({f['attempt_id'] for f in ed.shift_preview.frames() if f['kind']=='BBOX'})==4
    assert len({r['frame']['attempt_id'] for r in ed.shift_preview_labels if r['frame']['kind']=='BBOX'})==4
    picture(h,'image06_Shift_four_UNKNOWN_light_1649x896.png')
    ed.c.event_generate('<KeyRelease-Shift_L>');painted(root,h,ed)
    assert len(stacks.snapshot(s,G6)['plates'])==1 and frozen(s)==before
    picture(h,'image06_source_NA_light_1649x896.png')

def all_widgets(root):
    result=[]
    for w in root.winfo_children():result.append(w);result.extend(all_widgets(w))
    return result
def no_image_action(h,ed):
    assert not ed.approve.winfo_ismapped() and not hasattr(ed,'image_inspection')
    for w in all_widgets(h.window):
        if not w.winfo_ismapped():continue
        try:text=str(w.cget('text'))
        except tk.TclError:continue
        assert 'Sprawdź cały obraz' not in text and 'Potwierdź przejrzenie' not in text
    assert not hasattr(ed.contexts.fields['image'],'_inspection_target')
def picture(h,name):
    out=OUT/'native';out.mkdir(exist_ok=True);h.window.update();time.sleep(.12);h.window.update()
    # The operator may keep another review window in front. Capture this own
    # native Tk client HWND rather than unrelated desktop pixels.
    im=ImageGrab.grab(window=h.window.winfo_id());im.save(out/name);return im.size

@pytest.mark.parametrize('gui',[(theme,w,h,'red') for theme in ('light','dark') for w,h in ((1649,896),(1180,760))],indirect=True)
def test_A01_D06_real_image07_four_native_configurations(gui,request):
    root,h,ed,s=gui;theme,width,height,_=request.node.callspec.params['gui'];before=frozen(s)
    assert nav.select_layer(ed,'GTR','GT:'+G7);painted(root,h,ed)
    no_image_action(h,ed)
    assert not core.scene(s,G7)['checked'] and [ed.contexts.counters[k].cget('text') for k in ('gt','tp','fp','fn')]==['1','1','1','0']
    assert 'Obraz zweryfikowany' in ed.contexts.fields['image'].cget('text')
    iid=h.subject_tree.selection()[0];assert 'Zweryfikowany' in str(h.subject_tree.item(iid))
    assert ed.inventory()['remaining']==0 and h.complete_button.cget('state')=='normal'
    assert picture(h,f'{theme}_{width}x{height}_image07.png')==(width,height)
    assert frozen(s)==before
    (OUT/'native'/f'{theme}_{width}x{height}.json').write_text(json.dumps(dict(status='PASS',theme=theme,size=[width,height],sid=SID,gid=G7,revision=s.review.review_revision,checked=False,counts=dict(gt=1,tp=1,fp=1,fn=0),input='native_Tk_event_generate_and_own_widget_API',physical_operator='NOT_RUN',dpi=DPI),indent=2),encoding='utf-8')

@pytest.mark.parametrize('gui',[('light',1649,896,'two')],indirect=True)
def test_A03_A06_B07_last_actual_Space_PPM_updates_every_view_once(gui):
    root,h,ed,s=gui;assert nav.select_layer(ed,'POSE',OID);painted(root,h,ed);revision=s.review.review_revision
    native.key(h,'space');settled(root,h,ed);assert s.review.review_revision==revision+1
    assert ed.inventory()['decisions']==1 and frame.for_session(s,G)['state']==frame.PENDING
    assert nav.select_layer(ed,'BBOX','GT:a1');painted(root,h,ed);x,y=source_badge(h,ed,'GT:a1');revision=s.review.review_revision
    ed.c.event_generate('<ButtonPress-3>',x=round(x),y=round(y));ed.c.event_generate('<ButtonRelease-3>',x=round(x),y=round(y));settled(root,h,ed)
    assert s.review.review_revision==revision+1 and ed.inventory()['remaining']==0
    assert not core.scene(s,G)['checked'] and frame.for_session(s,G)['state']==frame.READY
    assert [ed.contexts.counters[k].cget('text') for k in ('gt','tp','fp','fn')]==['1','1','1','0']
    no_image_action(h,ed);ed.undo();settled(root,h,ed);assert ed.inventory()['remaining']==1
    ed.undo(redo=True);settled(root,h,ed);assert ed.inventory()['remaining']==0

@pytest.mark.parametrize('gui',[('light',1649,896,'empty')],indirect=True)
def test_B01_B02_C01_C02_actual_D_and_corner_drag_need_no_Space(gui):
    root,h,ed,s=gui;before=frozen(s);assert frame.for_session(s,G)['state']==frame.NO_REFERENCE
    native.key(h,'space');painted(root,h,ed);assert frozen(s)==before
    revision=s.review.review_revision;native.key(h,'d');painted(root,h,ed)
    for p in MANUAL:native.click(h,ed,p)
    settled(root,h,ed);assert s.review.review_revision==revision+1
    v=stacks.snapshot(s,G);pid=next(iter(v['plates']));oid=v['plates'][pid]['origin_object_id']
    assert frame.for_session(s,G)['state']==frame.READY and mt.evaluate_unit(s,G)['fn']==1
    q=v['plates'][pid]['gt_quad'];center=tuple(sum(p[i] for p in q)/4 for i in (0,1))
    native.click(h,ed,center);painted(root,h,ed)
    assert clean.selected(ed)['pid']==pid
    native.key(h,'d');painted(root,h,ed)
    q=clean.selected(ed)['gt'];x,y=map(round,ed.to_canvas(q[0]));revision=s.review.review_revision
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag
    ed.c.event_generate('<B1-Motion>',x=x+4,y=y+2);ed.c.event_generate('<ButtonRelease-1>',x=x+4,y=y+2);settled(root,h,ed)
    assert s.review.review_revision==revision+1 and set(stacks.snapshot(s,G)['plates'])=={pid}
    assert frame.for_session(s,G)['state']==frame.READY and mt.evaluate_unit(s,G)['fn']==1
    no_image_action(h,ed)

def test_B10_D07_view_and_old_action_area_are_read_only(gui):
    root,h,ed,s=gui;before=frozen(s);no_image_action(h,ed)
    assert nav.select_layer(ed,'GTR','GT:'+G7);painted(root,h,ed)
    for k in ('z','x','Shift_L','s','a'):native.key(h,k);painted(root,h,ed)
    h._view_zoom*=1.01;h._view_center=(h._view_center[0]+2,h._view_center[1]+1);h._paint_image();painted(root,h,ed)
    # Task text formerly hosted the image check link; it has no action now.
    line=ed.contexts.fields['image'];line.event_generate('<ButtonPress-1>',x=5,y=5);line.event_generate('<ButtonRelease-1>',x=5,y=5)
    ed.approve.invoke();painted(root,h,ed);assert frozen(s)==before

@pytest.mark.parametrize('gui',[('light',1649,896,'nav')],indirect=True)
def test_B11_navigation_is_not_a_review_write(gui):
    root,h,ed,s=gui;before=frozen(s);target=s._test_navigation_target
    ed.next_pending();native.pump(root,lambda:h.current_invocation==target);settled(root,h,ed)
    assert frozen(s)==before and frame.for_session(s,target)['state']==frame.PENDING
