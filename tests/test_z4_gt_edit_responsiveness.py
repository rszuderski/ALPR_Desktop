"""Persistent GT input and a cheap, read-only mouse-move path in real Tk."""
from pathlib import Path
import cProfile,copy,json,pstats,shutil,statistics,time
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_clean_canvas import pair,pick
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean
from auto_annotation_tool.gui.mobile_canvas_stacks import view
from auto_annotation_tool.ranking import mobile_session_metrics as live

OUT=Path(__file__).resolve().parents[1]/'output/Z4_GT_EDIT_RESPONSIVENESS_20261026_001'

@pytest.fixture
def current_edit_gui(tmp_path,monkeypatch):
    b=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))
    row=next(r for r in b['sessions'] if r['session_id']=='exp-1792702658896-47b4e60b')
    sidecar=tmp_path/'private.review.json';shutil.copy2(row['sidecar'],sidecar)
    s=MobileReviewSession.open(Path(row['path']),sidecar_path=sidecar)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);value=next(gen)
    try:yield value
    finally:
        try:next(gen)
        except StopIteration:pass

def start_corner(root,h,ed,oid=None):
    oid=oid or next(c['object_id'] for c in view(ed)['candidates'] if c['role']=='FINAL' and c['plate_id'])
    ed._choose_plate(oid);ed.mode='edit';ready(root,h,ed)
    quad=copy.deepcopy(core._object(ed._read(),oid)['quad'])
    x,y=ed.to_canvas(quad[0]);native.focus_canvas(h)
    ed._test_press_time=getattr(ed,'_test_press_time',1000)+1000
    h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y),time=ed._test_press_time)
    native.pump(root);assert ed.gt_focus_id==oid and ed.drag
    return oid,x,y

def open_badge(root,h,ed,oid):
    assert oid in ed.badges
    l,t,r,b=ed.badges[oid];x,y=round((l+r)/2),round((t+b)/2)
    assert ed.hit(x,y)==(oid,'badge',None)
    ed._test_press_time=getattr(ed,'_test_press_time',1000)+1000
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y,time=ed._test_press_time)
    h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y,time=ed._test_press_time+10)
    native.pump(root);assert ed.entry is not None and ed.entry.winfo_ismapped()

def wait_saved(root,h,ed):
    native.pump(root,lambda:not h._saving and not ed.in_refresh and not ed.refresh_after
        and live.current(h.stats.get('live_session_metrics_v1'),live.result_key(h.session)),seconds=20)
    ready(root,h,ed)

def test_active_gt_badge_is_visible_clickable_and_edit_keeps_corner_focus(current_edit_gui):
    root,h,ed,s=current_edit_gui;oid,x,y=start_corner(root,h,ed)
    parents=copy.deepcopy(ed._read()['parents'])
    assert oid in ed.badges and h.canvas.find_withtag('gt_badge')
    assert ed.inf_visible.get() and h.canvas.find_withtag('mt_original')
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    open_badge(root,h,ed,oid);assert ed.gt_focus_id==oid
    ed.entry_var.set('VISIBLE94');ed.entry.event_generate('<Return>');wait_saved(root,h,ed)
    assert ed.gt_focus_id==oid and oid in ed.badges
    assert core._object(core.scene(s,h.current_invocation),oid)['gt_text']=='VISIBLE94'
    assert ed._read()['parents']==parents

def test_escape_cancels_text_first_then_leaves_corner_mode(current_edit_gui):
    root,h,ed,s=current_edit_gui;oid,x,y=start_corner(root,h,ed)
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    before=s.sidecar_path.read_bytes();open_badge(root,h,ed,oid)
    ed.entry_var.set('UNSAVED');ed.entry.event_generate('<Escape>');ready(root,h,ed)
    assert ed.entry is None and ed.gt_focus_id==oid and oid in ed.badges
    assert s.sidecar_path.read_bytes()==before
    native.key(h,'Escape');ready(root,h,ed)
    assert not ed.gt_focus_id and ed.selected==oid and s.sidecar_path.read_bytes()==before

def test_focus_gt_input_survives_zoom_and_toolbar_fallback(current_edit_gui):
    root,h,ed,s=current_edit_gui;oid,x,y=start_corner(root,h,ed)
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    before=s.sidecar_path.read_bytes()
    for size in ('1649x896','1180x760'):
        h.window.geometry(size+'+10+10');ed.fit();ready(root,h,ed);open_badge(root,h,ed,oid)
        entry=ed.entry;quad=core._object(ed._read(),oid)['quad']
        h._view_center=tuple(sum(p[i] for p in quad)/4*(h._image.size[i]/h._image_source_size[i]) for i in (0,1))
        h._view_zoom=100;h._paint_image();native.pump(root)
        assert ed.gt_focus_id==oid and ed.entry is entry and entry.winfo_ismapped()
        assert ed.clean_gt_slot.winfo_ismapped() and entry.place_info()['in'] is ed.clean_gt_slot
        ed.fit();ready(root,h,ed)
        assert entry.winfo_ismapped() and entry.place_info()['in'] is h.canvas
        entry.event_generate('<Return>');ready(root,h,ed)
    assert s.sidecar_path.read_bytes()==before

def test_mouse_motion_does_not_rebuild_history_panel_or_session_metrics(current_edit_gui,monkeypatch):
    root,h,ed,s=current_edit_gui;oid,x,y=start_corner(root,h,ed)
    frozen=s.sidecar_path.read_bytes();revision=s.review.review_revision
    original=copy.deepcopy(ed._read()['parents']);metrics=copy.deepcopy(ed.contexts.live_metrics.data)
    def heavy(*a,**kw):pytest.fail('Heavy saved-state work in the mouse-move path')
    with monkeypatch.context() as guard:
        for obj,name in ((core,'scene'),(stacks,'diagnostics'),(s,'statistics'),
                         (ed,'_gt_policy'),(ed,'_render_contexts'),(ed.contexts.live_metrics,'render')):
            guard.setattr(obj,name,heavy)
        for i in range(20):
            h.canvas.event_generate('<B1-Motion>',x=round(x+1+i/3),y=round(y+2))
            root.update_idletasks()
            assert oid in ed.badges and ed.contexts.live_metrics.data==metrics
        assert 'niezapisana korekta' in ed.contexts.fields['object'].cget('text')
    assert s.sidecar_path.read_bytes()==frozen and s.review.review_revision==revision
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x+7),y=round(y+2));wait_saved(root,h,ed)
    assert s.review.review_revision==revision+1 and ed._read()['parents']==original
    assert h.stats['live_session_metrics_v1']['review_revision']==s.review.review_revision

def test_display_cache_does_not_replace_fresh_input_validation(current_edit_gui,monkeypatch):
    root,h,ed,s=current_edit_gui;oid,x,y=start_corner(root,h,ed)
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    assert clean.text_editable(ed,oid)
    before=s.sidecar_path.read_bytes();calls=[]
    def denied(target):calls.append(target);return dict(gt_editable=False,gt_reason='Source validation test')
    monkeypatch.setattr(ed,'_gt_policy',denied)
    ed.start_entry(oid)
    assert calls and ed.entry is None and s.sidecar_path.read_bytes()==before

def test_shared_gt_entry_preserves_focused_candidate_and_reference_identity(pair):
    root,h,ed,s,gid=pair;oid=pick(pair,1);_,x,y=start_corner(root,h,ed,oid)
    h.canvas.event_generate('<B1-Motion>',x=round(x+3),y=round(y+1));root.update_idletasks()
    assert 'niezapisana korekta' in ed.contexts.fields['object'].cget('text')
    assert 'IoU ' in ed.contexts.fields['object'].cget('text') and 'NMCE ' in ed.contexts.fields['object'].cget('text')
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    parents=copy.deepcopy(ed._read()['parents']);open_badge(root,h,ed,oid)
    assert ed.entry_target==(gid,'GT:a0') and ed.gt_focus_id==oid
    ed.entry_var.set('SHAREDFOCUS');ed.entry.event_generate('<Return>');ready(root,h,ed)
    assert ed.gt_focus_id==oid and len(stacks.snapshot(s,gid)['plates'])==1
    assert all(r['gt_text']=='SHAREDFOCUS' for r in stacks.diagnostics(s,[gid])['rows'])
    assert ed._read()['parents']==parents

def test_profile_current_f4_after(current_edit_gui):
    root,h,ed,s=current_edit_gui;oid,x,y=start_corner(root,h,ed)
    quad=copy.deepcopy(ed.drag['before']);frozen=s.sidecar_path.read_bytes();times=[]
    profiler=cProfile.Profile();profiler.enable()
    for i in range(20):
        ed.drag['quad'][0][0]=quad[0][0]+i/10
        start=time.perf_counter();ed.overlay();root.update_idletasks();times.append(1000*(time.perf_counter()-start))
    profiler.disable();profiler.dump_stats(str(OUT/'drag_after.prof'))
    with (OUT/'drag_after_profile.txt').open('w',encoding='utf-8') as stream:
        pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(28)
    result=dict(frames=len(times),median_ms=statistics.median(times),max_ms=max(times),mean_ms=statistics.mean(times),
        gt_badge_visible=bool(ed.badges),toolbar_gt_visible=bool(ed.clean_gt_slot.winfo_ismapped()))
    (OUT/'drag_after.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    before=json.loads((OUT/'drag_before.json').read_text(encoding='utf-8'))
    assert result['median_ms']<before['median_ms']/2
    assert s.sidecar_path.read_bytes()==frozen and result['gt_badge_visible']
    ed.drag=None
