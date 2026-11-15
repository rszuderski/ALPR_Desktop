"""Global REF/INF presentation on real Tk; all writes use private fixtures."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_one_navigator import gui as linked_gui, bbox as bbox_gui, multiple
from test_z4_gtr_collisions_replace import gui as collision_gui, Q1,Q2,center,advance,click,rect
from test_z4_mt_geometry_layers import source_session
from test_z4_p0_physical_gt import session_fixture
from auto_annotation_tool.gui import mobile_canvas_visibility as visibility, mobile_canvas_navigator as nav, mobile_canvas_geometry_layers as layers, mobile_canvas_clean_view as clean
from auto_annotation_tool.ranking import mobile_canvas_review as core, mobile_candidate_stacks as data
from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit

G='mt-one';OID='GT:a0'
def frozen(s):return copy.deepcopy(s.review.to_dict()),s.sidecar_path.read_bytes()
def toggle(root,h,ed,ref,inf):
    ed.ref_visible.set(ref);ed.inf_visible.set(inf);clean.refresh_contours(ed)
    native.pump(root);ed.overlay()
def kinds(ed):return [item['kind'] for item in visibility.contours(ed)]
def rendered(ed):return sorted([ed.c.gettags(i)[3] for i in ed.c.find_withtag('navigator_geometry')],key=('GTR','POSE','BBOX').index)

def test_V01_V02_V04_defaults_all_layers_and_exact_widths(linked_gui):
    root,h,ed,s=linked_gui;ready(root,h,ed)
    assert ed.ref_visible.get() and ed.inf_visible.get()
    assert layers.model(ed)['active']=='POSE' and rendered(ed)==['GTR','POSE']
    assert not ed.c.find_withtag('model_location_bbox')
    assert [float(ed.c.itemcget(i,'width')) for i in ed.c.find_withtag('navigator_geometry')]==[3,2.5]
    assert all(ed.c.itemcget(i,'fill')=='' for i in ed.c.find_withtag('navigator_geometry'))

def test_V03_invalid_pose_uses_only_source_bbox(bbox_gui):
    root,h,ed,s=bbox_gui;ready(root,h,ed)
    assert rendered(ed)==['GTR','BBOX'] and not ed.c.find_withtag('mt_original')
    bbox=ed.c.find_withtag('model_location_bbox')[0]
    assert float(ed.c.itemcget(bbox,'width'))==2 and ed.c.itemcget(bbox,'dash')
    row=data.diagnostics(s,[G])['rows'][0]
    assert row['mt_quad'] is None and row['polygon_iou_pct'] is row['nmce_pct'] is None

@pytest.mark.parametrize('ref,inf',[(True,True),(True,False),(False,True),(False,False)])
def test_V09_to_V12_V35_to_V37_flags_are_visual_only(linked_gui,ref,inf):
    root,h,ed,s=linked_gui;ready(root,h,ed);before=frozen(s)
    metrics=copy.deepcopy((core.summary(s),data.diagnostics(s),evaluate_unit(s,G)))
    selected=(ed.selected,layers.model(ed)['active']);hud=ed.source_hud.cget('text')
    from auto_annotation_tool.gui.mobile_canvas_info_badge import badge_rows
    panel=clean.context(ed,{});badges=copy.deepcopy(badge_rows(ed))
    toggle(root,h,ed,ref,inf)
    expected=(['GTR'] if ref else [])+(['POSE'] if inf else [])
    assert rendered(ed)==expected and frozen(s)==before
    assert (core.summary(s),data.diagnostics(s),evaluate_unit(s,G))==metrics
    assert (ed.selected,layers.model(ed)['active'])==selected and ed.source_hud.cget('text')==hud
    assert clean.context(ed,{})==panel and badge_rows(ed)==badges
    for _ in range(3):toggle(root,h,ed,True,True)
    assert rendered(ed)==['GTR','POSE'] and frozen(s)==before

def test_V05_V07_multi_final_and_shared_reference_once(multiple):
    root,h,ed,s=multiple;ready(root,h,ed);before=frozen(s)
    assert kinds(ed).count('GTR')==1 and kinds(ed).count('POSE')==3
    assert {r['oid'] for r in visibility.contours(ed) if r['kind']=='POSE'}=={'GT:a0','GT:a1','GT:a2'}
    nav.select_layer(ed,'BBOX');ready(root,h,ed)
    assert kinds(ed).count('GTR')==1 and kinds(ed).count('POSE')==2 and kinds(ed).count('BBOX')==1
    assert frozen(s)==before

def test_V08_suppressed_raw_is_not_an_extra_final(tmp_path,monkeypatch):
    s=session_fixture(tmp_path/'data',n=2,suppressed=1);s.set_review_mode('assisted')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);v=data.snapshot(s,G);suppressed={r['object_id'] for r in v['candidates'] if r['role']=='SUPPRESSED'}
        assert suppressed and not (suppressed & {r['oid'] for r in visibility.contours(ed)})
    finally:
        try:next(gen)
        except StopIteration:pass

def test_V15_V16_V17_layer_selection_does_not_enable_flags(linked_gui):
    root,h,ed,s=linked_gui;ready(root,h,ed);before=frozen(s)
    for ref,inf in ((True,True),(False,True),(True,False),(False,False)):
        toggle(root,h,ed,ref,inf)
        for kind in ('BBOX','POSE','GTR'):
            assert nav.select_layer(ed,kind);ready(root,h,ed)
            assert (ed.ref_visible.get(),ed.inf_visible.get())==(ref,inf)
            assert rendered(ed)==(['GTR'] if ref else [])+([kind if kind=='BBOX' else 'POSE'] if inf else [])
            assert bool(ed.c.find_withtag('corner_handle'))==(ref and kind=='GTR')
    assert frozen(s)==before

def test_V17_hidden_reference_cannot_be_dragged_by_invisible_handle(linked_gui):
    root,h,ed,s=linked_gui;ready(root,h,ed);nav.select_layer(ed,'GTR');ready(root,h,ed)
    toggle(root,h,ed,False,False);q=data.snapshot(s,G)['plates'];p=next(iter(q.values()))['gt_quad'][0]
    hit=nav.hit(ed,*ed.to_canvas(p))
    assert hit is None or hit[1] not in ('corner','layer:GTR')

@pytest.mark.parametrize('count',[1,2,3])
def test_V20_V21_V22_draft_escape_and_shift_preserve_flags(collision_gui,count):
    root,h,ed,s,a,b=collision_gui;ed.escape();ready(root,h,ed);before=frozen(s)
    toggle(root,h,ed,False,False);native.key(h,'d');ready(root,h,ed)
    for p in ((540,520),(580,520),(580,550))[:count]:click(root,h,ed,p)
    points=copy.deepcopy(ed.points)
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert ed.c.find_withtag('shift_saved_gtr') and ed.c.find_withtag('shift_source_preview')
    ed.shift_preview.focus_out();ed.overlay()
    assert not ed.c.find_withtag('shift_source_preview') and ed.points==points
    native.key(h,'Escape');ready(root,h,ed)
    assert not ed.points and not rendered(ed) and not ed.ref_visible.get() and not ed.inf_visible.get() and frozen(s)==before

@pytest.mark.parametrize('ref',[False,True])
def test_V23_V24_V27_collision_uses_data_and_preserves_draft(collision_gui,ref):
    root,h,ed,s,a,b=collision_gui;ed.escape();ready(root,h,ed);before=frozen(s)
    toggle(root,h,ed,ref,False);native.key(h,'d');ready(root,h,ed)
    click(root,h,ed,(540,520));points=copy.deepcopy(ed.points)
    click(root,h,ed,center(Q1))
    assert ed.points==points and ed.collision_warning.active and ed.c.cget('cursor')=='crosshair'
    assert ed.c.find_withtag('gtr_collision_warning') and frozen(s)==before

def test_V25_V26_hidden_collision_two_pulses_and_renewed_timeout(collision_gui,monkeypatch):
    root,h,ed,s,a,b=collision_gui;ed.escape();ready(root,h,ed);toggle(root,h,ed,False,False)
    before=frozen(s);native.key(h,'d');ready(root,h,ed);w=ed.collision_warning;trace=[];paint=w.paint
    def observed():trace.append(w.phase);return paint()
    monkeypatch.setattr(w,'paint',observed);click(root,h,ed,center(Q1));started=w.started_at
    advance(root,.85);assert w.active and w.phase=='normal'
    click(root,h,ed,center(Q1));assert w.started_at==started
    advance(root,1.95);assert w.active
    advance(root,.65);assert not w.active and not w.after_ids and not ed.c.find_withtag('gtr_collision_warning')
    transitions=[p for i,p in enumerate(trace) if not i or trace[i-1]!=p]
    assert transitions==['red','normal','red','normal'] and frozen(s)==before and not ed.ref_visible.get()

def test_V28_V29_replacement_exempts_self_only(collision_gui):
    root,h,ed,s,a,b=collision_gui;ed._choose_plate(a);ready(root,h,ed);toggle(root,h,ed,False,False)
    before=frozen(s);native.key(h,'d');ready(root,h,ed);assert ed.draw_target['kind']=='replace'
    click(root,h,ed,center(Q1));assert len(ed.points)==1
    click(root,h,ed,center(Q2));assert len(ed.points)==1 and ed.collision_warning.active and frozen(s)==before

def test_V31_enclosure_blocks_final_point_when_ref_hidden(collision_gui):
    root,h,ed,s,a,b=collision_gui;ed.escape();ready(root,h,ed);toggle(root,h,ed,False,True)
    before=frozen(s);native.key(h,'d');ready(root,h,ed)
    for point in rect(620,330,830,440):click(root,h,ed,point)
    assert len(ed.points)==3 and ed.collision_warning.active and frozen(s)==before

@pytest.mark.parametrize('action',['escape','focus_out','mode','dispose','undo','redo'])
def test_V32_warning_cancelled_without_timer_leak(collision_gui,action):
    root,h,ed,s,a,b=collision_gui;ed.escape();ready(root,h,ed);toggle(root,h,ed,False,False)
    native.key(h,'d');ready(root,h,ed);click(root,h,ed,center(Q1));assert ed.collision_warning.active
    if action=='focus_out':ed.c.event_generate('<FocusOut>');root.update()
    elif action=='mode':ed.draw_toggle()
    elif action in ('undo','redo'):ed.undo(redo=action=='redo')
    else:getattr(ed,action)()
    assert not ed.collision_warning.active and not ed.collision_warning.after_ids

def test_V19_draw_completion_restores_global_contours(collision_gui):
    root,h,ed,s,a,b=collision_gui;ed.escape();ready(root,h,ed);native.key(h,'d');ready(root,h,ed)
    for p in rect(540,510,600,550):click(root,h,ed,p)
    ready(root,h,ed)
    assert ed.mode!='draw' and kinds(ed).count('GTR')==3 and 'POSE' in kinds(ed)
    assert ed.ref_visible.get() and ed.inf_visible.get()

def test_V39_bad_roundtrip_on_explicit_source_badge(linked_gui):
    root,h,ed,s=linked_gui;ready(root,h,ed)
    nav.select_layer(ed,'BBOX');ready(root,h,ed)
    refs=copy.deepcopy(data.snapshot(s,G)['plates'])
    # PPM source badge is unambiguous even when the GTR overlaps its source.
    box=ed.mt_source_badge_hits[OID];x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
    assert nav.right_hit(ed,x,y)==OID
    ed.right_press(NS(widget=ed.c,x=round(x),y=round(y),state=0));ready(root,h,ed)
    assert core._object(ed._read(),OID)['error'] and data.snapshot(s,G)['plates']==refs

@pytest.mark.parametrize('ref',[False,True])
def test_V39_PPM_on_saved_reference_preserves_draft(collision_gui,ref):
    root,h,ed,s,a,b=collision_gui;ed.escape();ready(root,h,ed);toggle(root,h,ed,ref,False)
    native.key(h,'d');ready(root,h,ed);click(root,h,ed,(540,520))
    before=frozen(s);points=copy.deepcopy(ed.points)
    native.click(h,ed,center(Q1),button=3);ready(root,h,ed)
    assert ed.mode=='draw' and ed.points==points and frozen(s)==before

def test_V22_window_release_and_real_focus_change_clear_preview(linked_gui):
    root,h,ed,s=linked_gui;ed.escape();ready(root,h,ed);toggle(root,h,ed,False,False)
    native.key(h,'d');ready(root,h,ed);before=frozen(s)
    ed.c.event_generate('<KeyPress-Shift_L>');root.update();ed.overlay()
    assert ed.shift_preview.active
    h.window.event_generate('<KeyRelease-Shift_L>');root.update();ed.overlay()
    assert not ed.shift_preview.active and not ed.c.find_withtag('shift_source_preview')
    ed.c.event_generate('<KeyPress-Shift_L>');root.update();ed.overlay()
    assert ed.shift_preview.active
    ed.navigator.result.focus_force();root.update();ed.overlay()
    assert not ed.shift_preview.active and not ed.c.find_withtag('shift_source_preview') and frozen(s)==before

def test_V39_initial_overlap_is_not_explicit_source_selection(linked_gui):
    root,h,ed,s=linked_gui;ready(root,h,ed);before=frozen(s)
    q=next(r['quad'] for r in visibility.contours(ed) if r['kind']=='GTR')
    p=tuple(sum(point[i] for point in q)/4 for i in (0,1))
    native.click(h,ed,p,button=3);ready(root,h,ed)
    assert frozen(s)==before and 'GTR' in h.status_var.get()
    toggle(root,h,ed,False,True)
    assert nav.right_hit(ed,*ed.to_canvas(p))==OID
