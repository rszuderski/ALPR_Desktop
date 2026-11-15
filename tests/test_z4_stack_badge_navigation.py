"""Source MT candidates of one physical GT: badge and native navigation."""
from pathlib import Path
from types import SimpleNamespace
import copy,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import stack_gui,stacks,ready,linked
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_stacks as ui
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession

OUT=Path(__file__).resolve().parents[1]/'output/Z4_STACK_BADGE_NAV_20261028_001'
G='exp-1792699303016-c8b05efb-a00000010'

@pytest.fixture
def real_gui(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    side=tmp_path/'copy.review.json';shutil.copyfile(OUT/'private/baseline.review.json',side)
    s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=side)
    monkeypatch.setattr(native,'source_session',lambda p:s)
    for root,h,ed,s in native.gui.__wrapped__(tmp_path,monkeypatch):
        iid=next(i for i in h.subject_tree.get_children('') if G in subject_groups(s,h._subject_ids[i]))
        h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry'] and not h._saving)
        ed._choose_plate('GT:'+G);ready(root,h,ed)
        yield root,h,ed,s

def click_control(h,ed,back=False):
    box,oid,_=next(x for x in ed.mt_stack_controls if x[2]==back)
    x,y=round((box[0]+box[2])/2),round((box[1]+box[3])/2)
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y,time=1000+getattr(ed,'_test_click_time',0))
    h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y,time=1020+getattr(ed,'_test_click_time',0))
    ed._test_click_time=getattr(ed,'_test_click_time',0)+1000;h.window.update()

def badge(h):return [h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('mt_stack_count')]

def test_N01_badge_on_MT_without_selection(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('');ready(root,h,ed)
    assert badge(h)==['MT 1/3'] and len(ed.mt_stack_controls)==2
    assert 'GT:a0' in ed.mt_stack_badges

def test_N02_click_controls_both_directions_wrap_read_only(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('');ready(root,h,ed)
    before=s.sidecar_path.read_bytes();summary=core.summary(s);undo=copy.deepcopy(ed.undo_stack)
    for back,oid in [(False,'GT:a1'),(False,'GT:a2'),(False,'GT:a0'),(True,'GT:a2'),(True,'GT:a1')]:
        click_control(h,ed,back);ready(root,h,ed);assert ed.selected==oid
        assert len(h.canvas.find_withtag('mt_original'))==1 and len(h.canvas.find_withtag('operator_quad'))==1
    assert s.sidecar_path.read_bytes()==before and core.summary(s)==summary and ed.undo_stack==undo

def test_N03_arrows_and_ZX_switch_same_final_candidates(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    frozen=s.sidecar_path.read_bytes()
    for key,oid in [('Down','GT:a1'),('Down','GT:a2'),('Down','GT:a0'),('Up','GT:a2'),('z','GT:a0'),('x','GT:a2')]:
        native.key(h,key);ready(root,h,ed);assert ed.selected==oid
        assert f'MT {int(oid[-1])+1}/3' in badge(h)
        candidate=clean.selected(ed)['candidate']
        parent=next(p for p in ed._read()['parents'] if p['attempt_id']==candidate['attempt_id'])
        assert candidate['mz_prediction']==parent['mz_prediction']
    assert s.sidecar_path.read_bytes()==frozen

def test_N04_arrows_do_not_steal_text_or_image_list(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    ed.start_entry('GT:a0');native.pump(root);ed.entry.focus_force();before=ed.selected
    ed.entry.event_generate('<KeyPress-Down>');native.pump(root)
    assert ed.selected==before and ed.entry is not None
    assert ed.stack_arrow_hotkey(SimpleNamespace(widget=h.subject_tree,keysym='Down',state=0)) is None

def test_N05_clean_edit_hides_and_blocks_stack(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    q=ed._current_quad(core._object(ed._read(),ed.selected));x,y=map(round,ed.to_canvas(q[0]))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);native.pump(root);ed.overlay()
    assert ed.gt_focus_id and not h.canvas.find_withtag('mt_stack_count') and not ed.mt_stack_controls
    assert ui.cycle(ed,SimpleNamespace(widget=ed.c),False)=='break' and ed.selected=='GT:a0'
    h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y);ed.escape();ready(root,h,ed)
    assert badge(h)==['MT 1/3']

def test_N06_badge_remains_in_view_during_zoom_pan(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a1');ready(root,h,ed)
    before=s.sidecar_path.read_bytes()
    for zoom in (1.0,2.0,6.0):
        h._view_zoom=zoom;h._paint_image();ready(root,h,ed)
        assert badge(h)==['MT 2/3']
        for l,t,r,b in ed.mt_stack_badges.values():
            assert 0<=l<r<=h.canvas.winfo_width() and 0<=t<b<=h.canvas.winfo_height()
    assert s.sidecar_path.read_bytes()==before

def test_N07_real_image10_single_final_no_false_second_candidate(real_gui):
    root,h,ed,s=real_gui;before=s.sidecar_path.read_bytes();scene=copy.deepcopy(ed._read())
    assert badge(h)==['MT 1/1'] and not ed.mt_stack_controls
    assert len(clean.records(ed))==2 and len(data.snapshot(s,G)['plates'])==2
    for key in ('Down','Up','z','x'):native.key(h,key);ready(root,h,ed)
    assert ed._read()==scene and s.sidecar_path.read_bytes()==before

def test_N08_pending_link_still_cycles(stack_gui):
    from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
    root,h,ed,s=stack_gui;partial.toggle_plate(s,'mt-one','GT:a0');ed._choose_plate('GT:a0');ready(root,h,ed)
    frozen=s.sidecar_path.read_bytes()
    native.key(h,'Down');ready(root,h,ed)
    assert ed.selected=='GT:a1' and badge(h)==['MT 2/3'] and s.sidecar_path.read_bytes()==frozen

def test_N09_RAW_not_in_stack_and_two_GT_stay_separate(tmp_path):
    s=linked(tmp_path,data,n=3,suppressed=2)
    oid=core.add_plate(s,'mt-one',[[600,300],[800,300],[800,350],[600,350]],partial_review=True)
    snapshot=data.snapshot(s,'mt-one');pid=snapshot['links']['GT:a0']['plate_id']
    assert len(snapshot['plates'])==2
    assert [c['object_id'] for c in ui.members(snapshot,pid)]==['GT:a0','GT:a1']
    assert oid not in [c['object_id'] for c in ui.members(snapshot,pid)]

def test_N10_PPM_on_count_targets_current_source(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a1');ready(root,h,ed)
    l,t,r,b=ed.mt_stack_badges['GT:a1'];x,y=l+8,(t+b)/2
    assert clean.right_hit(ed,x,y)=='GT:a1'

def test_N11_switch_after_corner_release_exits_focus(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    ed.gt_focus_id=ed.selected;ed.overlay();frozen=s.sidecar_path.read_bytes()
    native.key(h,'Down');ready(root,h,ed)
    assert ed.selected=='GT:a1' and not ed.gt_focus_id and badge(h)==['MT 2/3']
    assert s.sidecar_path.read_bytes()==frozen

def test_N12_click_during_GTZ_typing_does_not_commit_or_lose_focus(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    ed.start_entry('GT:a0');ed.entry_var.set('UNSAVED');ed.entry.focus_force();native.pump(root)
    frozen=s.sidecar_path.read_bytes();entry=ed.entry
    click_control(h,ed);native.pump(root)
    assert ed.entry is entry and h.window.focus_get() is entry and ed.selected=='GT:a0'
    assert s.sidecar_path.read_bytes()==frozen

def test_N13_double_click_on_navigation_is_not_swallowed(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    click_control(h,ed);ready(root,h,ed);assert ed.selected=='GT:a1'
    box,oid,back=next(x for x in ed.mt_stack_controls if not x[2])
    ed.double_press(SimpleNamespace(x=(box[0]+box[2])/2,y=(box[1]+box[3])/2,widget=ed.c,state=0))
    ready(root,h,ed);assert ed.selected=='GT:a2'
