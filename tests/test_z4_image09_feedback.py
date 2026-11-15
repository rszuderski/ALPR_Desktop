"""Image09: explicit review work, reversible approval and traceable FP/FN."""
from pathlib import Path
import copy,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial,mobile_mt_detection_evaluation as scoring
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers,mobile_canvas_navigator as nav
OUT=Path(__file__).resolve().parents[1]/'output/Z4_IMAGE09_DECISIONS_20261030_001'
F='exp-1792699303016-c8b05efb';G=F+'-a00000009'

@pytest.fixture
def gui(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    side=tmp_path/'review.json';shutil.copyfile(OUT/'private'/(F+'.review.json'),side)
    s=MobileReviewSession.open(OUT/'private/day.alprsession',sidecar_path=side)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    iid=next(i for i in h.subject_tree.get_children('') if G in subject_groups(s,h._subject_ids[i]))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry'] and not h._saving);ready(root,h,ed)
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

@pytest.mark.parametrize('focus',['canvas','navigator','image_list'])
def test_ctrl_z_after_Space_works_from_review_focus(gui,focus):
    root,h,ed,s=gui;ed.undo();ready(root,h,ed)
    assert partial.progress(core.scene(s,G))['decisions']==1
    native.key(h,'space');ready(root,h,ed);assert partial.progress(core.scene(s,G))['complete']
    widget={'canvas':h.canvas,'navigator':ed.navigator.layer,'image_list':h.subject_tree}[focus]
    widget.focus_force();native.pump(root,lambda:root.focus_get() is widget)
    widget.event_generate('<Control-KeyPress-z>');widget.event_generate('<Control-KeyRelease-z>');ready(root,h,ed)
    assert partial.progress(core.scene(s,G))['decisions']==1

def test_image09_source_explains_FP_and_FN_without_extra_plate(gui):
    from auto_annotation_tool.gui import mobile_canvas_review_feedback as feedback
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();r=feedback.image_feedback(ed)
    assert r['status']=='MEASURED' and len(r['fp'])==len(r['fn'])==1
    assert r['fp'][0]['attempt_id']==G and r['fn'][0]['gt_id']==clean.selected(ed)['pid']
    assert '43,21%' in r['text'] and '50%' in r['text'] and 'tej samej tablicy' in r['text']
    assert s.sidecar_path.read_bytes()==before

def test_visible_undo_redo_is_one_review_action_and_keeps_GT(gui):
    root,h,ed,s=gui;before=stacks.snapshot(s,G);rev=s.review.review_revision
    assert ed.clean_undo_button.winfo_ismapped() and 'Cofnij zatwierdzenie' in ed.clean_undo_button['text']
    ed.clean_undo_button.invoke();ready(root,h,ed)
    assert s.review.review_revision==rev+1 and partial.progress(core.scene(s,G))['decisions']==1
    assert 'Wskazanie MT 1' in clean.context(ed,{})['image_note'] and 'obrys GTR 1' in clean.context(ed,{})['image_note']
    assert ed.clean_redo_button.winfo_ismapped();ed.clean_redo_button.invoke();ready(root,h,ed)
    after=stacks.snapshot(s,G)
    for pid in before['plates']:
        for key in ('gt_text','gt_quad'):assert after['plates'][pid][key]==before['plates'][pid][key]
    assert partial.progress(core.scene(s,G))['complete']

@pytest.mark.parametrize('kind',['fp','fn'])
def test_counter_navigates_to_real_source_or_reference_without_write(gui,kind):
    from auto_annotation_tool.gui import mobile_canvas_review_feedback as feedback
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();ed.inf_visible.set(False);ed.ref_visible.set(False)
    ed.contexts.bbox_counters[kind].invoke();ready(root,h,ed)
    assert layers.model(ed)['active']==('BBOX' if kind=='fp' else 'GTR')
    assert ed.inf_visible.get() and ed.ref_visible.get() and s.sidecar_path.read_bytes()==before
    assert ed.c.find_withtag('navigator_geometry')

def test_bbox_explanation_is_not_a_fabricated_match(gui):
    from auto_annotation_tool.gui.mobile_canvas_info_badge import measurements,context_model
    root,h,ed,s=gui;nav.select_layer(ed,'BBOX');ready(root,h,ed)
    before=scoring.evaluate_unit(s,G);value=measurements(ed,clean.selected(ed))
    assert value['bbox_iou'] is None and value['reference_comparison']['iou']==pytest.approx(.4320993296871864)
    assert 'BBOX IoU z GTR 1: 43,2%' in context_model(ed)['detail'] and '43,21%' in clean.context(ed,{})['object']
    assert scoring.evaluate_unit(s,G)==before and before['matched_pairs']==[]

def test_ctrl_y_redo_from_navigator_and_typing_are_isolated(gui):
    import tkinter as tk
    root,h,ed,s=gui;ed.undo();ready(root,h,ed)
    ed.navigator.layer.focus_force();native.pump(root,lambda:root.focus_get() is ed.navigator.layer)
    ed.navigator.layer.event_generate('<Control-KeyPress-y>');ready(root,h,ed);assert partial.progress(core.scene(s,G))['complete']
    before=s.sidecar_path.read_bytes();host=tk.Frame(h.window);host.place(x=5,y=5);entry=tk.Entry(host);entry.pack();entry.focus_force();native.pump(root,lambda:root.focus_get() is entry)
    entry.event_generate('<Control-KeyPress-z>');native.pump(root);assert s.sidecar_path.read_bytes()==before
    other=tk.Toplevel(root);box=tk.Entry(other);box.pack();box.focus_force();native.pump(root,lambda:root.focus_get() is box)
    box.event_generate('<Control-KeyPress-z>');native.pump(root);assert s.sidecar_path.read_bytes()==before
    other.destroy();host.destroy()

def test_two_physical_GTs_stay_separate_when_cycling_FN(gui):
    from auto_annotation_tool.gui import mobile_canvas_review_feedback as feedback
    root,h,ed,s=gui;oid=core.add_plate(s,G,[[40,50],[140,50],[140,90],[40,90]],partial_review=True)
    state=stacks.snapshot(s,G);pid=next(pid for pid,p in state['plates'].items() if p['origin_object_id']==oid)
    partial.confirm_stack(s,G,plate_id=pid);ready(root,h,ed);view=feedback.image_feedback(ed)
    assert len(view['fn'])==2 and 'tej samej tablicy' not in view['text']
    before=s.sidecar_path.read_bytes();seen=[]
    for _ in range(2):feedback.inspect(ed,'fn');ready(root,h,ed);seen.append(clean.selected(ed)['pid'])
    assert len(set(seen))==2 and s.sidecar_path.read_bytes()==before

def test_space_toggles_verification_and_undo_does_not_touch_other_images(gui):
    root,h,ed,s=gui;before=copy.deepcopy(stacks.snapshot(s,G)['plates']);other=copy.deepcopy(s.review.invocation_annotations)
    native.key(h,'space');ready(root,h,ed);assert not partial.progress(core.scene(s,G))['complete']
    native.key(h,'space');ready(root,h,ed);assert partial.progress(core.scene(s,G))['complete']
    after=stacks.snapshot(s,G)['plates'];assert all(after[pid][k]==p[k] for pid,p in before.items() for k in ('gt_quad','gt_text'))
    ed.undo();ready(root,h,ed);assert not partial.progress(core.scene(s,G))['complete']
    assert {k:v for k,v in s.review.invocation_annotations.items() if k!=G}=={k:v for k,v in other.items() if k!=G}

def test_counter_cannot_project_coordinates_onto_a_different_image(gui):
    from auto_annotation_tool.gui import mobile_canvas_review_feedback as feedback
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();view=(h._view_zoom,h._view_center,ed.selected)
    original=h._image_entry
    try:
        h._image_entry='samples/crops/different-image.jpg'
        assert feedback.inspect(ed,'fp') is False and feedback.inspect(ed,'fn') is False
        assert (h._view_zoom,h._view_center,ed.selected)==view and s.sidecar_path.read_bytes()==before
    finally:h._image_entry=original
