"""Ordinary clicks select one stack, with the same layer state as X/Z."""
import copy
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_mt_geometry_layers import source_session
from test_z4_space_gtr_hud_shift import fixture_archive
from test_z4_one_navigator import multiple
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_clean_view as clean
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_mt_geometry_layers as evidence
G='mt-one';OID='GT:a0';GT=[[190,155],[260,155],[260,195],[190,195]]
POINTS={'GTR':(220,175),'BBOX':(160,150),'POSE':(280,185)}

def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())

@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    case=getattr(request,'param','linked')
    if case=='bbox':
        def invalid(rows,trace,payload):
            rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
        s=fixture_archive(tmp_path/'data',n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    else:s=source_session(tmp_path/'data',n=1)
    if case=='linked':evidence.create_reference(s,G,OID,GT)
    if case=='manual':core.add_plate(s,G,[[650,350],[800,350],[800,420],[650,420]],partial_review=True)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:ready(root,h,ed);yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

@pytest.mark.parametrize('area',['GTR','POSE','BBOX'])
@pytest.mark.parametrize('initial',['BBOX','POSE','GTR',None])
def test_click_any_associated_area_selects_GTR_without_edit_focus(gui,area,initial):
    root,h,ed,s=gui
    if initial:nav.select_layer(ed,initial)
    else:ed.escape()
    ready(root,h,ed);before=frozen(s)
    native.click(h,ed,POINTS[area]);ready(root,h,ed)
    assert ed.selected==OID and layers.model(ed)['active']=='GTR' and ed.navigator.selection.get()=='GTR'
    assert not ed.gt_focus_id and ed.drag is None and not ed.points and not ed.collision_warning.active
    assert ed._context_badge['kind']=='GTR' and ed._stack_counter['count']==3 and frozen(s)==before

@pytest.mark.parametrize('gui',['source'],indirect=True)
@pytest.mark.parametrize('area',['POSE','BBOX'])
@pytest.mark.parametrize('initial',['BBOX','POSE',None])
def test_no_GTR_selects_Pose_including_unpainted_bbox_area(gui,area,initial):
    root,h,ed,s=gui
    if initial:nav.select_layer(ed,initial)
    else:ed.escape()
    ready(root,h,ed);before=frozen(s)
    native.click(h,ed,POINTS[area]);ready(root,h,ed)
    assert layers.model(ed)['active']=='POSE' and not ed.gt_focus_id and frozen(s)==before

@pytest.mark.parametrize('gui',['bbox'],indirect=True)
def test_no_GTR_or_Pose_selects_BBOX(gui):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);before=frozen(s)
    native.click(h,ed,(220,175));ready(root,h,ed)
    assert layers.model(ed)['active']=='BBOX' and len(layers.model(ed)['layers'])==1 and frozen(s)==before

def test_hidden_GTR_can_be_selected_without_turning_REF_on(gui):
    root,h,ed,s=gui;ed.ref_visible.set(False);ed.escape();ready(root,h,ed);before=frozen(s)
    native.click(h,ed,POINTS['BBOX']);ready(root,h,ed)
    assert layers.model(ed)['active']=='GTR' and not ed.ref_visible.get() and not ed.gt_focus_id
    assert not ed.c.find_withtag('operator_quad') and not ed.c.find_withtag('corner_handle') and frozen(s)==before

def test_hidden_all_contours_do_not_create_an_invisible_hit(gui):
    root,h,ed,s=gui;ed.ref_visible.set(False);ed.inf_visible.set(False);ed.escape();ready(root,h,ed)
    native.click(h,ed,POINTS['GTR']);ready(root,h,ed)
    assert not ed.selected

def test_source_image_mismatch_does_not_select_a_stale_stack(gui):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);before=frozen(s)
    entry=h._image_entry;h._image_entry='different-frame'
    try:
        native.click(h,ed,POINTS['GTR']);assert not ed.selected and frozen(s)==before
    finally:h._image_entry=entry

def test_click_does_not_change_X_Z_cycle_or_PPM_source_selection(gui):
    root,h,ed,s=gui;before=frozen(s);nav.select_layer(ed,'POSE');ready(root,h,ed)
    native.click(h,ed,POINTS['BBOX']);ready(root,h,ed)
    assert layers.model(ed)['active']=='GTR'
    for key,expected in (('x','BBOX'),('x','POSE'),('z','BBOX'),('z','GTR')):
        native.key(h,key);ready(root,h,ed);assert layers.model(ed)['active']==expected
    nav.select_layer(ed,'POSE');ready(root,h,ed)
    assert nav.right_hit(ed,*ed.to_canvas(POINTS['POSE']))==OID
    assert frozen(s)==before

def test_actual_drag_pans_without_selecting_a_different_layer(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'POSE');h.zoom_image(4);ready(root,h,ed);before=frozen(s)
    x,y=map(round,ed.to_canvas(POINTS['GTR']));center=h._view_center
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y)
    ed.c.event_generate('<B1-Motion>',x=x+35,y=y+15)
    ed.c.event_generate('<ButtonRelease-1>',x=x+35,y=y+15);ready(root,h,ed)
    assert layers.model(ed)['active']=='POSE' and h._view_center!=center and frozen(s)==before

@pytest.mark.parametrize('gui',['manual'],indirect=True)
def test_unlinked_GT_does_not_become_a_source_association(gui):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);before=frozen(s)
    native.click(h,ed,(720,385));ready(root,h,ed)
    assert clean.selected(ed)['role']=='GT' and layers.model(ed) is None and not ed.gt_focus_id
    native.click(h,ed,POINTS['POSE']);ready(root,h,ed)
    assert ed.selected==OID and layers.model(ed)['active']=='POSE' and frozen(s)==before

def test_click_other_FINAL_keeps_that_stack_and_its_real_GTR(multiple):
    root,h,ed,s=multiple;ready(root,h,ed);before=frozen(s)
    native.click(h,ed,(500,180));ready(root,h,ed)
    assert ed.selected=='GT:a1' and layers.model(ed)['active']=='GTR'
    assert not ed.gt_focus_id and frozen(s)==before
    native.click(h,ed,(700,180));ready(root,h,ed)
    assert ed.selected=='GT:a2' and layers.model(ed)['active']=='POSE' and frozen(s)==before
