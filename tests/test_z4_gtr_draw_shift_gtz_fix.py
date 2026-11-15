"""Displayed GTR is reserved; draw/Shift and explicit GTZ/MZ stay separate."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_mt_geometry_layers import source_session
from test_z4_badge4_hud2 import bbox as bbox_gui
from test_z4_space_gtr_hud_shift import fixture_archive
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data,mobile_canvas_partial_review as partial,mobile_manual_frame_guard as guard
from auto_annotation_tool.ranking.mobile_mt_geometry_layers import create_reference
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers
G='mt-one';OID='GT:a0';GT=[[190,155],[260,155],[260,195],[190,195]]
ENCLOSING=[[120,120],[330,120],[330,240],[120,240]]
def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())

@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    case=getattr(request,'param','provisional')
    if case=='bbox':
        def invalid(rows,trace,payload):
            rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
        s=fixture_archive(tmp_path/'data',n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
        partial.toggle_plate(s,G,OID);data.edit_gt(s,G,OID,text='KEEPGTZ')
    else:s=source_session(tmp_path/'data',n=3 if case=='many' else 1)
    if case in ('provisional','many'):partial.toggle_plate(s,G,OID)
    elif case=='corrected':create_reference(s,G,OID,GT)
    if case=='many':
        data.toggle_unassociated(s,G,'GT:a1',partial_review=True)
        core.add_plate(s,G,[[650,350],[800,350],[800,420],[650,420]],partial_review=True)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:ready(root,h,ed);yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

@pytest.mark.parametrize('quad',[GT,ENCLOSING,[[130,170],[330,170],[330,180],[130,180]]],ids=['inside','superset','crossing'])
def test_presence_accepted_visible_GTR_blocks_second_GT_atomically(tmp_path,quad):
    s=source_session(tmp_path,n=1);partial.toggle_plate(s,G,OID);before=frozen(s)
    refs=list(guard.reference_records(core.scene(s,G)));assert len(refs)==1
    with pytest.raises(guard.GTRCollisionError):core.add_plate(s,G,quad,partial_review=True)
    assert len(data.snapshot(s,G)['plates'])==1 and frozen(s)==before

@pytest.mark.parametrize('gui',['provisional','corrected'],indirect=True)
@pytest.mark.parametrize('initial',['BBOX','POSE','GTR'])
def test_D_on_existing_GTR_opens_same_edit_without_hiding_or_replacing(gui,initial,monkeypatch):
    root,h,ed,s=gui;nav.select_layer(ed,initial);ready(root,h,ed);before=frozen(s)
    phases=[];paint=ed.collision_warning.paint
    def observed():paint();phases.append(ed.collision_warning.phase)
    monkeypatch.setattr(ed.collision_warning,'paint',observed)
    old=copy.deepcopy(data.snapshot(s,G)['plates']);native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='edit' and ed.draw_target is None and not ed.points and ed.gt_focus_id
    assert 'red' in phases and len(ed.c.find_withtag('corner_handle'))==4
    assert data.snapshot(s,G)['plates']==old and frozen(s)==before

@pytest.mark.parametrize('gui',['provisional','corrected'],indirect=True)
def test_new_enclosing_quad_with_no_vertex_in_reserved_area_is_blocked(gui):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);before=frozen(s);native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='draw'
    for point in ENCLOSING:
        assert guard.find_point_collision(ed._read(),point,session=s,gid=G) is None
        native.click(h,ed,point)
    ready(root,h,ed)
    assert ed.mode=='edit' and ed.gt_focus_id and not ed.points,(ed.points,h.status_var.get())
    assert len(data.snapshot(s,G)['plates'])==1 and ed._stack_counter['count']==3 and frozen(s)==before

@pytest.mark.parametrize('gui',['many'],indirect=True)
def test_corner_edit_cannot_enclose_other_existing_GTR(gui):
    root,h,ed,s=gui;before=frozen(s)
    with pytest.raises(guard.GTRCollisionError):data.edit_gt(s,G,OID,quad=[[100,100],[900,100],[900,500],[100,500]])
    assert frozen(s)==before

@pytest.mark.parametrize('gui',['many'],indirect=True)
@pytest.mark.parametrize('focus',[False,True])
def test_Shift_shows_entire_image_without_selection_or_with_GT_focus(gui,focus):
    root,h,ed,s=gui
    if focus:nav.select_layer(ed,'GTR');ed.gt_focus_id=OID
    else:ed.escape()
    ed.ref_visible.set(False);ed.inf_visible.set(False);ed.show_fp_var.set(False);ready(root,h,ed);before=frozen(s)
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert ed.shift_preview.active and len(ed.c.find_withtag('shift_mt_source_bbox'))==3
    assert len(ed.c.find_withtag('shift_mt_pose'))==3 and len(ed.c.find_withtag('shift_saved_gtr'))==2
    assert any(ed.c.itemcget(i,'outline')=='#EF5262' for i in ed.c.find_withtag('shift_mt_pose'))
    assert not ed.c.find_withtag('navigator_geometry') and not ed.c.find_withtag('corner_handle')
    ed.shift_preview.release(NS(keysym='Shift_L'));ed.overlay()
    assert not ed.c.find_withtag('shift_source_preview') and not ed.ref_visible.get() and not ed.inf_visible.get() and frozen(s)==before

@pytest.mark.parametrize('gui',['many'],indirect=True)
def test_D_hides_all_other_frames_even_while_Shift_is_held(gui):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);before=frozen(s);native.key(h,'d');ready(root,h,ed)
    native.click(h,ed,(540,520));ready(root,h,ed);points=copy.deepcopy(ed.points)
    for tag in ('navigator_geometry','operator_quad','mt_original','model_location_bbox','mt_source_bbox_layer','drawing_mt_bbox','fp_diagonal','shift_source_preview'):
        assert not ed.c.find_withtag(tag),tag
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert not ed.shift_preview.active and not ed.c.find_withtag('shift_source_preview') and ed.points==points
    ed.shift_preview.release(NS(keysym='Shift_L'));ed.overlay()
    assert not ed.c.find_withtag('shift_source_preview') and ed.points==points and frozen(s)==before

@pytest.mark.parametrize('gui',['bbox'],indirect=True)
def test_bbox_completion_draw_has_no_source_bbox_guide(gui):
    root,h,ed,s=gui;before=frozen(s)
    native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='draw' and not ed.c.find_withtag('drawing_mt_bbox') and not ed.c.find_withtag('model_location_bbox')
    assert frozen(s)==before

def test_GTR_badge_has_distinct_editable_GTZ_field_and_source_MZ(gui):
    root,h,ed,s=gui;data.edit_gt(s,G,OID,text='MYGTZ123');ready(root,h,ed);native.focus_canvas(h);nav.select_layer(ed,'GTR');ready(root,h,ed);before=frozen(s)
    value=ed._context_badge;candidate=clean.selected(ed)['candidate']
    assert value['detail']=='GTZ: MYGTZ123' and value['mz_detail']=='MZ: '+candidate['mz_prediction']
    assert len(ed.c.find_withtag('gtz_edit_field'))==1 and len(ed.c.find_withtag('gtz_edit_action'))==2
    assert ed.c.itemcget(ed.c.find_withtag('context_badge_mz')[0],'text')==value['mz_detail']
    assert candidate['mz_prediction']!='MYGTZ123' and frozen(s)==before
    button=next(i for i in ed.c.find_withtag('gtz_edit_action') if ed.c.type(i)=='rectangle')
    l,t,r,b=ed.c.coords(button);ed.c.event_generate('<ButtonPress-1>',x=round((l+r)/2),y=round((t+b)/2));ed.c.event_generate('<ButtonRelease-1>',x=round((l+r)/2),y=round((t+b)/2));ready(root,h,ed)
    assert ed.entry is not None and ed.entry_var.get()=='MYGTZ123' and frozen(s)==before
    assert ed.entry.winfo_x()+ed.entry.winfo_width()<=ed._gtz_entry_right+2
    assert ed._gtz_edit_font.measure(ed.entry_var.get())+4<=ed.entry.winfo_width()
    ed.escape();ready(root,h,ed);assert frozen(s)==before

def test_GTR_MZ_NOT_RUN_stays_separate_from_GTZ(bbox_gui):
    root,h,ed,s=bbox_gui;nav.select_layer(ed,'GTR');ready(root,h,ed)
    assert ed._context_badge['mz_detail']=='MZ: nie uruchomiono (NOT_RUN)'
    assert not data.snapshot(s,G)['plates'][clean.selected(ed)['pid']]['gt_text']

def test_Shift_optional_frame_identity_failure_preserves_current_image(gui,monkeypatch):
    root,h,ed,s=gui;before=frozen(s)
    from auto_annotation_tool.ranking import mobile_canvas_source_hud as hud
    def unavailable(*a):raise ValueError('No cross-call source identity')
    with monkeypatch.context() as patch:
        patch.setattr(hud,'_context',unavailable)
        ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
        assert len(ed.c.find_withtag('shift_source_preview'))==3
        ed.shift_preview.release(NS(keysym='Shift_L'));ed.overlay()
    assert frozen(s)==before

def test_Shift_INVALID_QUAD_shows_bbox_and_GT_without_synthetic_Pose(bbox_gui):
    root,h,ed,s=bbox_gui;ed.escape();ready(root,h,ed);before=frozen(s)
    ed.ref_visible.set(False);ed.inf_visible.set(False)
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert len(ed.c.find_withtag('shift_mt_source_bbox'))==1 and len(ed.c.find_withtag('shift_saved_gtr'))==1
    assert not ed.c.find_withtag('shift_mt_pose') and frozen(s)==before
    ed.shift_preview.release(NS(keysym='Shift_L'));ed.overlay()

def test_D_missing_active_reference_never_starts_a_replacement(gui,monkeypatch):
    root,h,ed,s=gui;before=frozen(s)
    with monkeypatch.context() as patch:
        patch.setattr(guard,'reservation_records',lambda *a,**k:[])
        native.key(h,'d');ready(root,h,ed)
        assert ed.mode!='draw' and ed.draw_target is None and not ed.points and frozen(s)==before
