"""A fits physical objects; approval colors follow stored decisions, read-only."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_gtr_reapproval import verified,edit,pid,MANUAL,G,OID
from test_z4_mt_geometry_layers import source_session
from test_z4_space_gtr_hud_shift import fixture_archive
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.gui import mobile_canvas_plate_cycle as cycle,mobile_canvas_approval_visuals as colors,mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers,mobile_canvas_navigator as nav

def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict()),core.summary(s)

@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    case=getattr(request,'param','many')
    if case=='bbox':
        def invalid(rows,trace,payload):
            rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
        s=fixture_archive(tmp_path/'data',n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    elif case in ('empty','source'):
        s=source_session(tmp_path/'data',n=0 if case=='empty' else 3,mz=False)
    elif case=='shared':
        s=source_session(tmp_path/'data',n=3,mz=False);partial.toggle_plate(s,G,OID);stacks.associate(s,G,'GT:a1',OID)
    else:s=verified(tmp_path/'data',n=3 if case=='many' else 1)
    if case in ('many','shared'):
        oid=core.add_plate(s,G,MANUAL,partial_review=True);partial.confirm_gtr(s,G,pid(s,oid))
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:ready(root,h,ed);yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def press(root,h,ed,key='a'):
    native.key(h,key);ready(root,h,ed)

def check_fit(ed,target):
    points=[ed.to_canvas(p) for p in target['quad']];w,h=ed.c.winfo_width(),ed.c.winfo_height()
    assert 0<min(p[0] for p in points)<max(p[0] for p in points)<w
    assert 0<min(p[1] for p in points)<max(p[1] for p in points)<h

def color_for(ed,plate_id):
    matches=[i for i in ed.c.find_withtag('operator_quad') if plate_id in ed.c.gettags(i)]
    assert len(matches)==1;return ed.c.itemcget(matches[0],'outline')

def border(ed):
    items=ed.c.find_withtag('image_verification_border');assert len(items)==1
    return items[0],ed.c.itemcget(items[0],'outline')

@pytest.mark.parametrize('size',['1649x896','1180x760'])
def test_A_visits_each_GT_including_manual_once_wraps_and_never_restores_full_view(gui,size):
    root,h,ed,s=gui;h.window.state('normal');h.window.geometry(size+'+10+10');ready(root,h,ed)
    choices=cycle.targets(ed);assert len(choices)==4;before=frozen(s);initial=clean.selected(ed)['pid'];visited=[]
    for _ in range(5):
        target=cycle.next_target(ed,cycle.targets(ed));press(root,h,ed)
        row=clean.selected(ed);visited.append(row['pid']);assert row['pid']==target['pid']
        check_fit(ed,target);assert ed.plate_fit_active and h._view_zoom>1
        assert not layers.model(ed) or layers.model(ed)['active']=='GTR'
    assert len(set(visited[:4]))==4 and visited[3]==initial and visited[4]==visited[0]
    press(root,h,ed,'f');assert h._view_zoom==1 and not ed.plate_fit_active and frozen(s)==before

@pytest.mark.parametrize('gui',['shared','source','bbox'],indirect=True)
def test_A_groups_shared_GT_and_uses_Pose_or_bbox_when_GT_absent(gui):
    root,h,ed,s=gui;choices=cycle.targets(ed);before=frozen(s)
    if len(choices)>1:assert len(choices)==3
    else:assert choices[0]['kind']=='BBOX'
    seen=[]
    for _ in choices:
        target=cycle.next_target(ed,cycle.targets(ed));press(root,h,ed);seen.append(target['key']);check_fit(ed,target)
        value=layers.model(ed)
        assert not value or value['active']==target['kind']
    assert len(set(seen))==len(choices) and frozen(s)==before

def test_A_from_image_list_does_not_need_pointer_and_key_repeat_does_not_skip_objects(gui):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);choices=cycle.targets(ed);before=frozen(s)
    h.subject_tree.focus_force();root.update();ed.pointer_canvas=(-100,-100)
    h.subject_tree.event_generate('<KeyPress-a>');root.update();ready(root,h,ed)
    first=clean.selected(ed)['pid'];assert first==choices[0]['pid']
    h.subject_tree.event_generate('<KeyPress-a>');root.update();assert clean.selected(ed)['pid']==first
    h.subject_tree.event_generate('<KeyRelease-a>');root.update();ready(root,h,ed)
    press(root,h,ed);assert clean.selected(ed)['pid']==choices[1]['pid'] and frozen(s)==before

@pytest.mark.parametrize('gui',['one'],indirect=True)
def test_A_single_object_stays_fitted_and_typing_or_Ctrl_A_do_not_navigate(gui):
    root,h,ed,s=gui;before=frozen(s);press(root,h,ed);zoom=h._view_zoom;press(root,h,ed)
    assert h._view_zoom==zoom and ed.plate_fit_active
    before_view=(ed.selected,h._view_zoom,h._view_center)
    ed.fit_plate_toggle(NS(widget=ed.c,state=4));assert (ed.selected,h._view_zoom,h._view_center)==before_view
    ed.start_entry(ed.selected);ready(root,h,ed)
    ed.entry.event_generate('<KeyPress-a>');ed.entry.event_generate('<KeyRelease-a>');root.update()
    assert (ed.selected,h._view_zoom,h._view_center)==before_view;ed.escape();ready(root,h,ed);assert frozen(s)==before

def test_A_refuses_draft_and_live_drag_without_saving_them(gui):
    root,h,ed,s=gui;ed.escape();ed.draw_toggle();ready(root,h,ed);native.click(h,ed,(900,550));ready(root,h,ed)
    before=frozen(s);points=copy.deepcopy(ed.points);press(root,h,ed);assert ed.points==points and ed.mode=='draw' and frozen(s)==before
    assert not ed.c.find_withtag('image_verification_border')
    ed.escape();nav.select_layer(ed,'GTR',OID);ready(root,h,ed)
    x,y=map(round,ed.to_canvas(core.scene(s,G)['objects'][0]['quad'][0]));ed.c.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag
    press(root,h,ed);assert ed.drag and ed.selected==OID and frozen(s)==before

@pytest.mark.parametrize('gui',['empty'],indirect=True)
def test_A_empty_image_keeps_view_and_review(gui):
    root,h,ed,s=gui;before=frozen(s);view=h._view_zoom,h._view_center
    press(root,h,ed);assert not cycle.targets(ed) and (h._view_zoom,h._view_center)==view and frozen(s)==before

def test_per_GT_approval_and_image_border_follow_edit_confirm_undo_and_redo(gui):
    root,h,ed,s=gui;before=frozen(s);a,b=pid(s),pid(s,'GT:a1')
    assert color_for(ed,a)==color_for(ed,b)==border(ed)[1]==colors.GREEN
    edit(s);ready(root,h,ed)
    assert color_for(ed,a)==colors.PENDING and color_for(ed,b)==colors.GREEN and border(ed)[1]==colors.RED
    nav.select_layer(ed,'GTR',OID);ready(root,h,ed);assert ed.c.itemcget(ed.c.find_withtag('context_badge_background')[0],'outline')==colors.PENDING
    press(root,h,ed,'space');assert color_for(ed,a)==border(ed)[1]==colors.GREEN
    stacks.restore_history(s,G);ready(root,h,ed);assert color_for(ed,a)==colors.PENDING and border(ed)[1]==colors.RED
    stacks.restore_history(s,G,redo=True);ready(root,h,ed);assert color_for(ed,a)==border(ed)[1]==colors.GREEN
    assert before[1]['invocation_annotations'][G][core.KEY]['parents']==core.scene(s,G)['parents']

def test_border_uses_image_bounds_then_clips_at_zoom_and_ignores_REF_INF(gui):
    root,h,ed,s=gui;before=frozen(s);ed.fit();ready(root,h,ed);i,c=border(ed)
    x,y=ed.to_canvas((0,0));r,b=ed.to_canvas(ed._read()['image_size']);coords=ed.c.coords(i)
    assert coords==[max(2,x+1.5),max(2,y+1.5),min(ed.c.winfo_width()-2,r-1.5),min(ed.c.winfo_height()-2,b-1.5)]
    ed.ref_visible.set(False);ed.inf_visible.set(False);ed.h.zoom_image(4);ready(root,h,ed);i,c=border(ed)
    assert c==colors.GREEN and all(0<=v for v in ed.c.coords(i)) and frozen(s)==before

def test_Shift_uses_same_GTR_colors_and_view_paints_never_write(gui):
    root,h,ed,s=gui;edit(s);ready(root,h,ed);before=frozen(s);a=pid(s)
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    saved=ed.c.find_withtag('shift_saved_gtr');assert len(saved)==4
    assert next(ed.c.itemcget(i,'outline') for i in saved if a in ed.c.gettags(i))==colors.PENDING
    assert sum(ed.c.itemcget(i,'outline')==colors.GREEN for i in saved)==3
    ed.shift_preview.release(NS(keysym='Shift_L'))
    for _ in range(3):ed.overlay();h._paint_image();ready(root,h,ed)
    assert frozen(s)==before
