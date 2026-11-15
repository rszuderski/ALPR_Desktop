"""The active layer's one-based index shares a safe position with the total."""
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_one_navigator import gui
from test_z4_badge4_hud2 import bbox
from test_z4_a_cycle_approval_visuals import gui as objects
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_clean_view as clean

def values(ed):
    current=ed.c.find_withtag('stack_current_text');total=ed.c.find_withtag('stack_count_text')
    assert len(current)==len(total)==1
    return ed.c.itemcget(current[0],'text'),ed.c.itemcget(total[0],'text')

@pytest.mark.parametrize('size',['1649x896','1180x760'])
def test_layer_menu_and_ZX_update_current_while_total_stays_three(gui,size):
    root,h,ed,s=gui;h.window.state('normal');h.window.geometry(size+'+10+10');ready(root,h,ed);before=s.sidecar_path.read_bytes()
    for kind,index in (('BBOX','1'),('POSE','2'),('GTR','3')):
        assert nav.select_layer(ed,kind);ready(root,h,ed);assert values(ed)==(index,'3')
    for key,expected in (('x','1'),('x','2'),('x','3'),('z','2')):
        native.key(h,key);ready(root,h,ed);assert values(ed)==(expected,'3')
    assert s.sidecar_path.read_bytes()==before

def test_pair_is_adjacent_and_does_not_cover_badge_or_geometry(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);counter=ed._stack_counter
    a=counter['current_rect'];b=counter['rect'];group=counter['group_rect']
    assert a[2]<b[0] and a[1]==b[1] and a[3]==b[3] and counter['current']==counter['count']==3
    assert 0<=group[0]<group[2]<=ed.c.winfo_width() and 0<=group[1]<group[3]<=ed.c.winfo_height()
    assert not any(clean.intersects(group,o) for o in ed.clean_geometry_boxes+[ed._context_badge['rect']])
    assert not ed.hit((a[0]+a[2])/2,(a[1]+a[3])/2)

def test_Escape_preserves_current_index_and_draw_cancellation_restores_pair(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'POSE');ready(root,h,ed);before=s.sidecar_path.read_bytes();native.key(h,'Escape');ready(root,h,ed)
    assert not ed.selected and ed._context_badge['inspection'] and values(ed)==('2','3')
    native.key(h,'d');ready(root,h,ed);assert not ed.c.find_withtag('stack_position_badges')
    native.key(h,'Escape');ready(root,h,ed);assert values(ed)==('2','3') and s.sidecar_path.read_bytes()==before

def test_missing_Pose_skips_that_layer_in_current_and_total(bbox):
    root,h,ed,s=bbox;nav.select_layer(ed,'BBOX');ready(root,h,ed);assert values(ed)==('1','2')
    native.key(h,'x');ready(root,h,ed);assert values(ed)==('2','2')
    native.key(h,'x');ready(root,h,ed);assert values(ed)==('1','2')

@pytest.mark.parametrize('objects',['source','many'],indirect=True)
def test_unaccepted_source_and_manual_GT_use_their_actual_stack(objects):
    root,h,ed,s=objects
    if len(ed._read()['objects'])==3:
        nav.select_layer(ed,'BBOX');ready(root,h,ed);assert values(ed)==('1','2')
        nav.select_layer(ed,'POSE');ready(root,h,ed);assert values(ed)==('2','2')
    else:
        manual=next(o['object_id'] for o in ed._read()['objects'] if not o['parent_id'])
        ed._choose_plate(manual);ready(root,h,ed);assert values(ed)==('1','1')

def test_REF_INF_and_focus_cleanup_remove_both_counters(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'POSE');ready(root,h,ed);ed.inf_visible.set(False);ed.overlay()
    assert not ed._stack_counter and not ed.c.find_withtag('stack_position_badges')
    ed.inf_visible.set(True);ed.overlay();assert values(ed)==('2','3')
    from auto_annotation_tool.gui.mobile_canvas_info_badge import clear
    clear(ed);assert not ed.c.find_withtag('stack_current_badge') and not ed.c.find_withtag('stack_count_badge')
