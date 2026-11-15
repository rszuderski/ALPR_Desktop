"""Escape restores inspection badges; vivid contours keep a thin dark edge."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_gtr_reapproval_gui import gui
from test_z4_one_navigator import multiple
from test_z4_frame_stacks import ready
from test_z4_gtr_reapproval import G,OID,pid,MANUAL
from auto_annotation_tool.gui import mobile_canvas_info_badge as badge,mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_approval_visuals as colors
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_canvas_partial_review as partial

def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict()),core.summary(s)

def escaped(root,h,ed):
    native.key(h,'Escape');ready(root,h,ed)
    assert ed.mode=='select' and not ed.selected and not ed.gt_focus_id and ed.entry is None and not ed.drag
    assert not ed.c.find_withtag('corner_handle')
    assert len(ed.c.find_withtag('context_badge_background'))==1
    assert len(ed.c.find_withtag('stack_count_badge'))==2
    assert ed._context_badge and ed._context_badge['inspection']
    assert ed._stack_counter['count']==3

@pytest.mark.parametrize('size',['1649x896','1180x760'])
def test_saved_corner_edit_then_one_Escape_restores_GTR_GTZ_MZ_and_count(gui,size):
    root,h,ed,s=gui;h.window.state('normal');h.window.geometry(size+'+10+10');ready(root,h,ed)
    native.focus_canvas(h);nav.select_layer(ed,'GTR');ready(root,h,ed)
    q=core.scene(s,G)['objects'][0]['quad'];x,y=map(round,ed.to_canvas(q[0]))
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag
    ed.c.event_generate('<B1-Motion>',x=x+7,y=y+2);ed.c.event_generate('<ButtonRelease-1>',x=x+7,y=y+2);ready(root,h,ed)
    before=frozen(s);stored=copy.deepcopy(core.scene(s,G)['objects'][0]['quad']);escaped(root,h,ed)
    assert ed._context_badge['quad']==stored and ed._context_badge['plate_id']==pid(s)
    assert ed.c.find_withtag('context_badge_detail') and ed.c.find_withtag('context_badge_mz')
    assert partial.progress(core.scene(s,G))['pending_gtr_ids']==[pid(s)] and not ed.checked_var.get()
    assert frozen(s)==before

def test_Escape_cancels_GTZ_and_keeps_badge_with_saved_text(gui):
    root,h,ed,s=gui;native.focus_canvas(h);nav.select_layer(ed,'GTR');ed.start_entry(OID);ready(root,h,ed)
    before=frozen(s);ed.entry_var.set('UNSAVEDGTZ');widget=ed.entry
    widget.event_generate('<KeyPress-Escape>');ready(root,h,ed)
    assert ed.entry is None and not ed.selected and ed._context_badge['inspection']
    text=ed.c.itemcget(ed.c.find_withtag('context_badge_detail')[0],'text')
    assert 'UNSAVEDGTZ' not in text and frozen(s)==before

def test_restored_Edytuj_button_reactivates_correct_GT_with_one_click(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);escaped(root,h,ed);before=frozen(s)
    item=next(i for i in ed.c.find_withtag('gtz_edit_action') if ed.c.type(i)=='rectangle');l,t,r,b=ed.c.coords(item)
    ed.c.event_generate('<ButtonPress-1>',x=round((l+r)/2),y=round((t+b)/2));ed.c.event_generate('<ButtonRelease-1>',x=round((l+r)/2),y=round((t+b)/2));ready(root,h,ed)
    assert ed.entry is not None and ed.entry_target==(G,OID) and ed.selected==OID and not ed._context_badge['inspection']
    assert frozen(s)==before

@pytest.mark.parametrize('kind',['BBOX','POSE'])
def test_source_badge_restores_without_selection_and_reselects_its_layer(gui,kind):
    root,h,ed,s=gui;nav.select_layer(ed,kind);ready(root,h,ed);before=frozen(s)
    escaped(root,h,ed);assert ed._context_badge['kind']==kind
    l,t,r,b=ed._context_badge['rect'];x,y=round((l+r)/2),round((t+b)/2)
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);ed.c.event_generate('<ButtonRelease-1>',x=x,y=y);ready(root,h,ed)
    assert ed.selected==OID and layers.model(ed)['active']==kind and frozen(s)==before

def test_draft_hides_badge_and_Escape_restores_it_without_approving(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);escaped(root,h,ed);before=frozen(s)
    native.key(h,'d');native.click(h,ed,MANUAL[0]);ready(root,h,ed)
    assert ed.mode=='draw' and not ed.c.find_withtag('context_badge_background') and not ed.c.find_withtag('frame_contrast_underlay')
    escaped(root,h,ed);assert frozen(s)==before

def test_preview_focus_loss_hides_then_focus_return_restores_and_paint_is_read_only(gui):
    import tkinter as tk
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);escaped(root,h,ed);before=frozen(s)
    other=tk.Toplevel(root);field=tk.Entry(other);field.pack();field.focus_force();root.update()
    try:assert not ed.c.find_withtag('context_badge_background')
    finally:other.destroy()
    native.focus_canvas(h);ready(root,h,ed)
    for _ in range(3):ed.overlay();h._paint_image();ready(root,h,ed)
    assert ed._context_badge and ed._context_badge['inspection'] and frozen(s)==before

def test_all_contours_have_same_geometry_dark_edge_and_saturated_palette(gui):
    root,h,ed,s=gui;before=frozen(s);nav.select_layer(ed,'BBOX');ready(root,h,ed)
    shapes=ed.c.find_withtag('navigator_geometry');edges=ed.c.find_withtag('navigator_contrast_underlay')
    assert len(shapes)==len(edges)==2
    for shape in shapes:
        matches=[edge for edge in edges if ed.c.coords(shape)==ed.c.coords(edge)
                 and float(ed.c.itemcget(edge,'width'))==float(ed.c.itemcget(shape,'width'))+2]
        assert len(matches)==1;edge=matches[0]
        assert float(ed.c.itemcget(edge,'width'))==float(ed.c.itemcget(shape,'width'))+2
        assert ed.c.itemcget(edge,'outline')==colors.DARK and ed.c.itemcget(shape,'fill')==''
    assert colors.GREEN=='#00F03C' and len({colors.GREEN,colors.PENDING,colors.POSE,colors.BBOX,colors.RED})==5
    assert min(badge.contrast(c,colors.DARK) for c in (colors.GREEN,colors.PENDING,colors.POSE,colors.BBOX,colors.RED))>=5
    assert frozen(s)==before

def test_Shift_keeps_contrast_edges_for_all_shapes(gui):
    root,h,ed,s=gui;before=frozen(s);ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    main=ed.c.find_withtag('shift_source_preview');edges=ed.c.find_withtag('shift_contrast_underlay')
    assert len(main)==len(edges)==3
    for shape,edge in zip(main,edges):assert ed.c.coords(shape)==ed.c.coords(edge)
    assert frozen(s)==before

def test_new_selection_clears_old_preview_and_other_context_never_reuses_it(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);escaped(root,h,ed)
    nav.select_layer(ed,'POSE',OID);ready(root,h,ed);assert ed._idle_badge is None
    escaped(root,h,ed);ed._idle_badge['context']=('other-session','other-image','other-binding')
    ed.overlay();assert not ed._context_badge and not ed.c.find_withtag('context_badge_background')

@pytest.mark.parametrize('kind',['GTR','POSE','BBOX'])
def test_Escape_keeps_exact_source_candidate_in_shared_GT_stack(multiple,kind):
    root,h,ed,s=multiple;nav.select_layer(ed,kind,'GT:a1');ready(root,h,ed);before=frozen(s)
    source_id=ed._context_badge['attempt_id'];native.key(h,'Escape');ready(root,h,ed)
    assert not ed.selected and ed._context_badge and ed._context_badge['inspection']
    assert ed._context_badge['oid']=='GT:a1' and ed._context_badge['kind']==kind and ed._context_badge['attempt_id']==source_id
    assert ed._stack_counter['count']==3 and frozen(s)==before

def test_new_manual_GT_keeps_its_own_badge_and_single_layer_after_Escape(gui):
    root,h,ed,s=gui;ed.escape();native.key(h,'d');ready(root,h,ed)
    for point in MANUAL:native.click(h,ed,point)
    ready(root,h,ed);manual=ed._context_badge['plate_id'];before=frozen(s)
    native.key(h,'Escape');ready(root,h,ed)
    assert not ed.selected and ed._context_badge and ed._context_badge['inspection']
    assert ed._context_badge['plate_id']==manual and ed._context_badge['attempt_id'] is None and ed._stack_counter['count']==1
    assert partial.progress(core.scene(s,G))['pending_gtr_ids']==[manual] and frozen(s)==before

def test_identical_Pose_cannot_cover_GTR_in_Escape_inspection(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);escaped(root,h,ed)
    gt=ed.c.find_withtag('operator_quad')[0];pose=ed.c.find_withtag('mt_original')[0]
    assert ed.c.coords(gt)==ed.c.coords(pose)
    stack=ed.c.find_all();assert stack.index(gt)>stack.index(pose)
    nav.select_layer(ed,'POSE',OID);ready(root,h,ed)
    assert ed.c.find_all().index(ed.c.find_withtag('mt_original')[0])>ed.c.find_all().index(ed.c.find_withtag('operator_quad')[0])

def test_restored_badge_survives_statistics_refresh_and_source_image_reload(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed)
    from test_z4_gtr_reapproval import edit
    edit(s);ready(root,h,ed);escaped(root,h,ed);before=frozen(s)
    ed.refresh_results()
    native.pump(root,lambda:not ed.in_refresh and not h._saving and ed.available(),seconds=15);ready(root,h,ed)
    assert not ed.selected and ed._context_badge and ed._context_badge['inspection'] and ed._stack_counter['count']==3
    assert frozen(s)==before
