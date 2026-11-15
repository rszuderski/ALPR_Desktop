"""C01-C30: selected geometry badge; model/reference provenance stays separate."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_one_navigator import gui,multiple,pick,keys,frozen
from test_z4_badge4_hud2 import bbox
from test_z4_mt_geometry_layers import source_session
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_info_badge as badge,mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_clean_view as clean
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
from auto_annotation_tool.ranking.mobile_manual_frame_guard import edge_distance
G='mt-one';OID='GT:a0'

def model(ed):return badge.context_model(ed,layers.model(ed))
def visible(ed):return ed._context_badge
def counts(ed):return len(ed.c.find_withtag('context_badge_background'))
def checked_anchor(ed):
    value=visible(ed);assert value and counts(ed)==1
    x,y,r,b=value['rect'];assert 0<=x<r<=ed.c.winfo_width() and 0<=y<b<=ed.c.winfo_height()
    assert not any(clean.intersects(value['rect'],obstacle) for obstacle in ed.clean_geometry_boxes)
    tether=ed.c.coords(ed.c.find_withtag('badge_tether')[0])
    assert edge_distance(tether[-2:],[ed.to_canvas(p) for p in value['quad']])<1e-7
    assert len(ed.c.find_withtag('context_badge_title'))==len(ed.c.find_withtag('context_badge_detail'))==1
    assert not ed.c.find_withtag('badge4_background')

@pytest.fixture
def unlinked(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data',n=1);monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def test_C01_bbox_reads_matcher_not_polygon_or_operator_label(gui):
    root,h,ed,s=gui
    partial.confirm_gtr(s,G,data.snapshot(s,G)['links'][OID]['plate_id'])
    ready(root,h,ed);pick(root,h,ed,'BBOX');value=visible(ed);result=evaluate_unit(s,G)
    pair=next(p for p in result['matched_pairs'] if p['attempt_id']=='a0')
    assert value['detail']=='BBOX IoU: '+badge.percent(100*pair['bbox_iou'])+' · TP'
    assert all(t not in value['detail'] for t in ('GTZ','MZ','Polygon','NMCE','WA12345'))
    assert value['quad']==[list(p) for p in nav.quad(layers.model(ed),'BBOX')] or value['quad']==nav.quad(layers.model(ed),'BBOX')
    checked_anchor(ed)

def test_C02_pose_uses_source_quad_and_existing_pair_metrics(gui):
    root,h,ed,s=gui;pick(root,h,ed,'POSE');value=visible(ed);row=data.diagnostics(s,[G])['rows'][0]
    assert value['quad']==row['mt_quad'] and value['attempt_id']=='a0'
    assert value['detail']=='Polygon IoU: '+badge.percent(row['polygon_iou_pct'])+' · NMCE: '+badge.percent(row['nmce_pct'])
    assert all(t not in value['detail'] for t in ('GTZ','MZ','BBOX','TP','FP'));checked_anchor(ed)

def test_C03_C14_reference_contains_only_its_GTZ(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');value=visible(ed)
    assert value['detail']=='GTZ: WA12345' and 'WA01234' not in value['detail']
    assert value['plate_id']==clean.selected(ed)['pid'] and value['quad']==clean.selected(ed)['gt']
    assert 'Obrys operatora' in value['title'];checked_anchor(ed)

def test_C04_C05_keyboard_and_menu_share_the_same_context(gui):
    root,h,ed,s=gui;before=frozen(s);pick(root,h,ed,'BBOX');expected={}
    for kind in ('POSE','GTR','BBOX'):
        keys(root,h,ed,'x');assert layers.model(ed)['active']==kind
        expected[kind]=copy.deepcopy(visible(ed));checked_anchor(ed)
        assert kind in ed.c.gettags(ed.c.find_withtag('navigator_geometry')[-1])
    for kind in ('GTR','POSE','BBOX'):keys(root,h,ed,'z');assert model(ed)['kind']==kind
    for index,kind in enumerate(('BBOX','POSE','GTR')):
        ed.navigator.menu.invoke(index);ready(root,h,ed);assert visible(ed)==expected[kind]
    assert frozen(s)==before

def test_C06_Escape_restores_inspection_and_focus_out_clears_badge_and_hits(gui):
    import tkinter as tk
    root,h,ed,s=gui;pick(root,h,ed,'BBOX');before=frozen(s)
    ed.escape();ready(root,h,ed);assert counts(ed)==1 and visible(ed)['inspection'] and visible(ed)['kind']=='BBOX' and not ed.selected
    nav.select_result(ed,0);ready(root,h,ed);native.focus_canvas(h);ed.overlay();assert counts(ed)==1
    other=tk.Toplevel(root);field=tk.Entry(other);field.pack();field.focus_force();root.update()
    try:assert counts(ed)==0 and visible(ed) is None and not ed.mt_source_badge_hits
    finally:other.destroy()
    native.focus_canvas(h);root.update();assert counts(ed)==1 and frozen(s)==before

def test_C06_focus_recovery_rechecks_actual_window_not_stale_focus_flag(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');native.focus_canvas(h)
    # Entry.bindtags deliberately excludes the Toplevel in this application.
    ed.start_entry(OID);ed.entry.focus_force();root.update()
    ed._badge_window_inactive=True;badge.clear(ed);ed.overlay()
    assert badge.focus_in_window(ed) and counts(ed)==1 and not ed._badge_window_inactive
    assert ed.entry is not None and ed.entry.winfo_ismapped()

def test_C07_GTZ_edit_refreshes_reference_and_sidebar_not_source_badge(gui):
    root,h,ed,s=gui;pick(root,h,ed,'BBOX');source=model(ed)['detail'];pick(root,h,ed,'GTR')
    box=ed.badges[OID];x,y=round((box[0]+box[2])/2),round((box[1]+box[3])/2)
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y);root.update()
    assert ed.entry is not None;ed.entry_var.set('NEWGT');ed.entry_enter();ready(root,h,ed)
    assert model(ed)['detail']=='GTZ: NEWGT' and 'GTZ: NEWGT' in clean.context(ed,{})['object']
    pick(root,h,ed,'BBOX');assert model(ed)['detail']==source and 'NEWGT' not in model(ed)['detail']

@pytest.mark.parametrize('reason,expected',[('','nie wpisano'),('nieczytelna','nieczytelny')])
def test_C08_empty_and_unreadable_GTZ_are_not_OCR(gui,reason,expected):
    root,h,ed,s=gui;data.edit_gt(s,G,OID,text='',reason=reason);ready(root,h,ed);pick(root,h,ed,'GTR')
    assert model(ed)['detail']=='GTZ: '+expected and 'WA01234' not in model(ed)['detail']

def test_C09_operator_OK_does_not_imply_matcher_TP(gui):
    root,h,ed,s=gui;q=copy.deepcopy(clean.selected(ed)['gt'])
    for p in q:p[0]+=500
    data.edit_gt(s,G,OID,quad=q)
    partial.confirm_gtr(s,G,data.snapshot(s,G)['links'][OID]['plate_id'])
    ready(root,h,ed);pick(root,h,ed,'BBOX')
    assert clean.selected(ed)['decision']==partial.PLATE and evaluate_unit(s,G)['final_outcomes'][0]['outcome']=='FP'
    # Show the overlap with the explicitly linked reference even below the
    # matcher threshold; this diagnostic must not create a matched pair.
    assert model(ed)['detail']=='BBOX IoU z GTR 1: 0,0% · FP'
    assert badge.measurements(ed,clean.selected(ed))['bbox_iou'] is None
    assert evaluate_unit(s,G)['matched_pairs']==[]

def test_C10_C11_missing_reference_is_not_zero_geometry(unlinked):
    root,h,ed,s=unlinked;before=frozen(s);pick(root,h,ed,'BBOX')
    assert model(ed)['detail']=='BBOX IoU: — · ocena oczekuje'
    pick(root,h,ed,'POSE');assert model(ed)['detail']=='Polygon IoU: — · NMCE: —'
    assert 'Brak powiązanej referencji' in clean.context(ed,{})['object'] and frozen(s)==before

def test_C12_C13_INVALID_QUAD_NOT_RUN_never_becomes_Pose(bbox):
    root,h,ed,s=bbox;pick(root,h,ed,'BBOX');assert model(ed)['kind']=='BBOX'
    assert not nav.select_layer(ed,'POSE') and [l['kind'] for l in layers.model(ed)['layers']]==['BBOX','GTR']
    assert 'MZ' not in model(ed)['detail'] and 'nie uruchomiono' in clean.context(ed,{})['object']
    assert data.diagnostics(s,[G])['rows'][0]['polygon_iou_pct'] is None

@pytest.mark.parametrize('ref,inf',[(True,True),(True,False),(False,True),(False,False)])
def test_C15_to_C18_single_badge_obeys_global_flags(gui,ref,inf):
    root,h,ed,s=gui;before=frozen(s);ed.ref_visible.set(ref);ed.inf_visible.set(inf)
    for kind in ('BBOX','POSE','GTR'):
        pick(root,h,ed,kind);assert counts(ed)==int(ref if kind=='GTR' else inf)
        assert (ed.ref_visible.get(),ed.inf_visible.get())==(ref,inf)
        assert 'WA12345' in clean.context(ed,{})['object'] and 'WA01234' in clean.context(ed,{})['object']
        assert len(ed.c.find_withtag('operator_quad'))==int(ref)
    assert frozen(s)==before

def test_C19_Shift_reveal_does_not_duplicate_badge_or_change_flags(gui):
    root,h,ed,s=gui;pick(root,h,ed,'POSE');ed.ref_visible.set(False);ed.inf_visible.set(False);ed.overlay()
    before=frozen(s);assert counts(ed)==0
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay();assert counts(ed)==1
    ed.shift_preview.release(NS(keysym='Shift_L'));ed.overlay();assert counts(ed)==0
    keys(root,h,ed,'Escape');keys(root,h,ed,'d');ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert counts(ed)==0 and frozen(s)==before and not ed.ref_visible.get() and not ed.inf_visible.get()

def test_C20_viewport_reanchors_without_recomputing_metrics(gui,monkeypatch):
    root,h,ed,s=gui;pick(root,h,ed,'POSE');before=frozen(s);quad=copy.deepcopy(model(ed)['quad'])
    def unexpected(*a,**k):pytest.fail('Viewport must use the current read-only metric snapshot')
    from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as detection
    from auto_annotation_tool.gui.mobile_canvas_zoom import paint_viewport
    with monkeypatch.context() as guard:
        guard.setattr(data,'diagnostics',unexpected)
        guard.setattr(detection,'evaluate_unit',unexpected)
        guard.setattr(ed,'_render_contexts',unexpected)
        h._view_zoom=1.2;paint_viewport(h,preview=True);checked_anchor(ed)
        h._view_center=(h._view_center[0]-10,h._view_center[1]-8);paint_viewport(h,preview=True);checked_anchor(ed)
    assert model(ed)['quad']==quad and frozen(s)==before

def test_C21_small_edge_plate_long_GTZ_remains_inside_and_clear(gui):
    root,h,ed,s=gui;long='LONG1234567890'*12;data.edit_gt(s,G,OID,text=long);ready(root,h,ed);pick(root,h,ed,'GTR')
    checked_anchor(ed);assert visible(ed)['displayed'][1].endswith('…') and long in clean.context(ed,{})['object']
    for dark in (False,True):
        p=badge.badge_palette(dark);assert badge.contrast(p['fg'],p['bg'])>=4.5 and badge.contrast(p['name'],p['bg'])>=4.5

def test_C21_tiny_reference_at_ROI_edge_has_safe_badge(gui):
    root,h,ed,s=gui
    oid=core.add_plate(s,G,[[101,101],[115,101],[115,108],[101,108]],partial_review=True)
    data.edit_gt(s,G,oid,text='EDGE');ed._choose_plate(oid);ready(root,h,ed)
    checked_anchor(ed)
    points=[ed.to_canvas(p) for p in model(ed)['quad']]
    assert max(p[0] for p in points)-min(p[0] for p in points)<30
    assert min(p[0] for p in points)<ed.c.winfo_width()*.2
    h.window.geometry('1180x760');root.update();ready(root,h,ed);checked_anchor(ed)

def test_C22_changing_FINAL_changes_attempt_and_summary(multiple):
    root,h,ed,s=multiple;pick(root,h,ed,'POSE');before=frozen(s)
    for oid,text in (('GT:a1','WA11234'),('GT:a2','WA21234'),(OID,'WA01234')):
        keys(root,h,ed,'Down');assert model(ed)['oid']==oid and model(ed)['attempt_id']==oid[3:]
        assert text in clean.context(ed,{})['object'] and counts(ed)==1
    assert frozen(s)==before

def test_C24_manual_reference_has_no_fictional_MT(gui):
    root,h,ed,s=gui;oid=core.add_plate(s,G,[[650,350],[800,350],[800,420],[650,420]],partial_review=True)
    data.edit_gt(s,G,oid,text='MANUALGT');ed._choose_plate(oid);ready(root,h,ed);value=model(ed)
    assert value['kind']=='GTR' and value['attempt_id'] is None and value['detail']=='GTZ: MANUALGT'
    assert 'brak powiązanej detekcji' in clean.context(ed,{})['object'];checked_anchor(ed)

@pytest.mark.parametrize('count',[1,2,3])
def test_C25_D_hides_badge_and_cancel_restores_live_inspection(gui,count):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');before=frozen(s);q=[[650,350],[800,350],[800,420],[650,420]]
    keys(root,h,ed,'Escape');keys(root,h,ed,'d')
    for point in q[:count]:native.click(h,ed,point)
    assert len(ed.points)==count and counts(ed)==0
    keys(root,h,ed,'Escape');assert not ed.points and counts(ed)==1 and visible(ed)['inspection'] and visible(ed)['kind']=='GTR' and frozen(s)==before
    nav.select_result(ed,0);ready(root,h,ed);assert counts(ed)==1

def test_C25_edit_undo_redo_refresh_reference_anchor(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');old=copy.deepcopy(model(ed)['quad'])
    keys(root,h,ed,'d')
    assert ed.mode=='edit' and ed.draw_target is None
    x,y=map(round,ed.to_canvas(old[0]))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y)
    h.canvas.event_generate('<B1-Motion>',x=x+2,y=y+1)
    h.canvas.event_generate('<ButtonRelease-1>',x=x+2,y=y+1)
    ready(root,h,ed);assert ed.mode!='draw';new=copy.deepcopy(model(ed)['quad']);assert new!=old
    ed.undo();ready(root,h,ed);assert model(ed)['quad']==old
    ed.undo(redo=True);ready(root,h,ed);assert model(ed)['quad']==new

def test_C26_source_and_reference_badges_keep_BAD_and_Space_distinct(gui):
    root,h,ed,s=gui
    partial.confirm_gtr(s,G,data.snapshot(s,G)['links'][OID]['plate_id'])
    ready(root,h,ed);pick(root,h,ed,'GTR');before=frozen(s);box=visible(ed)['rect']
    x,y=round(box[0]+4),round(box[1]+4)
    h.canvas.event_generate('<ButtonPress-3>',x=x,y=y);h.canvas.event_generate('<ButtonRelease-3>',x=x,y=y);ready(root,h,ed)
    assert frozen(s)==before
    refs=copy.deepcopy(data.snapshot(s,G)['plates']);keys(root,h,ed,'space');assert data.snapshot(s,G)['plates']==refs
    pick(root,h,ed,'BBOX');box=ed.mt_source_badge_hits[OID];x,y=round((box[0]+box[2])/2),round((box[1]+box[3])/2)
    h.canvas.event_generate('<ButtonPress-3>',x=x,y=y);h.canvas.event_generate('<ButtonRelease-3>',x=x,y=y);ready(root,h,ed)
    assert core._object(ed._read(),OID)['error'] and data.snapshot(s,G)['plates']==refs and counts(ed)==1

def test_C27_summary_and_detailed_scientific_panel_remain_available(gui):
    root,h,ed,s=gui;before=frozen(s);summary=core.summary(s);hud=ed.source_hud.cget('text')
    for kind in ('BBOX','POSE','GTR'):
        pick(root,h,ed,kind);panel=clean.context(ed,{})['object']
        assert all(name in panel for name in ('GTR','Model tablic MT','GTZ','Model znaków MZ','WA12345','WA01234'))
        assert ('BBOX IoU' if kind=='BBOX' else 'Polygon IoU') in panel
        assert ed.source_hud.cget('text')==hud and not ed.c.find_withtag('badge4_background')
    assert core.summary(s)==summary and frozen(s)==before
