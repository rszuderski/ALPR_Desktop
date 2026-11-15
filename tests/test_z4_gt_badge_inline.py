"""Direct GT badge editing; native Tk and unchanged source/physical identities."""
import copy,shutil
from pathlib import Path
import pytest
from test_z4_clean_canvas import night,pair,pick,focus,frozen,unchanged,attach
from test_z4_frame_stacks import ready
import test_mobile_p2b2_canvas_gui as native
from auto_annotation_tool.gui import mobile_canvas_clean_view as view
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession

def click_badge(ui,oid):
    root,h,ed,s,gid=ui;l,t,r,b=ed.badges[oid]
    native.focus_canvas(h)
    ed._badge_click_time=getattr(ed,'_badge_click_time',1000)+1000
    h.canvas.event_generate('<ButtonPress-1>',x=round((l+r)/2),y=round((t+b)/2),time=ed._badge_click_time)
    h.canvas.event_generate('<ButtonRelease-1>',x=round((l+r)/2),y=round((t+b)/2),time=ed._badge_click_time+10)
    native.pump(root)
    assert ed.entry is not None and ed.entry.place_info()['in'] is h.canvas
    assert h.window.focus_get() is ed.entry
    assert ed.entry.winfo_rooty()>=h.canvas.winfo_rooty()
    assert ed.entry.winfo_rooty()+ed.entry.winfo_height()<=h.canvas.winfo_rooty()+h.canvas.winfo_height()

def test_B01_real_badge_edits_without_selecting_or_top_bar(night):
    root,h,ed,s,gid=night;oid=core.scene(s,gid)['objects'][0]['object_id'];before=frozen(s);parents=copy.deepcopy(core.scene(s,gid)['parents'])
    assert not ed.selected and not ed.clean_gt_slot.winfo_ismapped()
    labels=[h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('clean_label_text')]
    assert any('ZS3838L' in label for label in labels),(labels,ed.c.winfo_width(),ed.clean_geometry_boxes,ed.badges)
    click_badge(night,oid);unchanged(s,before);assert not ed.selected
    assert 'ZS3838L' in ed.contexts.data['object'] and 'Wybierz ramkę' not in ed.contexts.data['object']
    ed.entry_var.set('ZS3838LX');ed.entry.event_generate('<Return>');ready(root,h,ed)
    assert core._object(core.scene(s,gid),oid)['gt_text']=='ZS3838LX' and core.scene(s,gid)['parents']==parents
    ed.undo();ready(root,h,ed);assert core._object(core.scene(s,gid),oid)['gt_text']=='ZS3838L'
    re=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);assert core.summary(re)==core.summary(s)

def test_B02_stack_badge_uses_one_reference_and_own_MZ(pair):
    root,h,ed,s,gid=pair;oid=pick(pair,1);before=copy.deepcopy(core.scene(s,gid)['parents']);rev=s.review.review_revision
    click_badge(pair,oid);assert ed.entry_target==(gid,'GT:a0')
    ed.entry_var.set('COMMONGT');ed.entry.event_generate('<Return>');ready(root,h,ed)
    assert s.review.review_revision==rev+1 and len(data.snapshot(s,gid)['plates'])==1
    assert all(r['gt_text']=='COMMONGT' for r in data.diagnostics(s,[gid])['rows']) and core.scene(s,gid)['parents']==before
    assert len({r['mz_prediction'] for r in data.diagnostics(s,[gid])['rows']})==2

def test_B03_focus_keeps_active_gt_badge_available(night):
    root,h,ed,s,gid=night;oid=pick(night);click_badge(night,oid)
    q=core._object(core.scene(s,gid),oid)['quad'];x,y=ed.to_canvas(q[0]);before=frozen(s)
    h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y),time=10000);native.pump(root);ed.overlay()
    assert ed.gt_focus_id==oid and ed.entry is None and oid in ed.badges
    assert len(h.canvas.find_withtag('corner_handle'))==4
    assert h.canvas.find_withtag('gt_badge') and h.canvas.find_withtag('clean_label')
    for tag in ('mt_original','overlap_hud','analyzed_roi'):assert not h.canvas.find_withtag(tag)
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y),time=10010);ready(root,h,ed)
    assert oid in ed.badges and ed.gt_focus_id==oid;ed.escape();ready(root,h,ed)
    assert oid in ed.badges;unchanged(s,before)

def test_B04_entry_tracks_badge_zoom_pan_resize_and_fallback(night):
    root,h,ed,s,gid=night;oid=pick(night);before=frozen(s)
    for size in ('1649x896','1180x760'):
        h.window.geometry(size+'+10+10');ed.fit();ready(root,h,ed);click_badge(night,oid)
        for z in (1,2,3):
            h._view_zoom=z;h._view_center=(h._view_center[0]+1,h._view_center[1]+1);h._paint_image();native.pump(root)
            for box in ed.clean_label_boxes:assert all(not view.intersects(box,q) for q in ed.clean_geometry_boxes)
            if oid in ed.badges:
                l,t,r,b=ed.badges[oid];assert ed.entry.place_info()['in'] is h.canvas
                prefix=ed._badge_font.measure('GTZ: ')+6
                assert abs(ed.entry.winfo_x()-l-prefix)<=1 and ed.entry.winfo_width()<=r-l-prefix
        # Deliberately fill the viewport: fallback must not cover the GT; fit restores inline entry.
        q=core._object(core.scene(s,gid),oid)['quad'];h._view_center=tuple(sum(p[i] for p in q)/4 for i in (0,1));h._view_zoom=100;h._paint_image();native.pump(root)
        assert oid not in ed.badges and ed.entry.place_info()['in'] is ed.clean_gt_slot
        ed.fit();ready(root,h,ed);assert ed.entry.place_info()['in'] is h.canvas
        assert ed.entry.winfo_width()<=ed.badges[oid][2]-ed.badges[oid][0]
        ed.entry.event_generate('<Return>');ready(root,h,ed)
    unchanged(s,before)

def test_B05_native_Tab_from_badge_is_not_stack_navigation(pair):
    root,h,ed,s,gid=pair;oid=pick(pair,1)
    for backwards in (False,True):
        click_badge(pair,oid);entry=ed.entry;entry.focus_force();native.pump(root)
        target=entry.tk_focusPrev() if backwards else entry.tk_focusNext()
        entry.event_generate('<KeyPress-Tab>',state=1 if backwards else 0);ready(root,h,ed)
        assert ed.entry is None and ed.selected==oid and h.window.focus_get() is target

def test_B06_RAW_FP_have_no_editable_GT_badges(night):
    root,h,ed,s,gid=night;ready(root,h,ed);before=frozen(s)
    excluded=[r['object_id'] for r in data.snapshot(s,gid)['candidates'] if r['role']=='SUPPRESSED' or r['association_status']=='UNASSOCIATED_NO_PLATE']
    assert excluded and all(oid not in ed.badges for oid in excluded);unchanged(s,before)

def test_B07_escape_cancels_unsaved_text_and_keeps_saved_review(night):
    root,h,ed,s,gid=night;oid=core.scene(s,gid)['objects'][0]['object_id'];before=frozen(s)
    click_badge(night,oid);ed.entry_var.set('UNSAVED');ed.entry.event_generate('<Escape>');ready(root,h,ed)
    assert ed.entry is None and oid in ed.badges;unchanged(s,before)

def test_B09_unselected_entry_survives_fallback_repaints(night):
    root,h,ed,s,gid=night;oid=core.scene(s,gid)['objects'][0]['object_id'];before=frozen(s)
    click_badge(night,oid);assert not ed.selected;entry=ed.entry
    q=core._object(core.scene(s,gid),oid)['quad']
    h._view_center=tuple(sum(p[i] for p in q)/4 * (h._image.size[i]/h._image_source_size[i]) for i in (0,1))
    h._view_zoom=100;h._paint_image();native.pump(root)
    for _ in range(3):
        ed.overlay();native.pump(root)
        assert ed.entry is entry and h.window.focus_get() is entry and entry.place_info()['in'] is ed.clean_gt_slot
    ed.fit();ready(root,h,ed);assert ed.entry is entry and entry.place_info()['in'] is h.canvas
    ed.entry.event_generate('<Escape>');ready(root,h,ed);unchanged(s,before)

def test_B08_two_real_plates_badges_write_only_their_target(tmp_path,monkeypatch):
    source=Path(__file__).resolve().parents[1]/'output/P0_Z4_CLEAN_UX_20261025_001/native_after_two'
    archive=next(source.glob('*.alprsession'));shutil.copy2(source/'review.json',tmp_path/'review.json')
    s=MobileReviewSession.open(archive,sidecar_path=tmp_path/'review.json');gid=next(iter(s.mt_invocations))
    gen=attach(s,tmp_path,monkeypatch);root,h,ed,_=next(gen);ui=(root,h,ed,s,gid)
    try:
        ready(root,h,ed);objects=core.scene(s,gid)['objects'];ids=list(data.snapshot(s,gid)['plates']);old=objects[0]['gt_text']
        assert len(ids)==len(ed.badges)==2
        click_badge(ui,objects[1]['object_id']);ed.entry_var.set('SECONDGT');ed.entry.event_generate('<Return>');ready(root,h,ed)
        assert core._object(core.scene(s,gid),objects[0]['object_id'])['gt_text']==old
        assert core._object(core.scene(s,gid),objects[1]['object_id'])['gt_text']=='SECONDGT'
        assert list(data.snapshot(s,gid)['plates'])==ids
    finally:
        try:next(gen)
        except StopIteration:pass
