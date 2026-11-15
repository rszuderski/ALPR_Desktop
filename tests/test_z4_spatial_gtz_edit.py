"""GTZ must edit the saved physical reference visible through spatial MT links."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import pytest
from test_z4_auto_spatial_gtr import actual,editor
from auto_annotation_tool.gui.mobile_canvas_editor import CanvasReviewEditor
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_navigator as nav
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks

@pytest.mark.parametrize('address',['source','reference'])
def test_spatial_GTZ_permission_uses_the_saved_reference(actual,address):
    s,gid=actual;ed=editor(s,gid);ed.h._image_entry=core.scene(s,gid)['image_entry'];ed.h._image_source_size=core.scene(s,gid)['image_size']
    row=clean.selected(ed);before=s.sidecar_path.read_bytes()
    oid=row['oid'] if address=='source' else row['reference_id']
    assert CanvasReviewEditor._gt_policy(ed,oid)['gt_editable']
    assert s.sidecar_path.read_bytes()==before

@pytest.fixture
def gui(actual,tmp_path,monkeypatch,request):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    s,gid=actual;monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        h.window.state('normal');h.window.geometry('1649x896+10+10');h.window.deiconify()
        iid=next(i for i,k in h._subject_ids.items() if gid in subject_groups(s,k))
        h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry'] and not h._saving)
        ready(root,h,ed);native.key(h,'a');ready(root,h,ed)
        assert nav.select_layer(ed,'GTR');ready(root,h,ed)
        native.pump(root,lambda:not h._saving and not ed.in_refresh and not ed.refresh_after,seconds=30)
        yield root,h,ed,s,gid
    finally:
        try:next(gen)
        except StopIteration:pass

def badge_click(root,h,ed):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    native.focus_canvas(h);ready(root,h,ed)
    assert ed.c.find_withtag('gtz_edit_action'),'No Edytuj control in the GTR badge'
    l,t,r,b=ed.badges[ed.selected];x=round(r-12);y=round((t+b)/2)
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);ed.c.event_generate('<ButtonRelease-1>',x=x,y=y)
    assert ed.entry is not None,'Click must open a GTZ entry, not just select a layer'
    native.pump(root,lambda:h.window.focus_get() is ed.entry)

@pytest.mark.parametrize('state',['unlinked','confirmed','bad','distinct'])
@pytest.mark.parametrize('method',['enter','autosave'])
def test_click_GTZ_save_undo_redo_same_reference(gui,state,method):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import canonical_action
    from auto_annotation_tool.ranking.mobile_canvas_resolution import keep_separate
    root,h,ed,s,gid=gui;row=clean.selected(ed);oid=ed.selected;pid=row['pid'];ref=row['reference_id']
    if state=='confirmed':canonical_action(s,gid,oid,pid,operation='toggle')
    elif state=='bad':stacks.toggle_unassociated(s,gid,oid,partial_review=True)
    elif state=='distinct':keep_separate(s,gid,ref,oid)
    ready(root,h,ed);nav.select_layer(ed,'GTR');ready(root,h,ed)
    ref=clean.selected(ed)['reference_id']
    before=copy.deepcopy(stacks.snapshot(s,gid));review=copy.deepcopy(s.review.to_dict());raw=s.sidecar_path.read_bytes();rev=s.review.review_revision;parents=copy.deepcopy(core.scene(s,gid)['parents'])
    badge_click(root,h,ed)
    assert ed.entry_target==(gid,ref)
    assert ed.entry_original==before['plates'][pid]['gt_text']
    assert s.sidecar_path.read_bytes()==raw and s.review.to_dict()==review
    ed.entry.delete(0,'end');ed.entry.insert(0,'QA12345')
    if method=='autosave':
        ed.entry.event_generate('<KeyRelease-5>')
        native.pump(root,lambda:s.review.review_revision>rev)
        assert ed.entry is not None,(h.status_var.get(),ed.selected,ed._gt_policy(ref),h.window.focus_get())
    ed.entry.event_generate('<Return>')
    ready(root,h,ed);after=stacks.snapshot(s,gid)
    assert after['plates'][pid]['gt_text']=='QA12345',h.status_var.get()
    assert set(after['plates'])==set(before['plates'])
    assert after['plates'][pid]['gt_quad']==before['plates'][pid]['gt_quad']
    assert after['links']==before['links']
    assert after['plates'][pid]['geometry_state']==before['plates'][pid]['geometry_state']
    assert after['plates'][pid]['confirmed']==before['plates'][pid]['confirmed']
    assert s.review.review_revision==rev+1 and core.scene(s,gid)['parents']==parents
    if state=='bad':assert after['links'][oid]['status']=='UNASSOCIATED_NO_PLATE'
    if state=='distinct':assert after['links']==before['links']
    ed.undo();ready(root,h,ed)
    assert stacks.snapshot(s,gid)['plates'][pid]['gt_text']==before['plates'][pid]['gt_text']
    ed.undo(redo=True);ready(root,h,ed)
    assert stacks.snapshot(s,gid)['plates'][pid]['gt_text']=='QA12345'

def test_Escape_from_badge_keeps_GTZ_and_revision(gui):
    from test_z4_frame_stacks import ready
    root,h,ed,s,gid=gui;raw=s.sidecar_path.read_bytes();rev=s.review.review_revision
    badge_click(root,h,ed);ed.entry.delete(0,'end');ed.entry.insert(0,'DISCARD');ed.entry.event_generate('<Escape>');ready(root,h,ed)
    assert ed.entry is None and s.sidecar_path.read_bytes()==raw and s.review.review_revision==rev

def test_RAW_and_wrong_display_frame_still_cannot_edit_GTZ(actual):
    s,gid=actual;ed=editor(s,gid);scene=core.scene(s,gid);ed.h._image_entry=scene['image_entry'];ed.h._image_source_size=scene['image_size']
    raw=next(c for c in stacks.snapshot(s,gid)['candidates'] if c['role']=='SUPPRESSED')
    assert not CanvasReviewEditor._gt_policy(ed,raw['object_id'])['gt_editable']
    ed.h._image_entry='wrong.jpg'
    assert not CanvasReviewEditor._gt_policy(ed,ed.selected)['gt_editable']
