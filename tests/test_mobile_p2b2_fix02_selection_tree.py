"""P2B2 FIX02: one selection owner, physical S toggle, shared-ROI tree aliases.

Uses native Tk and synthetic source archives. No model or real session required.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from test_mobile_p2b2_canvas_gui import gui, pump, key, click, select, focus_canvas
from test_mobile_p2b2_fix01_canvas_ux import multiple_gui
from test_mobile_p2b2_canvas_review import MANUAL, text_all
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_package_experiments import _file_sha256


def move(h, ed, point):
    focus_canvas(h)
    x, y = ed.to_canvas(point)
    h.canvas.event_generate("<Motion>", x=round(x), y=round(y))
    h.window.update()
    return x, y


def handles(h):
    return h.canvas.find_withtag("corner_handle")


def test_s_under_pointer_selects_then_deselects_without_data_writes(gui):
    root,h,ed,s=gui
    before=copy.deepcopy(s.review.to_dict())
    select(h,ed,(225,175))
    assert ed.mode=="edit" and ed.selected=="GT:a0" and len(handles(h))==4
    assert "S → R" not in ed.selection_hint.cget("text") and "S: odznacz" in ed.selection_hint.cget("text")
    key(h,"s",xy=ed.to_canvas((225,175)))
    assert ed.mode=="select" and not ed.selected and not handles(h)
    assert not h.subject_tree.selection()
    key(h,"s",xy=ed.to_canvas((225,175)))
    assert ed.selected=="GT:a0" and len(handles(h))==4
    assert s.review.to_dict()==before


def test_s_over_other_frame_explicitly_transfers_selection(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));select(h,ed,(445,175))
    assert ed.selected=="GT:a1" and len(handles(h))==4
    assert all("GT:a1" in h.canvas.gettags(i) for i in handles(h))
    key(h,"s")
    assert not ed.selected and not handles(h)


def test_s_over_background_deselects_and_does_not_pick_first_object(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));move(h,ed,(600,500));key(h,"s")
    assert not ed.selected and not handles(h)
    key(h,"s")
    assert not ed.selected and not handles(h)


def test_hover_does_not_select_or_draw_competing_handles(gui):
    root,h,ed,s=gui
    before=s.review.to_dict()
    for pt in ((225,175),(445,175),(150,140),(600,500)):
        move(h,ed,pt)
        assert not ed.selected and not handles(h)
    assert s.review.to_dict()==before


def test_hover_does_not_move_selection_tree_or_repaint_overlay(gui,monkeypatch):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    row=h.subject_tree.selection()
    calls=[]
    monkeypatch.setattr(ed,"request_overlay",lambda:calls.append(True))
    for pt in ((445,175),(520,210),(600,500),(150,140)):
        move(h,ed,pt)
        assert ed.selected=="GT:a0" and h.subject_tree.selection()==row
        assert all("GT:a0" in h.canvas.gettags(i) for i in handles(h))
    assert not calls


@pytest.mark.parametrize("selected",[False,True])
def test_body_left_click_never_chooses_another_frame(gui,selected):
    root,h,ed,s=gui
    if selected: select(h,ed,(225,175))
    before=s.review.to_dict()
    click(h,ed,(445,175))
    assert ed.selected==("GT:a0" if selected else "") and ed.drag is None
    assert s.review.to_dict()==before


def test_unselected_corner_is_not_draggable(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    original=copy.deepcopy(core.scene(s,"mt-one"))
    pt=original["objects"][1]["quad"][0];x,y=ed.to_canvas(pt)
    assert ed.hit(x,y)[1]!="corner"
    h.canvas.event_generate("<ButtonPress-1>",x=round(x),y=round(y))
    assert ed.drag is None
    h.canvas.event_generate("<B1-Motion>",x=round(x+15),y=round(y+3))
    h.canvas.event_generate("<ButtonRelease-1>",x=round(x+15),y=round(y+3));pump(root)
    assert core.scene(s,"mt-one")==original and ed.selected=="GT:a0"


def test_selected_corner_stays_draggable_when_pointer_is_outside_and_after_hovering_other(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));move(h,ed,(445,175))
    original=copy.deepcopy(core.scene(s,"mt-one"))
    pt=original["objects"][0]["quad"][0];x,y=ed.to_canvas(pt)
    h.canvas.event_generate("<Motion>",x=round(x-6),y=round(y-5));pump(root)
    assert ed.hit(x-6,y-5)==("GT:a0","corner",0) and len(handles(h))==4
    h.canvas.event_generate("<ButtonPress-1>",x=round(x-6),y=round(y-5))
    assert ed.drag["oid"]=="GT:a0"
    h.canvas.event_generate("<B1-Motion>",x=round(x+12),y=round(y+4))
    h.canvas.event_generate("<ButtonRelease-1>",x=round(x+12),y=round(y+4));pump(root)
    scene=core.scene(s,"mt-one")
    assert scene["parents"]==original["parents"]
    assert scene["objects"][0]["quad"]!=original["objects"][0]["quad"]
    assert scene["objects"][1]==original["objects"][1] and len(scene["objects"])==2


def test_s_during_drag_does_not_retarget_or_drop_draft(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    x,y=ed.to_canvas((150,140))
    h.canvas.event_generate("<ButtonPress-1>",x=round(x),y=round(y));pump(root)
    state=ed.drag
    move(h,ed,(445,175));key(h,"s")
    assert ed.drag is state and ed.selected=="GT:a0"
    h.canvas.event_generate("<ButtonRelease-1>",x=round(x),y=round(y));pump(root)


def test_right_click_cannot_steal_selection_or_flag_other_object(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));before=s.review.to_dict()
    click(h,ed,(445,175),3)
    assert ed.selected=="GT:a0" and s.review.to_dict()==before
    click(h,ed,(225,175),3)
    assert core._object(core.scene(s,"mt-one"),"GT:a0")["error"]
    assert not core._object(core.scene(s,"mt-one"),"GT:a1")["error"]


def test_right_click_after_deselection_does_not_create_a_flag(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));key(h,"s");before=s.review.to_dict()
    click(h,ed,(225,175),3)
    assert not ed.selected and s.review.to_dict()==before


def test_gt_badge_text_edit_does_not_steal_geometry_selection(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    l,t,r,b=ed.badges["GT:a1"]
    h.canvas.event_generate("<ButtonPress-1>",x=round((l+r)/2),y=round((t+b)/2));pump(root)
    assert ed.entry_target==("mt-one","GT:a1") and ed.selected=="GT:a0"
    ed.entry_var.set("SRS123");ed.entry_enter();pump(root)
    assert core._object(core.scene(s,"mt-one"),"GT:a1")["gt_text"]=="SRS123"
    assert ed.selected=="GT:a0" and all("GT:a0" in h.canvas.gettags(i) for i in handles(h))


def test_s_typed_in_badge_is_text_and_not_toggle(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));ed.start_entry("GT:a0");pump(root)
    ed.entry.delete(0,"end");ed.entry.focus_force();pump(root)
    ed.entry.event_generate("<KeyPress-s>");ed.entry.event_generate("<KeyRelease-s>");pump(root)
    assert ed.selected=="GT:a0" and ed.mode=="edit" and ed.entry_var.get().upper()=="S"
    ed.entry_enter();pump(root)
    assert core._object(core.scene(s,"mt-one"),"GT:a0")["gt_text"]=="S"


def test_mz_badge_can_be_targeted_by_s_but_not_used_as_gt_entry(gui):
    root,h,ed,s=gui
    x,y=ed.card_positions["GT:a1"]
    h.canvas.event_generate("<Motion>",x=round(x+20),y=round(y+10));pump(root);key(h,"s")
    assert ed.selected=="GT:a1"
    h.canvas.event_generate("<ButtonPress-1>",x=round(x+20),y=round(y+10));pump(root)
    assert ed.entry is None and ed.selected=="GT:a1"


def test_s_over_original_targets_existing_child_after_correction(gui):
    root,h,ed,s=gui
    scene=core.scene(s,"mt-one")
    q=copy.deepcopy(scene["objects"][0]["quad"])
    for p in q:p[1]+=180
    core.correct_plate(s,"mt-one","GT:a0",q);ed.cache_key=None;ed.refresh_tree();ed.sync();ed.request_overlay();pump(root)
    before=s.review.to_dict()
    select(h,ed,(225,175))
    assert ed.selected=="GT:a0" and len(handles(h))==4 and s.review.to_dict()==before


def test_windows_repeat_keypress_toggles_once_until_release(gui):
    root,h,ed,s=gui
    x,y = move(h,ed,(225,175));focus_canvas(h)
    for _ in range(5):
        h.canvas.event_generate("<KeyPress-s>",x=round(x),y=round(y));pump(root)
    assert ed.selected=="GT:a0"
    h.canvas.event_generate("<KeyRelease-s>",x=round(x),y=round(y));pump(root)
    key(h,"s")
    assert not ed.selected


def test_x11_repeat_release_press_pair_is_one_toggle(gui):
    root,h,ed,s=gui
    x,y = move(h,ed,(225,175));focus_canvas(h)
    h.canvas.event_generate("<KeyPress-s>",x=round(x),y=round(y));pump(root)
    for _ in range(3):
        h.canvas.event_generate("<KeyRelease-s>",x=round(x),y=round(y))
        h.canvas.event_generate("<KeyPress-s>",x=round(x),y=round(y))
        pump(root)
    assert ed.selected=="GT:a0"
    h.canvas.event_generate("<KeyRelease-s>",x=round(x),y=round(y));pump(root);key(h,"s")
    assert not ed.selected


def test_leaving_and_reentering_canvas_does_not_drop_selected_handles(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));h.canvas.event_generate("<Leave>");pump(root)
    assert ed.selected=="GT:a0" and len(handles(h))==4
    h.canvas.event_generate("<Enter>");move(h,ed,(445,175))
    assert ed.selected=="GT:a0" and len(handles(h))==4


def test_escape_clears_selection_tree_and_handles_but_not_gt(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    assert ed.selected == "GT:a0" and len(handles(h)) == 4
    before=s.review.to_dict();key(h,"Escape")
    assert not ed.selected and not handles(h) and not h.subject_tree.selection()
    assert s.review.to_dict()==before


def test_statistics_callback_does_not_restore_old_selection(gui,monkeypatch):
    root,h,ed,s=gui
    select(h,ed,(225,175));callbacks=[]
    monkeypatch.setattr(h,"_change",lambda operation,**kw:callbacks.append(kw["after"]))
    ed.refresh_results()
    h._saving=True
    try:
        select(h,ed,(445,175))
        assert ed.selected=="GT:a1"  # read-only selection works during statistics
        callbacks.pop()()
        assert ed.selected=="GT:a1"
    finally:h._saving=False
    ed.sync();pump(root)


def test_statistics_callback_does_not_reselect_after_s_toggle_off(gui,monkeypatch):
    root,h,ed,s=gui
    select(h,ed,(225,175));callbacks=[]
    monkeypatch.setattr(h,"_change",lambda operation,**kw:callbacks.append(kw["after"]))
    ed.refresh_results();key(h,"s");callbacks.pop()();pump(root)
    assert not ed.selected and not handles(h)


def test_pending_four_point_draw_is_not_lost_by_s(gui):
    root,h,ed,s=gui
    key(h,"d");click(h,ed,MANUAL[0]);move(h,ed,(225,175));key(h,"s")
    assert ed.mode=="draw" and len(ed.points)==1 and not ed.selected
    assert not core.has_scene(s,"mt-one")


@pytest.fixture
def shared_gui(tmp_path,monkeypatch):
    # Two different observation/subject rows, ONE common source frame + invocation.
    # This is the 06/07 structure reported by the operator, without real photos.
    import test_mobile_p2b2_canvas_review as cases
    import test_mobile_p2b2_canvas_gui as native
    original=cases.attempt
    def split_attempt(aid="a1",**kw):
        kw["subject"]=f"s/sg-1/entity-{int(aid[1:])+1}"
        return original(aid,**kw)
    monkeypatch.setattr(cases,"attempt",split_attempt)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)


def test_shared_invocation_second_image_branch_has_real_expandable_children(shared_gui):
    root,h,ed,s=shared_gui
    roots=list(h._subject_ids)
    assert len(roots)==2 and len(s.mt_invocations)==1
    for rid in roots:
        children=h.subject_tree.get_children(rid)
        assert len(children)==2
        assert {ed.tree_targets[i][2] for i in children}=={"GT:a0","GT:a1"}
    assert "↔01" in h.subject_tree.item(roots[1],"text")
    assert ed.inventory()["total"]==1 and len(core.scene(s,"mt-one")["objects"])==2


def test_selecting_alias_child_stays_under_chosen_branch(shared_gui):
    root,h,ed,s=shared_gui
    second=list(h._subject_ids)[1]
    target=next(i for i in h.subject_tree.get_children(second) if ed.tree_targets[i][2]=="GT:a1")
    h.subject_tree.selection_set(target);h.subject_tree.event_generate("<<TreeviewSelect>>");pump(root)
    assert ed.selected=="GT:a1" and ed.mode=="edit"
    assert h.subject_tree.selection()==(target,)
    ed.refresh_tree();ed.sync();pump(root)
    assert h.subject_tree.selection()==(target,) and ed.tree_branch==second
    move(h,ed,(225,175));key(h,"s")
    chosen=h.subject_tree.selection()[0]
    assert h.subject_tree.parent(chosen)==second and ed.tree_targets[chosen][2]=="GT:a0"


def test_alias_gt_status_and_error_updates_both_views_without_duplicate_metrics(shared_gui):
    root,h,ed,s=shared_gui
    sha=_file_sha256(s.path)
    ed._commit(lambda:core.set_text(s,"mt-one","GT:a0","WA01234"),"Test GT")
    refs=[iid for iid,t in ed.tree_targets.items() if t[2]=="GT:a0"]
    assert len(refs)==2
    assert all("WA01234" in h.subject_tree.item(i,"text") for i in refs)
    ed._commit(lambda:core.toggle_error(s,"mt-one","GT:a1"),"Test flag")
    bad=[iid for iid,t in ed.tree_targets.items() if t[2]=="GT:a1"]
    assert all(h.subject_tree.item(i,"values")[2]=="×" for i in bad)
    core.set_scene_review_state(s,"mt-one",checked=True)
    d=core.summary(s)
    assert d["checked_scenes"]==1 and d["false_positive_mt"]==1 and d["evaluated_reads"]==1
    assert len(d["rows"])==2 and _file_sha256(s.path)==sha


def test_removing_manual_alias_removes_both_references_not_mt(shared_gui):
    root,h,ed,s=shared_gui
    key(h,"d")
    for pt in MANUAL:click(h,ed,pt)
    oid=ed.selected
    assert len([t for t in ed.tree_targets.values() if t[2]==oid])==2
    key(h,"r");pump(root)
    assert not any(t[2]==oid for t in ed.tree_targets.values())
    assert not ed.selected and not handles(h)
    assert len(core.scene(s,"mt-one")["objects"])==2
    assert all(len(h.subject_tree.get_children(r))==2 for r in h._subject_ids)


def test_no_scene_writes_from_alias_navigation_and_expansion(shared_gui):
    root,h,ed,s=shared_gui
    before=copy.deepcopy(s.review.to_dict())
    for rid in h._subject_ids:
        h.subject_tree.item(rid,open=True)
        for child in h.subject_tree.get_children(rid):
            h.subject_tree.selection_set(child);h.subject_tree.event_generate("<<TreeviewSelect>>");pump(root)
    assert s.review.to_dict()==before
