"""FIX03: event-time selection coordinates and explicit Escape deselection.

Native Tk bindings, synthetic source archives, no desktop pointer warping.
These tests preserve review bytes; no data mutation is needed for selection.
"""
from __future__ import annotations

import copy
import pytest

from test_mobile_p2b2_canvas_gui import gui, pump, select, key
from auto_annotation_tool.ranking.mobile_package_experiments import _file_sha256


def native_key(h, name, xy):
    h.canvas.focus_force()
    pump(h.window, lambda: h.window.focus_get() is h.canvas)
    x,y = (round(v) for v in xy)
    h.canvas.event_generate(f"<KeyPress-{name}>",x=x,y=y)
    h.window.update()
    h.canvas.event_generate(f"<KeyRelease-{name}>",x=x,y=y)
    h.window.update()


def assert_selected(h,ed,oid):
    ids=h.canvas.find_withtag("corner_handle")
    assert ed.mode == "edit" and ed.selected == oid
    assert len(ids) == 4 and all(oid in h.canvas.gettags(i) for i in ids)
    assert len(h.subject_tree.selection()) == 1
    assert ed.tree_targets[h.subject_tree.selection()[0]][2] == oid


@pytest.mark.parametrize("cache", [None, "background", "other", "nonfinite"])
def test_s_uses_key_event_position_not_old_motion(gui,cache):
    root,h,ed,s=gui
    before=copy.deepcopy(s.review.to_dict()); sha=_file_sha256(s.path)
    ed.pointer_canvas = (None if cache is None else (float('nan'),1) if cache == "nonfinite"
        else ed.to_canvas((600,500) if cache == "background" else (445,175)))
    xy=ed.to_canvas((225,175))
    native_key(h,"s",xy)
    assert_selected(h,ed,"GT:a0")
    # A later cached hover elsewhere must not change the second S target.
    ed.pointer_canvas=ed.to_canvas((445,175))
    native_key(h,"s",xy)
    assert ed.mode == "select" and not ed.selected
    assert not h.canvas.find_withtag("corner_handle")
    assert not h.subject_tree.selection() and not h.subject_tree.focus()
    assert s.review.to_dict()==before and _file_sha256(s.path)==sha


@pytest.mark.parametrize("outside", [(-1,-1),(0,0),(100000,100000)])
def test_s_outside_target_does_not_fall_back_to_cached_frame(gui,outside):
    root,h,ed,s=gui
    before=copy.deepcopy(s.review.to_dict())
    ed.pointer_canvas=ed.to_canvas((225,175))
    native_key(h,"s",outside)
    assert ed.mode=="select" and not ed.selected
    assert not h.canvas.find_withtag("corner_handle")
    assert not h.subject_tree.selection()
    assert s.review.to_dict()==before


def test_focus_event_interleaving_does_not_replace_test_gesture(gui,monkeypatch):
    root,h,ed,s=gui
    before=copy.deepcopy(s.review.to_dict())
    original=h.canvas.focus_force
    calls=[]
    def focus_with_system_motion():
        original()
        x,y=ed.to_canvas((600,500))
        h.canvas.event_generate("<Motion>",x=round(x),y=round(y))
        calls.append(True)
    monkeypatch.setattr(h.canvas,"focus_force",focus_with_system_motion)
    select(h,ed,(225,175))
    assert calls
    assert_selected(h,ed,"GT:a0")
    assert s.review.to_dict()==before


def test_escape_clears_image_parent_selection_without_touching_loaded_image(gui):
    root,h,ed,s=gui
    iid=next(iter(h._subject_ids))
    h.subject_tree.selection_set(iid);h.subject_tree.focus(iid);pump(root)
    assert h.subject_tree.selection()==(iid,) and not ed.selected
    before=copy.deepcopy(s.review.to_dict());entry=h._image_entry;gid=h.current_invocation
    native_key(h,"Escape",(0,0))
    assert not h.subject_tree.selection() and not h.subject_tree.focus()
    assert not ed.selected and not h.canvas.find_withtag("corner_handle")
    assert h._image_entry==entry and h.current_invocation==gid
    ed.refresh_tree();ed.sync();pump(root)
    assert not h.subject_tree.selection()
    assert s.review.to_dict()==before


def test_escape_clears_child_focus_and_future_pending_selection(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175));assert_selected(h,ed,"GT:a0")
    before=copy.deepcopy(s.review.to_dict())
    ed.pending_tree_target=(h.current_invocation,"GT:a1")
    native_key(h,"Escape",(0,0))
    ed.sync();ed.refresh_tree();pump(root)
    assert ed.pending_tree_target is None and not ed.selected
    assert not h.subject_tree.selection() and not h.subject_tree.focus()
    assert not h.canvas.find_withtag("corner_handle")
    assert s.review.to_dict()==before


def test_parent_row_navigation_is_preserved_until_explicit_escape(gui):
    root,h,ed,s=gui
    iid=next(iter(h._subject_ids))
    h.subject_tree.selection_set(iid);h.subject_tree.focus(iid);pump(root)
    ed.sync();ed.refresh_tree();pump(root)
    assert h.subject_tree.selection()==(iid,) and not ed.selected
    native_key(h,"Escape",(0,0))
    assert not h.subject_tree.selection()


def test_s_toggle_after_late_motion_and_zoom_has_one_owner(gui):
    root,h,ed,s=gui
    h.zoom_image(1.3);pump(root)
    before=copy.deepcopy(s.review.to_dict())
    for source,expected in [((225,175),"GT:a0"),((445,175),"GT:a1"),((445,175),"")]:
        ed.pointer_canvas=None
        native_key(h,"s",ed.to_canvas(source))
        if expected: assert_selected(h,ed,expected)
        else:
            assert not ed.selected and not h.canvas.find_withtag("corner_handle")
    assert s.review.to_dict()==before
