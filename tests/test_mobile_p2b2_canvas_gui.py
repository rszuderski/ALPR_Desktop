"""Native Tk smoke tests. No inference/models; synthetic verified source frame.

Run under normal Windows desktop or Linux with a display/Xvfb. Tests skip only
when Tk cannot open a display; core/contract tests remain platform-independent.
"""
from __future__ import annotations

import copy
import gc
import time
from types import SimpleNamespace

import pytest
import tkinter as tk
from PIL import ImageGrab

from test_mobile_p2b2_canvas_review import source_session, MANUAL
from auto_annotation_tool.gui import z4_mobile_sample_review as module
from auto_annotation_tool.ranking import mobile_canvas_review as core


def pump(root,predicate=lambda:True,seconds=8):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        root.update()
        if predicate():
            return
        time.sleep(.015)
    raise AssertionError("Tk condition timed out")


@pytest.fixture
def gui(tmp_path,monkeypatch):
    # Collect closed Tk fixture cycles on the Tk/main thread, BEFORE the next
    # window starts its loader worker. Otherwise GC may finalize Tcl in a worker.
    gc.collect()
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display required")
    root.withdraw()
    errors=[]
    root.report_callback_exception=lambda *args:errors.append(args)
    s=source_session(tmp_path)
    store=SimpleNamespace(root_dir=tmp_path/"rank",apply_human_review=lambda session:False)
    browser=SimpleNamespace(window=root,palette={"panel":"#18252E","fg":"#EFF3F4","muted":"#B0BEC5","accent":"#FFCA80"},
        error="#FF7788",warning="#FFCC88",store=store,report_by_iid={})
    monkeypatch.setattr(module,"MobileReviewSession",lambda bundle:s)
    monkeypatch.setattr(module,"MobilePackageExperimentStore",lambda *args,**kwargs:store)
    h=module.MobileSampleReviewWindow(browser,s.bundle.report,s.bundle)
    try:
        pump(root,lambda:h.session is not None and h._image is not None and not h._saving)
        pump(root, lambda: (h.canvas.winfo_ismapped() and h.canvas.winfo_width() > 1
            and h.canvas.winfo_height() > 1 and h._view_center is not None
            and h._canvas_editor.skey == h.current_invocation))
        yield root,h,h._canvas_editor,s
        assert not errors,errors
    finally:
        h._canvas_editor.escape()
        pump(root, lambda: not h._saving)
        h.close()
        pump(root, lambda: h._closed)
        h._executor.shutdown(wait=True,cancel_futures=True)
        root.destroy()


def click(h,ed,point,button=1):
    x,y=ed.to_canvas(point)
    h.canvas.event_generate(f"<ButtonPress-{button}>",x=round(x),y=round(y))
    h.canvas.event_generate(f"<ButtonRelease-{button}>",x=round(x),y=round(y))
    h.window.update()


def focus_canvas(h):
    """Settle focus BEFORE emitting a synthetic pointer/key gesture."""
    h.canvas.focus_force()
    pump(h.window, lambda: h.window.focus_get() is h.canvas)


def key(h,name,*,xy=None):
    # Snapshot the test gesture before focus can enqueue native Motion/Leave.
    # Unspecified event_generate coordinates are zero, not the last mouse point.
    if xy is None:
        xy = h._canvas_editor.pointer_canvas
    if xy is None:
        xy = (-1,-1)
    x,y = (round(v) for v in xy)
    focus_canvas(h)
    h.canvas.event_generate("<KeyPress-"+name+">",x=x,y=y)
    h.window.update()
    h.canvas.event_generate("<KeyRelease-"+name+">",x=x,y=y)
    h.window.update()


def select(h,ed,point):
    """Native bindings, explicit position + one S press; no LPM or pointer warp."""
    focus_canvas(h)
    x,y=ed.to_canvas(point)
    assert 0 <= x < h.canvas.winfo_width() and 0 <= y < h.canvas.winfo_height()
    h.canvas.event_generate("<Motion>",x=round(x),y=round(y))
    # KeyPress already carries the intended point; a pending system Motion
    # must not substitute the real desktop cursor for the synthetic gesture.
    key(h,"s",xy=(x,y))


def test_source_canvas_is_default_and_old_forms_are_hidden(gui):
    root,h,ed,s=gui
    assert h._image_entry=="samples/frames/source.jpg"
    assert len(h.canvas.find_withtag("mt_original"))==2
    assert len(h.canvas.find_withtag("operator_quad"))==2
    assert not h._multi_host.winfo_ismapped()
    assert not h.mt_count_group.winfo_ismapped()
    assert not h.gt_entry.winfo_ismapped()
    assert len(ed.badges)==2
    assert h.canvas.find_withtag("analyzed_roi")
    assert h.review_sidebar_scroll.winfo_width() <= 250


def test_native_d_four_clicks_draws_lines_and_increments_only_at_four(gui):
    root,h,ed,s=gui
    key(h,"d")
    click(h,ed,MANUAL[0])
    assert len(ed.points)==1 and not core.has_scene(s,"mt-one")
    assert not ed.prepare_navigation() and len(ed.points)==1
    for p in MANUAL[1:]:click(h,ed,p)
    assert not ed.points
    assert core.counts(core.scene(s,"mt-one"))["expected"]==1
    manual=next(o for o in core.scene(s,"mt-one")["objects"] if not o["parent_id"])
    assert manual["object_id"] in ed.stack_badges
    row=next(r for r in core.summary(s)["rows"] if r["object_id"]==manual["object_id"])
    assert row["mz_prediction"]==""


def test_native_s_flag_keeps_both_parent_and_child_on_canvas(gui):
    root,h,ed,s=gui
    select(h,ed,(220,175))
    assert ed.selected=="GT:a0"
    click(h,ed,(220,175),button=3)
    assert core.scene(s,"mt-one")["objects"][0]["error"]
    assert len(h.canvas.find_withtag("mt_original"))==2
    assert len(h.canvas.find_withtag("operator_quad"))==2
    assert len(h.canvas.find_withtag("error_flag"))==2
    assert core.counts(core.scene(s,"mt-one"))["expected"]==1
    click(h,ed,(220,175),button=3)
    assert not core.scene(s,"mt-one")["objects"][0]["error"]


def test_native_corner_drag_keeps_original_and_does_not_add_a_plate(gui):
    root,h,ed,s=gui
    original=copy.deepcopy(core.scene(s,"mt-one")["parents"])
    select(h,ed,(220,175))
    assert len(h.canvas.find_withtag("corner_handle"))>=4
    p=original[0]["quad"][0]
    x,y=ed.to_canvas(p);nx,ny=ed.to_canvas((p[0]+15,p[1]+4))
    h.canvas.event_generate("<ButtonPress-1>",x=round(x),y=round(y))
    h.canvas.event_generate("<B1-Motion>",x=round(nx),y=round(ny))
    h.canvas.event_generate("<ButtonRelease-1>",x=round(nx),y=round(ny))
    pump(root)
    scene=core.scene(s,"mt-one")
    assert scene["parents"]==original
    assert scene["objects"][0]["quad"] != original[0]["quad"]
    assert core.counts(scene)["expected"]==1


def test_badge_enter_focusout_and_letters_do_not_trigger_canvas_shortcuts(gui):
    root,h,ed,s=gui
    ed.start_entry("GT:a0");pump(root)
    original_mode=ed.mode
    ed.entry_var.set("WADSQE1")
    ed.entry_typed(SimpleNamespace(keysym="d"))
    ed.entry_enter();pump(root)
    assert core.scene(s,"mt-one")["objects"][0]["gt_text"]=="WADSQE1"
    assert ed.mode==original_mode and ed.entry is None
    ed.start_entry("GT:a1");pump(root)
    ed.entry_var.set("WA11234")
    assert ed.prepare_navigation()
    assert core.scene(s,"mt-one")["objects"][1]["gt_text"]=="WA11234"
    assert core.scene(s,"mt-one")["objects"][0]["gt_text"]=="WADSQE1"


def test_zoom_roundtrip_and_idle_refresh_do_not_reset_source_view(gui):
    root,h,ed,s=gui
    h.zoom_image(2.0)
    pump(root)
    point=(420.25,365.75)
    assert ed.to_source(*ed.to_canvas(point))==pytest.approx(point)
    zoom=h._view_zoom;center=h._view_center;entry=h._image_entry
    ed.start_entry("GT:a0");ed.entry_var.set("WA01234");ed.entry_enter()
    ed.refresh_results()
    pump(root,lambda:not h._saving and not ed.in_refresh)
    assert h._view_zoom==zoom and h._view_center==center and h._image_entry==entry


def test_manual_wrong_flag_does_not_count_as_mt_false_positive(gui):
    root,h,ed,s=gui
    key(h,"d")
    for p in MANUAL:click(h,ed,p)
    frozen=s.sidecar_path.read_bytes()
    click(h,ed,(720,385),button=3)  # D completed: its new frame is already selected.
    c=core.counts(core.scene(s,"mt-one"))
    assert c["operator_errors"]==0 and c["false_positive"]==0 and c["expected"]==1
    assert s.sidecar_path.read_bytes()==frozen  # a physical manual GT is not an MT candidate
    ed.remove_selected();pump(root)
    assert core.counts(core.scene(s,'mt-one'))['expected']==0


def test_undo_redo_preserves_original_and_flags(gui):
    root,h,ed,s=gui
    before=copy.deepcopy(core.scene(s,"mt-one")["parents"])
    x,y=ed.to_canvas((225,175));h.canvas.event_generate('<Motion>',x=round(x),y=round(y),warp=True);pump(root)
    select(h,ed,(225,175));click(h,ed,(225,175),3)
    assert core._object(core.scene(s,"mt-one"),"GT:a0")["error"]
    ed.undo();pump(root)
    assert not core._object(core.scene(s,"mt-one"),"GT:a0")["error"]
    ed.undo(redo=True);pump(root)
    assert core._object(core.scene(s,"mt-one"),"GT:a0")["error"]
    assert core.scene(s,"mt-one")["parents"]==before


def test_mismatched_bitmap_size_is_readonly(gui):
    root,h,ed,s=gui
    before=s.review.to_dict()
    h._image_source_size=(999,700)
    assert not ed.available()
    key(h,"d")
    assert ed.mode!="draw" and s.review.to_dict()==before


def test_s_right_click_flags_directly_without_prior_left_click(gui):
    root,h,ed,s=gui
    ed.selected=""
    x,y=ed.to_canvas((225,175));h.canvas.event_generate('<Motion>',x=round(x),y=round(y),warp=True);pump(root)
    select(h,ed,(225,175));click(h,ed,(225,175),3)
    assert core._object(core.scene(s,"mt-one"),"GT:a0")["error"]
    assert ed.selected=='GT:a0' and h.canvas.find_withtag('fp_diagonal')
    ed._choose_plate('');pump(root)  # no prior selection is needed to reverse the flag
    click(h,ed,(225,175),3)
    assert not core._object(core.scene(s,"mt-one"),"GT:a0")["error"]
