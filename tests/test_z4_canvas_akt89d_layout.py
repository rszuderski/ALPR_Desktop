"""AKT89D: native layout, approval transitions and preserved Canvas controls."""
import copy
import tkinter as tk
from tkinter import font as tkfont

import pytest

import test_mobile_p2b2_canvas_gui as base
from auto_annotation_tool.ranking import mobile_canvas_review as core


@pytest.fixture
def gui(tmp_path, monkeypatch):
    yield from base.gui.__wrapped__(tmp_path, monkeypatch)


def settle(root, h, ed):
    base.pump(root, lambda: not h._saving and not ed.in_refresh)


@pytest.mark.parametrize("width,height", [(1649, 896), (1180, 760)])
def test_compact_layout_has_two_header_rows_and_unobstructed_canvas(gui, width, height):
    root, h, ed, s = gui
    h.window.geometry(f"{width}x{height}+0+0")
    base.pump(root)
    left = h.subject_tree.master.master
    assert 250 <= left.winfo_width() <= 315
    assert h.canvas.winfo_height() > height * .68
    assert 0 < h.canvas.winfo_rooty()-h.canvas.master.winfo_rooty() <= 36
    assert ed.clean_bar.winfo_rooty()+ed.clean_bar.winfo_height()<=h.canvas.winfo_rooty()
    assert int(h.canvas.cget("highlightthickness")) <= 1
    assert not h.record_label.winfo_ismapped()
    assert not h.image_help.winfo_ismapped()
    assert not h.zoom_value.winfo_ismapped()
    assert not h.source_frame_button.winfo_ismapped()
    header = h.session_label.master
    assert {int(w.grid_info()["row"]) for w in header.grid_slaves()} == {0}
    actions = h.complete_button.master
    assert h.form_status_label.master is actions
    assert h.form_status_label.winfo_rootx() < h.complete_button.winfo_rootx()
    for widget in (h.complete_button, ed.contexts):
        assert widget.winfo_rootx() + widget.winfo_width() <= h.window.winfo_rootx() + width
    assert not h.progress_label.winfo_ismapped() and not h.form_status_label.winfo_ismapped()
    assert h.subject_tree.cget("displaycolumns") == ""
    assert h.subject_tree.xview() == (0.0, 1.0)


def test_approval_is_explicit_and_edit_revokes_green_status(gui, monkeypatch):
    root, h, ed, s = gui
    assert h.form_status_label.cget("text") == "W TOKU"
    for i in range(2):
        ed.start_entry(f"GT:a{i}")
        ed.entry_var.set(f"WA{i}1234")
        ed.entry_enter()
    ed.approve.invoke()
    base.pump(root)
    assert "Przejrzane: 1/1" in h.progress_label.cget("text")
    assert 'Decyzje do podjęcia: 0' in h.progress_label.cget('text')
    assert s.review.review_status != "COMPLETED"
    assert h.form_status_label.cget("text") == "GOTOWA DO ZATWIERDZENIA"
    assert h.form_status_label.cget("fg") == ed.p["amber"]
    # Exercise the unchanged approval action, including the user's decision.
    ed.refresh_results()
    settle(root, h, ed)
    monkeypatch.setattr(base.module.messagebox, "askyesno", lambda *a, **kw: False)
    h.complete_button.invoke()
    base.pump(root)
    assert s.review.review_status != "COMPLETED"
    monkeypatch.setattr(base.module.messagebox, "askyesno", lambda *a, **kw: True)
    h.complete_button.invoke()
    base.pump(root, lambda: s.review.review_status == "COMPLETED" and not h._saving)
    assert h.form_status_label.cget("text") == "WERYFIKACJA ZATWIERDZONA"
    assert h.form_status_label.cget("fg") == ed.p["green"]
    assert "zatwierdz" not in h.status_var.get().lower()
    revision = s.review.review_revision
    ed.start_entry("GT:a0")
    ed.entry_var.set("WA01235")
    ed.entry_enter()
    base.pump(root)
    assert s.review.review_revision > revision
    assert s.review.review_status == "IN_PROGRESS"
    assert h.form_status_label.cget("fg") != ed.p["green"]
    assert h.form_status_label.cget("text") == "W TOKU"
    # Existing data semantics require rechecking an edited image.
    assert "Przejrzane: 0/1" in h.progress_label.cget("text")
    assert not core.scene(s, "mt-one")["checked"]
    ed.undo()
    base.pump(root)
    assert core.scene(s, "mt-one")["objects"][0]["gt_text"] == "WA01234"
    assert s.review.review_status == "IN_PROGRESS"
    assert h.form_status_label.cget("fg") != ed.p["green"]
    assert h.form_status_label.cget("text") == "GOTOWA DO ZATWIERDZENIA"
    assert core.scene(s,'mt-one')['checked']  # persistent Undo restores the image, never session approval
    ed.undo(redo=True)
    base.pump(root)
    assert core.scene(s, "mt-one")["objects"][0]["gt_text"] == "WA01235"
    assert h.form_status_label.cget("fg") != ed.p["green"]
    ed.approve.invoke()
    base.pump(root)
    assert h.form_status_label.cget("text") == "GOTOWA DO ZATWIERDZENIA"
    ed.approve.invoke()
    base.pump(root)
    assert h.form_status_label.cget("text") == "W TOKU"


def test_progress_does_not_certify_an_incomplete_form(gui):
    root, h, ed, s = gui
    inv = ed.inventory()
    frozen = s.review.to_dict()
    for checked in (16, 19):
        ed.update_header(dict(inv, inspected=checked, total=19, remaining=19-checked))
        assert f"Przejrzane: {checked}/19" in h.progress_label.cget("text")
        assert f"Decyzje do podjęcia: {inv['decisions']}" in h.progress_label.cget("text")
        assert f"Braki dowodów modelu: {inv['evidence_gaps']}" in h.progress_label.cget("text")
        assert h.form_status_label.cget("text") == "W TOKU"
    assert s.review.to_dict() == frozen


def test_hidden_toolbar_preserves_zoom_source_view_and_gt(gui, monkeypatch):
    root, h, ed, s = gui
    h.window.geometry("1180x760+0+0")
    base.pump(root)
    ed.start_entry("GT:a0")
    ed.entry_var.set("WA01234")
    ed.entry_enter()
    base.pump(root)
    frozen = s.sidecar_path.read_bytes()
    parents = copy.deepcopy(core.scene(s, "mt-one")["parents"])
    assert h._is_source_frame_view()
    base.key(h, "v")
    base.pump(root, lambda: not h._is_source_frame_view() and not h._saving)
    assert not h.source_frame_button.winfo_ismapped()
    base.key(h, "v")
    base.pump(root, lambda: h._is_source_frame_view() and not h._saving)
    assert not h.source_frame_button.winfo_ismapped()
    scale = h._view_zoom
    # Wheel routing deliberately follows the screen pointer, not event.x/y.
    monkeypatch.setattr(h.window, "winfo_pointerxy", lambda:
                        (h.canvas.winfo_rootx()+100, h.canvas.winfo_rooty()+100))
    monkeypatch.setattr(h.canvas, "winfo_pointerx", lambda: h.canvas.winfo_rootx()+100)
    monkeypatch.setattr(h.canvas, "winfo_pointery", lambda: h.canvas.winfo_rooty()+100)
    h.canvas.event_generate("<MouseWheel>", delta=120, x=100, y=100)
    base.pump(root)
    assert h._view_zoom > scale
    base.key(h, "f")
    assert h._view_zoom == 1.0
    ed._choose_plate("GT:a0")
    # Existing A binding still points at the Canvas fit controller.
    x, y = ed.to_canvas((225, 175))
    base.key(h, "a", xy=(x, y))
    assert ed.plate_fit_active
    base.key(h, "a", xy=(x, y))
    assert ed.plate_fit_active
    base.key(h, "f")
    assert not ed.plate_fit_active and h._view_zoom==1
    base.key(h, "t", xy=(100, 100))
    assert not ed.show_roi_overlay
    base.key(h, "t", xy=(100, 100))
    assert ed.show_roi_overlay
    base.pump(root,lambda:not h._saving and not ed.in_refresh and not ed.busy)
    h.canvas.event_generate('<Motion>',x=100,y=100,warp=True);base.pump(root)
    base.key(h, "n", xy=(100, 100))
    assert ed.selected == "GT:a1"
    assert s.sidecar_path.read_bytes() == frozen
    assert core.scene(s, "mt-one")["parents"] == parents
    assert ed.clean_menu.entrycget(1,'label')=='Skróty i obsługa'
    assert not ed.help_section.body.winfo_ismapped() and not h.review_sidebar_scroll.winfo_ismapped()


def test_narrow_image_list_keeps_next_pending_navigation(tmp_path, monkeypatch):
    from test_mobile_p2b2_fix01_canvas_ux import two_scene_session
    s = two_scene_session(tmp_path)
    core.set_text(s, "g0", "GT:t0", "FIRST")
    core.check_scene(s, "g0")
    monkeypatch.setattr(base, "source_session", lambda path: s)
    gen = base.gui.__wrapped__(tmp_path, monkeypatch)
    root, h, ed, _ = next(gen)
    try:
        h.window.geometry("1180x760+0+0")
        base.pump(root)
        frozen = s.sidecar_path.read_bytes()
        assert "Przejrzane: 1/2" in h.progress_label.cget("text")
        button = next(w for w in h.subject_tree.master.master.winfo_children()
                      if isinstance(w, tk.Button))
        assert button.cget("text") == "Następna do weryfikacji"
        button.invoke()
        base.pump(root, lambda: h.current_invocation == "g1" and not h._saving)
        assert h.subject_tree.selection()
        assert s.sidecar_path.read_bytes() == frozen
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
