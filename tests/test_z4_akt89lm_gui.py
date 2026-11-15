"""Native Tk operator decisions on isolated synthetic sessions, with undo/redo."""
import copy

import pytest

import test_mobile_p2b2_canvas_gui as base
from test_mobile_p2b2_canvas_gui import gui, pump
from test_mobile_p2b2_canvas_review import MANUAL, text_all
from test_mobile_human_review import load
from test_z4_akt89lm_resolution import invalid_source, metrics, GID
from auto_annotation_tool.ranking import mobile_canvas_review as core
import tkinter.messagebox as messagebox


def ready(root, h, ed):
    pump(root, lambda: not h._saving and not ed.busy and not ed.in_refresh)
    ed.cache_key = None
    ed.sync()


def test_inline_selection_is_read_only_and_link_undo_redo_preserve_gt(gui, monkeypatch):
    root, h, ed, s = gui
    text_all(s)
    manual = core.add_plate(s, GID, MANUAL)
    core.set_text(s, GID, manual, "HISTORY")
    ready(root, h, ed)
    before = copy.deepcopy(core.scene(s, GID)); revision = s.review.review_revision
    ed._choose_plate(manual); d = ed.open_resolution(); pump(root)
    assert s.review.review_revision == revision and str(d.same.cget("state")) == "disabled"
    d.pick.current(0);d.pick.event_generate("<<ComboboxSelected>>")
    assert str(d.same.cget("state")) == "normal" and s.review.review_revision == revision
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: pytest.fail("No routine confirmation modal"))
    d.same.invoke(); ready(root, h, ed)
    linked = core.scene(s, GID)
    assert len(linked["objects"]) == 2 and not linked["checked"]
    assert linked["objects"][1] == before["objects"][1]
    ed.undo(); ready(root, h, ed)
    assert core.scene(s, GID)["objects"] == before["objects"]
    ed.undo(redo=True); ready(root, h, ed)
    assert core.scene(s, GID)["objects"] == linked["objects"]
    assert "Wynik wstępny" in ed.assessment.fields["draft"].cget("text")
    assert ed._commit(lambda: core.check_scene(s, GID), "Test")
    ready(root, h, ed)
    assert "Wynik uwzględniony" in ed.assessment.fields["draft"].cget("text")
    assert metrics(s)["geometry_n"] == 2 and metrics(s)["missed_plates"] == 0
    assert core.scene(load(s.bundle.path, s.sidecar_path.parent), GID) == core.scene(s, GID)


@pytest.fixture
def invalid_gui(tmp_path, monkeypatch, request):
    monkeypatch.setattr(base, "source_session", invalid_source)
    return request.getfixturevalue("gui")


def test_invalid_mt_selectable_inline_fp_and_fn_undo_redo(invalid_gui, monkeypatch):
    root, h, ed, s = invalid_gui
    manual = core.add_plate(s, GID, MANUAL)
    core.set_scene_review_state(s, GID, checked=True)
    ready(root, h, ed)
    before = core.scene(s, GID)
    ed._choose_plate("GT:a0");d = ed.open_resolution(); pump(root)
    assert d.obj["quad"] is None and str(d.same.cget("state")) == "normal"
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: pytest.fail("No routine confirmation modal"))
    d.fp.invoke(); ready(root, h, ed)
    assert core.scene(s, GID)["objects"][0]["error"]
    ed.undo(); ready(root, h, ed)
    assert core.scene(s, GID)["objects"] == before["objects"]
    ed.undo(redo=True); ready(root, h, ed)
    assert core.scene(s, GID)["objects"][0]["error"]
    assert ed._commit(lambda: core.check_scene(s, GID), "Test")
    ready(root, h, ed)
    assert metrics(s)["false_positives"] == metrics(s)["missed_plates"] == 1
    assert metrics(s)["geometry_n"] == metrics(s)["gt_characters"] == 0
    assert core.scene(s, GID)["objects"][1]["object_id"] == manual


def test_invalid_mt_same_plate_is_inline_and_reversible(invalid_gui, monkeypatch):
    root, h, ed, s = invalid_gui
    manual = core.add_plate(s, GID, MANUAL)
    core.set_scene_review_state(s, GID, checked=True)
    ready(root, h, ed)
    before = core.scene(s, GID)
    ed._choose_plate(manual);d = ed.open_resolution()
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: pytest.fail("No routine confirmation modal"))
    d.same.invoke(); ready(root, h, ed)
    linked = core.scene(s, GID)
    assert len(linked["objects"]) == 1 and linked["parents"] == before["parents"]
    assert not linked["exclusion_reason"] and linked["checked"]
    assert ed.assessment.model["geometry_status"] == "NOT_AVAILABLE"
    assert metrics(s)["geometry_n"] == metrics(s)["gt_characters"] == 0
    ed.undo(); ready(root, h, ed)
    undone = core.scene(s, GID)
    assert undone["objects"] == before["objects"] and not undone["exclusion_reason"]
    assert undone['checked']==before['checked'] and core.group_issues(s,GID)
    ed.undo(redo=True); ready(root, h, ed)
    # Persistent shared-GT history restores image inspection, not session approval.
    assert core.scene(s,GID)['checked']==linked['checked'] and s.review.review_status=='IN_PROGRESS'
    assert core.scene(s, GID)["objects"] == linked["objects"]


def test_inline_rejects_stale_revision_and_manual_remove_can_be_undone(gui, monkeypatch):
    root, h, ed, s = gui
    manual = core.add_plate(s, GID, MANUAL)
    ready(root, h, ed)
    ed._choose_plate(manual);d = ed.open_resolution()
    d.pick.current(0);d.pick.event_generate("<<ComboboxSelected>>")
    core.set_text(s, GID, "GT:a0", "NEWGT")
    revision = s.review.review_revision
    d.same.invoke()
    assert s.review.review_revision == revision
    ready(root, h, ed)
    before = core.scene(s, GID)
    ed._choose_plate(manual);ed.remove_selected();ready(root,h,ed)
    assert not any(o["object_id"] == manual for o in core.scene(s, GID)["objects"])
    ed.undo(); ready(root, h, ed)
    assert core.scene(s, GID)["objects"] == before["objects"]


def test_overlay_uses_current_image_execution_before_panel_sync(gui):
    root, h, ed, s = gui
    before = s.review.to_dict()
    ed._canvas_execution = dict(state="MT_CANCELLED", mt_invocation_id="previous", image_entry="previous.jpg")
    ed.overlay()
    labels = [h.canvas.itemcget(item, "text") for item in h.canvas.find_withtag("clean_label_text")
              if h.canvas.type(item) == "text"]
    assert labels and all("MT " in label for label in labels)
    assert not any("nieustalone" in label for label in labels)
    assert s.review.to_dict() == before
