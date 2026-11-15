"""Native Tk acceptance for AKT89, using disposable synthetic sessions only."""
import os
import sys
from pathlib import Path
from dataclasses import replace

import pytest
from PIL import ImageGrab

import test_mobile_p2b2_canvas_gui as base
from test_mobile_p2b2_canvas_review import source_session, MANUAL
from auto_annotation_tool.ranking import mobile_canvas_review as core


def capture(h, name):
    target = os.environ.get("AKT89_SCREENSHOT_DIR")
    if target:
        h.window.lift(); h.window.update()
        dest = Path(target); dest.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            ImageGrab.grab(window=h.window.winfo_id()).save(dest / name)
        else:
            x, y = h.window.winfo_rootx(), h.window.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x+h.window.winfo_width(), y+h.window.winfo_height())).save(dest / name)


@pytest.fixture
def gui(tmp_path, monkeypatch):
    yield from base.gui.__wrapped__(tmp_path, monkeypatch)


def test_native_selection_gt_edit_undo_redo_and_revision_gate(gui):
    root, h, ed, s = gui
    assert not ed.assessment.technical.opened
    ed.assessment.technical.toggle()
    h.window.geometry("1649x896+20+20"); base.pump(root)
    base.click(h, ed, (225, 175)); base.pump(root)
    assert ed.selected == "GT:a0" and ed.assessment.model["state"] == "GT_MISSING"
    ed.start_entry("GT:a0"); ed.entry_var.set("WA01234"); ed.entry_enter()
    base.pump(root)
    assert ed.assessment.model["exact"] is True and ed.assessment.model["cer"] == 0
    ed.start_entry("GT:a1"); ed.entry_var.set("WA11235"); ed.entry_enter()
    base.pump(root)
    base.click(h, ed, (445, 175)); base.pump(root)
    assert ed.assessment.model["gt"] == "GT: WA11235"
    assert ed.assessment.model["edit_distance"] == 1
    # A seeded copy is not yet a geometry reference. Explicitly accept the
    # original before testing numeric correction and undo/redo against it.
    assert ed.assessment.model["polygon_iou_pct"] is None
    assert ed.assessment.model["geometry_status"] == "UNCONFIRMED"
    ed._commit(lambda: core.check_scene(s, "mt-one"), "Potwierdzenie oryginalnych ramek")
    base.pump(root)
    original_iou = ed.assessment.model["polygon_iou_pct"]
    assert original_iou == 100 and "ASSISTED" in ed.assessment.model["reference"]
    quad = core.scene(s, "mt-one")["objects"][1]["quad"]
    changed = [list(p) for p in quad]; changed[0][0] += 10
    ed._commit(lambda: core.correct_plate(s, "mt-one", "GT:a1", changed), "Test korekty")
    base.pump(root)
    assert ed.assessment.model["polygon_iou_pct"] < original_iou
    ed.undo(); base.pump(root)
    assert ed.assessment.model["polygon_iou_pct"] == original_iou
    ed.undo(); base.pump(root)
    assert ed.assessment.model["geometry_status"] == "UNCONFIRMED"
    assert ed.assessment.model["edit_distance"] == 1
    ed.undo(); base.pump(root)
    assert ed.assessment.model["state"] == "GT_MISSING"
    ed.undo(redo=True); base.pump(root)
    assert ed.assessment.model["edit_distance"] == 1
    # Same source and revision are necessary but not sufficient.
    h.stats["canvas_review"] = dict(core.summary(s), session_id="other-session")
    ed.update_report()
    assert ed.report_host.cards["cer"].value.cget("text") == "—"
    assert "Aktualizowanie" in h.science_dashboard.canvas_caption.cget("text")
    h.stats["canvas_review"] = core.summary(s); ed.update_report()
    assert "0/0" in h.science_dashboard.canvas_caption.cget("text")
    ed.approve.invoke(); base.pump(root)
    base.pump(root, lambda: not h._saving and not ed.in_refresh, seconds=10)
    ed.refresh_results(); base.pump(root, lambda: not h._saving and not ed.in_refresh)
    assert not ed.assessment.model["draft"]
    assert ed.report_host.cards["exact"].value.cget("text") == "1/2"
    assert "1/14" in h.science_dashboard.canvas_caption.cget("text")
    base.pump(root)
    bottom = ed.contexts.winfo_rooty() + ed.contexts.winfo_height()
    assert bottom < h.window.winfo_rooty() + h.window.winfo_height()
    capture(h, "01_selection.png")
    tabs = ed.report_host.master
    tabs.select(ed.report_host); base.pump(root)
    capture(h, "02_results.png")
    tabs.select(2); base.pump(root)
    capture(h, "03_history.png")
    tabs.select(0); h.window.geometry("1180x760+20+20"); base.pump(root)
    assert ed.assessment.fields["mz"].cget("text") == "MZ: WA11234"
    assert ed.contexts.fields['object'].winfo_reqheight() <= ed.contexts.fields['object'].winfo_height()
    assert ed.contexts.winfo_rootx() + ed.contexts.winfo_width() < h.window.winfo_rootx() + h.window.winfo_width()
    ed.start_entry("GT:a1"); ed.entry_var.set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"); ed.entry_enter()
    base.pump(root)
    assert ed.assessment.fields["gt"].cget("text") == "GT: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    assert ed.clean_gt_label.cget('text') == 'Numer: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    assert h.canvas.find_withtag('gt_badge') and not ed.clean_gt_slot.winfo_ismapped()
    capture(h, "04_narrow.png")


def test_native_navigation_does_not_keep_previous_plate(tmp_path, monkeypatch):
    from test_mobile_p2b2_fix01_canvas_ux import two_scene_session
    s = two_scene_session(tmp_path)
    core.set_text(s, "g0", "GT:t0", "FIRST")
    core.set_text(s, "g1", "GT:t1", "SECOND")
    monkeypatch.setattr(base, "source_session", lambda path: s)
    gen = base.gui.__wrapped__(tmp_path, monkeypatch)
    root, h, ed, _ = next(gen)
    try:
        for gid, oid, expected in (("g0", "GT:t0", "FIRST"), ("g1", "GT:t1", "SECOND"), ("g0", "GT:t0", "FIRST")):
            from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
            iid = next(i for i, subject in h._subject_ids.items() if gid in subject_groups(s, subject))
            h.subject_tree.selection_set(iid)
            h.subject_tree.event_generate("<<TreeviewSelect>>")
            base.pump(root, lambda: h.current_invocation == gid and h._image_entry == core.scene(s, gid)["image_entry"] and not h._saving)
            ed._choose_plate(oid); base.pump(root)
            assert ed.assessment.model["gt"] == "GT: " + expected
            assert ed.assessment.model["execution"]["mt_invocation_id"] == gid
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@pytest.mark.parametrize("not_run", [False, True])
def test_native_manual_zero_detection_and_not_run(tmp_path, monkeypatch, not_run):
    s = source_session(tmp_path, n=0, mz=False)
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    monkeypatch.setattr(base, "source_session", lambda path: s)
    gen = base.gui.__wrapped__(tmp_path, monkeypatch)
    root, h, ed, _ = next(gen)
    try:
        h.window.geometry("1649x896+20+20"); base.pump(root)
        base.click(h, ed, (720, 385)); base.pump(root)
        assert ed.assessment.model["gt"] == "GT: GXW9"
        assert ed.assessment.model["state"] == "MZ_NOT_RUN"
        assert ed.assessment.model["cer"] is None
        if not_run:
            g = s.mt_invocations["mt-one"]
            for row in g.records:
                row.update(mt_status="NOT_RUN", mt_executed=False)
            s.mt_invocations["mt-one"] = replace(g, executed=False)
            ed.cache_key = None; ed.sync(); base.pump(root)
            assert ed.assessment.model["execution"]["state"] == "MT_NOT_RUN"
            assert "GXW9" in ed.assessment.model["gt"]
            assert ed.assessment.model["mz"] == "MZ: nie uruchomiono"
            assert all(card.value.cget("text") == "—" for card in ed.tally_cards.values())
        else:
            assert "FN" in ed.assessment.model["outcome"] and ed.assessment.model["draft"]
            assert core.summary(s)["missed_plates"] == 0
        capture(h, "06_mt_not_run.png" if not_run else "05_mt_executed_zero.png")
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
