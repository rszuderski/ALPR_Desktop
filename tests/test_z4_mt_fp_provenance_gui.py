"""Native events: compact panel and clean Canvas preserve the qualified pair."""
import copy
from types import SimpleNamespace
from tkinter import ttk

import pytest

import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_gui import gui, pump, click
from test_mobile_p2b2_canvas_review import text_all
from test_mobile_p2b2_fix01_canvas_ux import two_scene_session
from test_mobile_human_review import load
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups

GID = "mt-one"


def ready(root, h, ed):
    pump(root, lambda: not h._saving and not ed.busy and not ed.in_refresh)
    ed.cache_key = None; ed.sync(); ed.overlay(); pump(root)


def texts(h, tag):
    return [h.canvas.itemcget(i, "text") for i in h.canvas.find_withtag(tag) if h.canvas.type(i) == "text"]


@pytest.mark.parametrize("size", ["1649x896", "1180x760"])
def test_actual_click_fp_and_real_mt_keep_panel_hud_identity_and_gt(gui, size):
    root, h, ed, s = gui
    h.window.geometry(size + "+0+0")
    text_all(s); core.toggle_error(s, GID, "GT:a1"); core.check_scene(s, GID)
    ready(root, h, ed)
    assert ed.show_fp_var.get()  # saved FP is visible by default
    frozen = s.sidecar_path.read_bytes(); summary = core.summary(s)
    assert ed.selected == "" and ed.assessment.model["focus_id"] == ""
    assert 'Wybierz ramkę' in ed.contexts.data['object']
    for point, oid in [((445, 175), "GT:a1"), ((225, 175), "GT:a0"), ((445, 175), "GT:a1")]:
        click(h, ed, point); pump(root); ed.overlay()
        m = ed.assessment.model
        assert ed.selected == oid and m["focus_id"] == oid
        hud = texts(h, "overlap_hud_text")
        assert not hud
        compact=ed.contexts.data['object']
        if oid == "GT:a1":
            assert m["geometry_status"] == "FP" and "MT 2" in compact
            assert 'błędne wykrycie' in compact and not any(word in compact for word in ('IoU','NMCE','CER'))
            assert m["polygon_iou_pct"] is m["nmce_pct"] is None
            assert any("MT 2" in t and "Błędna" in t for t in texts(h, "clean_label_text"))
        else:
            assert m["geometry_pair"]["object_id"] == "GT:a0"
            assert "Tablica 1" in compact and 'IoU 100,0%' in compact
            assert m["polygon_iou_pct"] == 100 and m["nmce_pct"] == 0
        assert s.sidecar_path.read_bytes() == frozen and core.summary(s) == summary


def test_unconfirmed_copy_then_correction_undo_redo_and_reload(gui):
    root, h, ed, s = gui
    text_all(s); core.toggle_error(s, GID, "GT:a1"); ready(root, h, ed)
    click(h, ed, (225, 175)); pump(root); ed.overlay()
    assert ed.assessment.model["geometry_status"] == "UNCONFIRMED"
    assert "jeszcze nieocenione" in ed.contexts.data['object'] and not texts(h,'overlap_hud_text')
    before = core.scene(s, GID)
    points = copy.deepcopy(before["objects"][0]["quad"]); points[0][0] += 10
    assert ed._commit(lambda: core.correct_plate(s, GID, "GT:a0", points), "Test korekty")
    ready(root, h, ed)
    corrected = ed.assessment.model
    assert 'IoU ' in ed.contexts.data['object'] and not texts(h,'overlap_hud_text')
    assert corrected["geometry_pair"]["parent_id"] == "MT:a0"
    assert any("GT 1" in text for text in texts(h, "clean_label_text"))
    assert core.summary(s)["geometry_n"] == 0
    ed.undo(); ready(root, h, ed)
    assert ed.assessment.model["geometry_status"] == "UNCONFIRMED"
    assert core.scene(s, GID)["objects"] == before["objects"]
    ed.undo(redo=True); ready(root, h, ed)
    assert ed.assessment.model["geometry"] == corrected["geometry"]
    assert core.scene(s, GID)["parents"] == before["parents"]
    assert ed._commit(lambda: core.check_scene(s, GID), "Test potwierdzenia")
    ready(root, h, ed)
    reopened = load(s.path, s.sidecar_path.parent)
    assert core.summary(reopened) == core.summary(s)
    assert core.scene(reopened, GID)["objects"][0]["gt_text"] == before["objects"][0]["gt_text"]


def test_live_drag_panel_badge_and_hud_share_unsaved_reference(gui):
    root, h, ed, s = gui
    ed._choose_plate("GT:a0"); ready(root, h, ed)
    before = s.sidecar_path.read_bytes(); summary = core.summary(s)
    original = copy.deepcopy(core.scene(s, GID)["objects"][0]["quad"])
    x, y = ed.to_canvas(original[0])
    ed.press(SimpleNamespace(x=x, y=y))
    assert ed.drag is not None
    x, y = ed.to_canvas((original[0][0]+11, original[0][1]+3))
    ed.motion(SimpleNamespace(x=x, y=y)); ed.overlay()
    m = ed.assessment.model
    assert ed.gt_focus_id=='GT:a0' and m['geometry_status']=='UNCONFIRMED'
    assert not texts(h,'overlap_hud_text') and not texts(h,'clean_label_text') and not h.canvas.find_withtag('mt_original')
    polygon=h.canvas.find_withtag('operator_quad');assert len(polygon)==1 and len(h.canvas.find_withtag('corner_handle'))==4
    assert h.canvas.coords(polygon[0])==[v for p in ed.drag['quad'] for v in ed.to_canvas(p)]
    assert s.sidecar_path.read_bytes() == before and core.summary(s) == summary
    ed.escape(); ready(root, h, ed)
    assert s.sidecar_path.read_bytes() == before
    ed._choose_plate("GT:a0"); ed.overlay()
    assert ed.assessment.model["geometry_status"] == "UNCONFIRMED"


def test_fp_remains_unmeasured_during_corner_drag(gui):
    root, h, ed, s = gui
    core.toggle_error(s, GID, "GT:a1"); ready(root, h, ed)
    assert ed.show_fp_var.get()
    ed._choose_plate("GT:a1"); ed.overlay()
    before = s.sidecar_path.read_bytes()
    points = copy.deepcopy(core.scene(s, GID)["objects"][1]["quad"])
    x,y = ed.to_canvas(points[0]); ed.press(SimpleNamespace(x=x,y=y))
    assert ed.drag is None  # FP has no editable physical GT reference
    x,y = ed.to_canvas((points[0][0]+8,points[0][1])); ed.motion(SimpleNamespace(x=x,y=y)); ed.overlay()
    assert ed.assessment.model["geometry_status"] == "FP"
    assert ed.assessment.model["polygon_iou_pct"] is ed.assessment.model["nmce_pct"] is None
    assert 'błędne wykrycie' in ed.contexts.data['object'] and not texts(h,'overlap_hud_text')
    assert s.sidecar_path.read_bytes() == before
    ed.escape()


def test_navigation_zoom_and_tab_return_use_current_pair_without_saving(tmp_path, monkeypatch):
    s = two_scene_session(tmp_path)
    for i in range(2):
        gid, oid = f"g{i}", f"GT:t{i}"
        points = copy.deepcopy(core.scene(s, gid)["objects"][0]["quad"]); points[0][0] += 5+i*8
        core.correct_plate(s, gid, oid, points); core.set_text(s, gid, oid, f"WA{i}1234"); core.check_scene(s, gid)
    monkeypatch.setattr(native, "source_session", lambda p: s)
    gen = native.gui.__wrapped__(tmp_path, monkeypatch); root,h,ed,_ = next(gen)
    frozen = s.sidecar_path.read_bytes(); summary = core.summary(s)
    try:
        tabs = next(w for w in h.window.winfo_children() if isinstance(w, ttk.Notebook))
        for i in [1,0,1,0]:
            gid,oid = f"g{i}",f"GT:t{i}"
            iid = next(iid for iid in h.subject_tree.get_children("") if gid in subject_groups(s, h._subject_ids[iid]))
            h.subject_tree.selection_set(iid); h.subject_tree.event_generate("<<TreeviewSelect>>")
            pump(root, lambda: h.current_invocation == gid and h._image_entry == core.scene(s,gid)["image_entry"] and not h._saving)
            ed._choose_plate(oid); ready(root,h,ed)
            reference = ed.assessment.model["geometry"]
            for zoom in (1.0,1.6):
                h._view_zoom=zoom; h._paint_image(); pump(root)
                assert ed.assessment.model["geometry"] == reference
                assert ed.assessment.model["geometry_pair"]["attempt_id"] == f"t{i}"
                assert "nieustalone" not in " ".join(texts(h,"mz_badge"))
            current = tabs.select(); other = next(t for t in tabs.tabs() if t != current)
            tabs.select(other); pump(root); tabs.select(current); pump(root)
            assert ed.assessment.model["geometry"] == reference
        assert s.sidecar_path.read_bytes() == frozen and core.summary(s) == summary
    finally:
        try: next(gen)
        except StopIteration: pass
