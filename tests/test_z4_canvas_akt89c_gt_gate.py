"""AKT89C: per-plate GT UI gate, persistent text and truthful geometry provenance.

Historical GT fixtures deliberately use the unchanged core import/write API;
the user-facing editor must never offer that operation for an unlinked plate.
"""
import copy
from dataclasses import replace
import os
from pathlib import Path

import pytest
from PIL import ImageGrab

import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import source_session, MANUAL, text_all
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_presentation import (
    execution_state, plate_review_state, image_frame_status,
)


def state(s, oid="GT:a0", scene=None):
    scene = scene or core.scene(s, "mt-one")
    return plate_review_state(scene, core._object(scene, oid), execution_state(s, "mt-one"))


@pytest.mark.parametrize("mz,no_read", [(True, False), (True, True), (False, False)])
def test_final_detection_allows_text_independently_of_mz_run_state(tmp_path, mz, no_read):
    s = source_session(tmp_path, n=1, mz=mz, no_read=no_read)
    assert state(s)["gt_editable"]
    assert state(s)["frame_state"] == "MT_ORIGINAL"
    core.set_text(s, "mt-one", "GT:a0", "WA01234")
    core.check_scene(s, "mt-one")
    d = core.summary(s)
    assert d["evaluated_reads"] == int(mz)
    assert d["cer"] == (1 if no_read else 0) if mz else d["cer"] is None


def test_one_detected_one_manual_gate_is_per_plate(tmp_path):
    s = source_session(tmp_path, n=1)
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    text_all(s)
    before = s.sidecar_path.read_bytes()
    assert state(s)["gt_editable"]
    manual = state(s, oid)
    assert not manual["gt_editable"] and manual["frame_state"] == "MANUAL"
    assert "tylko do odczytu" in manual["gt_note"]
    assert s.sidecar_path.read_bytes() == before
    core.check_scene(s, "mt-one")
    d = core.summary(s)
    assert (d["expected_plates"], d["linked_plates"], d["missed_plates"]) == (2, 1, 1)
    assert d["geometry_n"] == d["evaluated_reads"] == 1 and d["gt_characters"] == 7


@pytest.mark.parametrize("translation", [False, True])
def test_current_geometry_and_recorded_history_are_separate(tmp_path, translation):
    s = source_session(tmp_path, n=1)
    original = copy.deepcopy(core.scene(s, "mt-one")["parents"][0]["quad"])
    assert state(s)["frame_state"] == "MT_ORIGINAL"
    assert "nieustalona" in state(s)["correction_history"]
    # Checking is not an edit, and the legacy geometry_state cannot override coordinates.
    text_all(s); core.check_scene(s, "mt-one")
    assert state(s)["frame_state"] == "MT_ORIGINAL"
    points = copy.deepcopy(original)
    if translation:
        points = [[x+12, y+7] for x,y in points]
    else:
        points[0][0] += 12
    core.correct_plate(s, "mt-one", "GT:a0", points)
    assert state(s)["frame_state"] == "MT_CORRECTED"
    assert "zapisano" in state(s)["correction_history"]
    core.remove_operator_geometry(s, "mt-one", "GT:a0")
    assert state(s)["frame_state"] == "MT_ORIGINAL"
    assert "obecnie oryginał" in state(s)["correction_history"]
    sc = core.scene(s, "mt-one")
    sc["objects"][0]["geometry_state"] = "operator_corrected"
    assert state(s, scene=sc)["frame_state"] == "MT_ORIGINAL"
    sc["events"] = []
    assert "nieustalona" in state(s, scene=sc)["correction_history"]
    assert sc["parents"][0]["quad"] == original


def test_image_status_counts_current_objects_without_raw_or_deleted_objects(tmp_path, monkeypatch):
    from auto_annotation_tool.ranking import mobile_mt_postdedup
    s = source_session(tmp_path)
    monkeypatch.setattr(mobile_mt_postdedup, "post_nms_proof", lambda *a: dict(
        raw=2, after_dedup=1, suppressed=1, primary_id="a0", suppressed_id="a1"))
    oid = core.add_plate(s, "mt-one", MANUAL)
    assert not state(s, "GT:a1")["gt_editable"]
    assert state(s, "GT:a1")["frame_state"] == "RAW_SUPPRESSED"
    image = image_frame_status(core.scene(s, "mt-one"), execution_state(s, "mt-one"))
    assert image == "Obrysy: 1 MT bez korekty · 1 ręczna"
    core.remove_operator_geometry(s, "mt-one", oid)
    assert image_frame_status(core.scene(s, "mt-one"), execution_state(s, "mt-one")) == "Obrysy: 1 MT bez korekty"


@pytest.mark.parametrize("flag", ["executed", "cancelled", "execution_failed"])
def test_incomplete_execution_never_allows_gt_or_creates_fn(tmp_path, flag):
    s = source_session(tmp_path, n=0, mz=False)
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    sc = core.scene(s, "mt-one")
    g = s.mt_invocations["mt-one"]
    if flag == "executed":
        for row in g.records:
            row.update(mt_status="NOT_RUN", mt_executed=False)
    s.mt_invocations["mt-one"] = replace(g, **{flag: flag != "executed"})
    p = state(s, oid, sc)
    assert not p["gt_editable"] and "bez FN" in p["gt_reason"]
    assert "nie wykrył" not in p["frame"]
    assert core.summary(s)["missed_plates"] == 0


@pytest.fixture
def manual_gui(tmp_path, monkeypatch, request):
    s = source_session(tmp_path, n=0, mz=False)
    oid = core.add_plate(s, "mt-one", MANUAL)
    if getattr(request, "param", True):
        core.set_text(s, "mt-one", oid, "GXW9")
    monkeypatch.setattr(native, "source_session", lambda path: s)
    yield from native.gui.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def detected_gui(tmp_path, monkeypatch):
    s = source_session(tmp_path, n=1)
    core.set_text(s, "mt-one", "GT:a0", "WA01234")
    monkeypatch.setattr(native, "source_session", lambda path: s)
    yield from native.gui.__wrapped__(tmp_path, monkeypatch)


def canvas_gt(h, oid):
    ed=h._canvas_editor
    if ed.entry is not None and ed.entry_target[1]==oid:return ['Numer: '+ed.entry_var.get()]
    return [ed.clean_gt_label.cget('text')] if ed.selected==oid else []


def capture(h, name):
    dest = os.environ.get("AKT89C_SCREENSHOT_DIR")
    if dest:
        h.window.update()
        Path(dest).mkdir(parents=True, exist_ok=True)
        ImageGrab.grab(window=h.window.winfo_id()).save(Path(dest)/name)


def badge_click(h, ed, oid):
    ed._choose_plate(oid);h.window.update()
    native.focus_canvas(h)
    box=ed.badges.get(oid) or ed.stack_badges[oid][:4]
    l,t,r,b=box
    h.canvas.event_generate('<ButtonPress-1>',x=round((l+r)/2),y=round((t+b)/2))
    h.canvas.event_generate('<ButtonRelease-1>',x=round((l+r)/2),y=round((t+b)/2))
    h.window.update()


@pytest.mark.parametrize("manual_gui", [True, False], indirect=True)
def test_manual_gt_blocked_from_badge_direct_and_host_actions(manual_gui, monkeypatch):
    root,h,ed,s = manual_gui
    obj = core.scene(s, "mt-one")["objects"][0]; oid=obj["object_id"]
    before = s.sidecar_path.read_bytes()
    ed._choose_plate(oid);native.pump(root)
    assert canvas_gt(h, oid) == ["Numer: " + (obj["gt_text"] or "nie dotyczy")]
    for _ in range(2):  # single/double-click dispatch reaches the same guarded action
        badge_click(h, ed, oid)
        assert ed.entry is None
    ed.start_entry(oid); assert ed.entry is None
    h.save_gt(explicit=True); ed.autosave_entry(); ed.entry_enter(); ed.entry_typed()
    monkeypatch.setattr("auto_annotation_tool.gui.mobile_canvas_editor.simpledialog.askstring",
                        lambda *a,**kw: pytest.fail("Manual GT must not open an exclusion editor"))
    ed.unreadable()
    ed.sync(); h.zoom_image(1.5); native.pump(root); h.fit_image(); native.pump(root)
    assert obj["gt_text"] in canvas_gt(h, oid)[0]
    assert s.sidecar_path.read_bytes() == before
    assert not ed.assessment.model["gt_editable"]
    # Geometry and scene confirmation remain available even with no textual GT.
    points = copy.deepcopy(obj["quad"]); points[0][0] += 5
    ed._commit(lambda: core.correct_plate(s, "mt-one", oid, points), "Test obrysu")
    ed.approve.invoke(); native.pump(root)
    d = core.summary(s)
    assert d["missed_plates"] == d["expected_plates"] == 1
    assert d["geometry_n"] == d["evaluated_reads"] == d["gt_characters"] == 0
    assert d["cer"] is None
    assert core.scene(s, "mt-one")["objects"][0]["gt_text"] == obj["gt_text"]


def test_native_multiplate_gate_does_not_disable_detected_plate(detected_gui):
    root,h,ed,s = detected_gui
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    ed.cache_key=None; ed.sync(); ed.request_overlay(); native.pump(root)
    badge_click(h, ed, oid); assert ed.entry is None
    badge_click(h, ed, "GT:a0"); assert ed.entry is not None
    ed.entry_var.set("WA01235"); ed.entry_typed(); native.pump(root)
    assert canvas_gt(h, "GT:a0") == ["Numer: WA01235"]
    ed.entry_enter(); native.pump(root)
    assert core._object(core.scene(s, "mt-one"), oid)["gt_text"] == "GXW9"
    assert core._object(core.scene(s, "mt-one"), "GT:a0")["gt_text"] == "WA01235"
    assert ed.assessment.model["image_frames"] == "Obrysy: 1 MT bez korekty · 1 ręczna"


@pytest.mark.parametrize("change", ["cancelled", "execution_failed", "navigation", "raw"])
def test_open_editor_is_cancelled_without_saving_when_gate_changes(detected_gui, change):
    root,h,ed,s = detected_gui
    before=s.sidecar_path.read_bytes(); g=s.mt_invocations["mt-one"]
    ed.start_entry("GT:a0"); ed.entry_var.set("NEVER_SAVE"); ed.entry_typed()
    original_current=h.current_invocation
    saved=None
    if change in {"cancelled", "execution_failed"}:
        s.mt_invocations["mt-one"] = replace(g, **{change:True})
    elif change == "navigation":
        h.current_invocation = "another-scene"
    else:
        saved = s.review.invocation_annotations["mt-one"][core.KEY]
        saved["objects"][0]["post_nms_suppressed"] = True
    try:
        ed.save_entry(close=False); native.pump(root)
        assert ed.entry is None and ed.entry_after is None
        ed.autosave_entry(); h.save_gt(explicit=True)
        assert s.sidecar_path.read_bytes() == before
    finally:
        s.mt_invocations["mt-one"] = g; h.current_invocation=original_current
        if saved:
            saved["objects"][0].pop("post_nms_suppressed",None)
        ed.cache_key=None; ed.sync(); native.pump(root)
    assert core._object(core.scene(s, "mt-one"), "GT:a0")["gt_text"] == "WA01234"


def test_native_provenance_r_undo_redo_preserve_gt(detected_gui):
    root,h,ed,s = detected_gui
    h.window.geometry("1649x896+0+0"); native.pump(root)
    ed._choose_plate("GT:a0"); native.pump(root)
    assert ed.assessment.model["frame_state"] == "MT_ORIGINAL"
    capture(h, "synthetic_original.png")
    points = copy.deepcopy(core.scene(s, "mt-one")["objects"][0]["quad"])
    points[0][0] += 12
    ed._commit(lambda: core.correct_plate(s,"mt-one","GT:a0",points),"Korekta testowa")
    assert ed.assessment.model["frame_state"] == "MT_CORRECTED"
    native.pump(root); capture(h, "synthetic_corrected.png")
    ed.undo(); native.pump(root)
    assert ed.assessment.model["frame_state"] == "MT_ORIGINAL"
    assert "zapisano" in ed.assessment.model["correction_history"]
    ed.undo(redo=True); native.pump(root)
    assert ed.assessment.model["frame_state"] == "MT_CORRECTED"
    ed.remove_selected(); native.pump(root)
    assert ed.assessment.model["frame_state"] == "MT_ORIGINAL"
    capture(h, "synthetic_reset.png")
    assert canvas_gt(h,"GT:a0") == ["Numer: WA01234"]
    ed.start_entry("GT:a0"); ed.entry_var.set("UNSAVED"); ed.entry_typed(); ed.escape()
    native.pump(root)
    assert canvas_gt(h,'GT:a0')==[] and not ed.selected
    assert core._object(core.scene(s,'mt-one'),'GT:a0')['gt_text']=='WA01234'
    ed._choose_plate('GT:a0');native.pump(root)
    assert canvas_gt(h,"GT:a0") == ["Numer: WA01234"]


@pytest.mark.parametrize("size", ["1920x1043", "1180x760"])
def test_full_gt_remains_on_canvas_with_zoom_and_without_selection(detected_gui, size):
    root,h,ed,s = detected_gui
    h.window.geometry(size+"+0+0"); native.pump(root)
    for zoom in (1.0, 1.7, 8.0):
        ed.h._view_zoom=zoom; ed.h._paint_image(); native.pump(root)
        ed._choose_plate(""); native.pump(root)
        h._view_center=(225,175);h._paint_image();native.pump(root)
        labels=[h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('GT:a0') if h.canvas.type(i)=='text']
        assert any('WA01234' in label for label in labels) and h.canvas.find_withtag('gt_badge')
        assert core._object(core.scene(s,'mt-one'),'GT:a0')['gt_text']=='WA01234'
        ed._choose_plate('GT:a0');native.pump(root)
        assert canvas_gt(h,"GT:a0") == ["Numer: WA01234"]
        assert not ed.clean_gt_slot.winfo_ismapped()  # the editable number is beside the plate
    h.fit_image(); native.pump(root)
    protected=ed._overlay_plate_bounds(core.scene(s,"mt-one"))
    for tag in ("clean_label_text",):
        for i in h.canvas.find_withtag(tag):
            if h.canvas.type(i)=="text":
                x,y,r,b=h.canvas.bbox(i)
                assert all(not(x<qr and r>ql and y<qb and b>qt) for ql,qt,qr,qb in protected)
