"""AKT89B: read-only focus, non-obscuring overlays and separate history."""
import copy
from dataclasses import replace

import pytest

import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import source_session, MANUAL
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_presentation import execution_state, focused_assessment
from auto_annotation_tool.gui.mobile_canvas_editor import overlay_label_position


def focus(session, selected="", scene=None, **kwargs):
    return focused_assessment(scene or core.scene(session, "mt-one"), selected,
                              execution_state(session, "mt-one"), **kwargs)


def test_single_pair_has_auto_preview_without_writes(tmp_path):
    s = source_session(tmp_path, n=1)
    core.set_text(s, "mt-one", "GT:a0", "WA01234")
    core.check_scene(s, "mt-one")
    before = s.sidecar_path.read_bytes()
    model = focus(s)
    assert model["focus_mode"] == "AUTO" and model["focus_id"] == "GT:a0"
    assert model["mz"] == "MZ: WA01234" and model["gt"] == "GT: WA01234"
    assert model["exact"] and model["edit_distance"] == 0 and model["gt_characters"] == 7
    assert model["polygon_iou_pct"] == 100 and model["nmce_pct"] == 0
    assert s.sidecar_path.read_bytes() == before


def test_manual_preview_and_confirmation_only_change_mt_denominator(tmp_path):
    s = source_session(tmp_path, n=0, mz=False)
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    before = core.summary(s)
    model = focus(s)
    assert model["focus_mode"] == "AUTO" and model["gt"] == "GT: GXW9"
    assert model["state"] == "MZ_NOT_RUN" and model["draft"]
    assert "roboczo" in model["outcome"] and "potwierdzeniu" in model["outcome"]
    assert model["cer"] is model["polygon_iou_pct"] is model["nmce_pct"] is None
    assert "Brak obrysu MT" in model["geometry"]
    core.check_scene(s, "mt-one")
    after = core.summary(s)
    assert after["expected_plates"] == after["missed_plates"] == 1
    for key in ("exact_reads", "evaluated_reads", "text_errors", "gt_characters", "cer", "geometry_n",
                "polygon_iou_mean_pct", "nmce_mean_pct"):
        assert after[key] == before[key]
    assert not focus(s)["draft"]


@pytest.mark.parametrize("change", [dict(executed=False), dict(cancelled=True), dict(execution_failed=True)])
def test_non_executed_manual_gt_never_becomes_quality_fn(tmp_path, change):
    s = source_session(tmp_path, n=0, mz=False)
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    scene = core.scene(s, "mt-one")
    g = s.mt_invocations["mt-one"]
    if "executed" in change:
        for row in g.records:
            row.update(mt_status="NOT_RUN", mt_executed=False)
    s.mt_invocations["mt-one"] = replace(g, **change)
    model = focus(s, scene=scene)
    assert model["state"] in {"MT_NOT_RUN", "MT_CANCELLED", "MT_EXECUTION_FAILED"}
    assert "nie FN" in model["outcome"] and model["gt"] == "GT: GXW9"
    assert core.summary(s)["missed_plates"] == 0


def test_multi_plate_requires_identity_and_does_not_guess_matching_text(tmp_path):
    s = source_session(tmp_path)
    core.set_text(s, "mt-one", "GT:a0", "SAME")
    core.set_text(s, "mt-one", "GT:a1", "SAME")
    model = focus(s)
    assert model["focus_mode"] == "NONE" and model["state"] == "NO_SELECTION"
    assert "wskaż tablicę" in model["geometry"] and "brak pary" not in model["geometry"]
    assert model["execution"]["detections"] == 2
    assert focus(s, "GT:a1")["mz"] == "MZ: WA11234"
    text_edit = focus(s, "GT:a1", text_edit_id="GT:a0")
    assert text_edit["focus_mode"] == "TEXT_EDIT" and text_edit["mz"] == "MZ: WA01234"
    assert focus(s, "missing")["focus_mode"] == "NONE"


def test_nms_suppression_is_not_a_second_plate_and_pending_nms_blocks_auto(tmp_path, monkeypatch):
    from auto_annotation_tool.ranking import mobile_mt_postdedup
    s = source_session(tmp_path)
    proof = dict(raw=2, after_dedup=1, suppressed=1, primary_id="a0", suppressed_id="a1")
    monkeypatch.setattr(mobile_mt_postdedup, "post_nms_proof", lambda *args: proof)
    model = focus(s)
    assert model["focus_mode"] == "AUTO" and model["focus_id"] == "GT:a0"
    scene = core.scene(s, "mt-one")
    scene["_post_nms_pending"] = "a1"
    assert focus(s, scene=scene)["focus_mode"] == "NONE"


def test_ambiguous_mapping_does_not_create_auto_focus(tmp_path):
    s = source_session(tmp_path, n=1)
    scene = core.scene(s, "mt-one")
    scene["parents"].append(copy.deepcopy(scene["parents"][0]))
    assert focus(s, scene=scene)["focus_mode"] == "NONE"
    scene = core.scene(s, "mt-one")
    scene["objects"][0]["attempt_id"] = "another-attempt"
    assert focus(s, scene=scene)["focus_mode"] == "NONE"


@pytest.mark.parametrize("bounds", [(0, 0, 200, 20), (300, 280, 500, 300), (160, 100, 350, 200)])
def test_label_placement_protects_all_plate_boxes(bounds):
    obstacles = [bounds, (0, 150, 140, 220)]
    pos = overlay_label_position(bounds, (112, 48), (500, 300), obstacles)
    assert pos is not None
    x, y = pos
    assert 0 <= x <= 500-112 and 0 <= y <= 300-48
    assert all(not (x < r and x+112 > l and y < b and y+48 > t) for l,t,r,b in obstacles)
    assert overlay_label_position((0,0,500,300), (112,48), (500,300), [(0,0,500,300)]) is None


@pytest.fixture
def single_gui(tmp_path, monkeypatch):
    s = source_session(tmp_path, n=1)
    core.set_text(s, "mt-one", "GT:a0", "WA01234")
    monkeypatch.setattr(native, "source_session", lambda path: s)
    yield from native.gui.__wrapped__(tmp_path, monkeypatch)


def assert_unselected_preview(h, ed, expected):
    assert ed.selected == "" and ed.mode == "select"
    assert not h.canvas.find_withtag("corner_handle")
    assert ed.assessment.model["focus_mode"] == "NONE"
    assert not ed.assessment.model["focus_id"]
    assert 'Wybierz ramkę' in ed.contexts.data['object']
    assert expected in [o['gt_text'] for o in core.scene(h.session,h.current_invocation)['objects']]


def test_native_auto_focus_and_gt_badge_edit_leave_geometry_unselected(single_gui):
    root, h, ed, s = single_gui
    before = s.sidecar_path.read_bytes()
    assert_unselected_preview(h, ed, "WA01234")
    ed.sync(); ed.request_overlay(); native.pump(root)
    assert s.sidecar_path.read_bytes() == before
    native.click(h,ed,(225,175));native.pump(root)
    ed.clean_gt_label.event_generate('<Button-1>')
    native.pump(root)
    assert ed.entry is not None and ed.assessment.model["focus_mode"] == "TEXT_EDIT"
    assert ed.selected == 'GT:a0' and not ed.gt_focus_id
    ed.entry_var.set("WA01235"); ed.entry_enter(); native.pump(root)
    ed._choose_plate('');native.pump(root);assert_unselected_preview(h, ed, "WA01235")
    assert ed.assessment.model["edit_distance"] is None  # no selected object
    assert focus(s, "GT:a0")["edit_distance"] == 1  # persisted result is unchanged
    ed.undo(); native.pump(root); assert_unselected_preview(h, ed, "WA01234")
    ed.undo(redo=True); native.pump(root); assert_unselected_preview(h, ed, "WA01235")


def test_native_add_remove_and_explicit_manual_selection(single_gui):
    root, h, ed, s = single_gui
    oid = ed._commit(lambda: core.add_plate(s, "mt-one", MANUAL), "Test dodania")
    native.pump(root)
    assert ed.assessment.model["focus_mode"] == "NONE"
    native.click(h, ed, (720, 385)); native.pump(root)
    assert ed.selected == oid and ed.assessment.model["focus_id"] == oid
    assert ed.assessment.model["state"] == "MZ_NOT_RUN"
    ed.remove_selected(); native.pump(root)
    assert_unselected_preview(h, ed, "WA01234")
    ed.undo(); native.pump(root)
    assert ed.assessment.model["focus_mode"] == "NONE"


@pytest.mark.parametrize("size", ["1920x1043", "1180x760"])
def test_native_overlay_text_never_covers_plate_and_numbers_wrap(single_gui, size):
    root, h, ed, s = single_gui
    h.window.geometry(size+"+0+0"); native.pump(root)
    for selected in (False, True):
        if selected:
            native.click(h, ed, (225, 175)); native.pump(root)
        protected = ed._overlay_plate_bounds(core.scene(s, "mt-one"))
        ids = set(h.canvas.find_withtag("clean_label_text"))
        text_boxes = [h.canvas.bbox(i) for i in ids if h.canvas.type(i) == "text"]
        assert text_boxes
        for x,y,r,b in text_boxes:
            assert all(not (x < qr and r > ql and y < qb and b > qt) for ql,qt,qr,qb in protected)
    ed.start_entry("GT:a0"); ed.entry_var.set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"); ed.entry_enter()
    native.pump(root)
    ed.start_entry('GT:a0');native.pump(root)
    assert ed.entry.get() == "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    assert ed.entry.place_info()['in'] is h.canvas
    assert ed.entry.winfo_rootx()+ed.entry.winfo_width() <= h.canvas.winfo_rootx()+h.canvas.winfo_width()
    assert ed.entry.winfo_reqheight() <= ed.entry.winfo_height()
    ed.entry_enter();native.pump(root)


def test_native_navigation_resolves_current_scene_without_selection(tmp_path, monkeypatch):
    from test_mobile_p2b2_fix01_canvas_ux import two_scene_session
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    s = two_scene_session(tmp_path)
    core.set_text(s, "g0", "GT:t0", "FIRST")
    core.set_text(s, "g1", "GT:t1", "SECOND")
    monkeypatch.setattr(native, "source_session", lambda path: s)
    gen = native.gui.__wrapped__(tmp_path, monkeypatch)
    root,h,ed,_ = next(gen)
    try:
        before = s.sidecar_path.read_bytes()
        for gid, expected in (("g0", "FIRST"), ("g1", "SECOND"), ("g0", "FIRST")):
            iid = next(i for i in h.subject_tree.get_children("") if gid in subject_groups(s, h._subject_ids[i]))
            h.subject_tree.selection_set(iid); h.subject_tree.event_generate("<<TreeviewSelect>>")
            native.pump(root, lambda: h.current_invocation == gid and h._image_entry == core.scene(s, gid)["image_entry"] and not h._saving)
            assert_unselected_preview(h, ed, expected)
        assert s.sidecar_path.read_bytes() == before
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


def test_history_missing_protocol_collapses_without_changing_metrics(single_gui):
    from auto_annotation_tool.gui.mobile_science_viewmodel import make_science_dashboard
    root,h,ed,s = single_gui
    panel = h.science_dashboard
    assert not panel.history_progress.opened
    original = copy.deepcopy(h._science_data)
    model = make_science_dashboard(original, source_verified=True)
    assert model["pose_pair_count"] == 0 and model["mz_reviewed_count"] == 0
    report = copy.deepcopy(original)
    report["conditions"]["variant"] = "brak danych"
    changed = make_science_dashboard(report, source_verified=True)
    assert "Wariant: nie podano w raporcie" in changed["conditions"]
    assert changed["cards"] == model["cards"] and changed["rows"] == model["rows"]
    assert h._science_data == original
    panel.history_progress.toggle(); native.pump(root)
    assert panel.history_progress.opened
