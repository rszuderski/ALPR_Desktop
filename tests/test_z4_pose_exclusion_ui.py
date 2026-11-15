"""Native Z4 review distinguishes a scoped Pose exception from invalid GT."""
from copy import deepcopy
from pathlib import Path
import time
import tkinter as tk

from PIL import Image
import pytest

from auto_annotation_tool.gui import z4_mt_geometry as ui


@pytest.fixture
def report():
    exclusion = {"policy_id": "EVAL396_POSE478_BBOX479_v1", "plate_id": "plate_000421",
                 "reason": "GT_CORNER_ORDER_REVERSED", "source_scene_sha256": "test-scene-sha",
                 "original_bbox": [120, 20, 180, 60],
                 "original_quad": [[120, 20], [120, 60], [180, 60], [180, 20]]}
    issue = {"plate_id": "plate_000421", "reason": exclusion["reason"], "sha256": "test-scene-sha"}
    plates = [
        {"plate_id": "plate_other", "bbox": [10, 10, 90, 50],
         "quad": [[10, 10], [90, 10], [90, 50], [10, 50]], "geometry_issue": "",
         "bbox_eligible": True, "pose_eligible": True, "ocr_eligible": True, "pose_exclusion": None},
        {"plate_id": "plate_000421", "bbox": exclusion["original_bbox"],
         "quad": exclusion["original_quad"], "geometry_issue": exclusion["reason"],
         "bbox_eligible": True, "pose_eligible": False, "ocr_eligible": False, "pose_exclusion": exclusion},
    ]
    scene = {"file": "synthetic.png", "sha256": "test-scene-sha", "plates": plates,
             "image": {"exif_orientation": 1, "raw_size": [220, 90], "oriented_size": [220, 90],
                       "decoder_comparison": {"mae": 0, "p95": 0}}}
    return {"status": "READY_FOR_REVIEW", "scenes": [scene], "geometry_issues": [issue],
            "blocking_geometry_issues": [], "metric_exclusions": [exclusion],
            "review_scene_hashes": [scene["sha256"]], "models": [{"label": name} for name in ("MT-n", "MT-DAY", "MT-NIGHT")],
            "metric_counts": {"scenes": 396, "bbox": 479, "pose": 478, "ocr": 449, "pose_exclusions": 1},
            "metric_policy": {"id": exclusion["policy_id"], "exclusions": [exclusion],
                              "authorization": {"source": "operator_message", "date": "2026-10-09",
                                                "statement": "EXPLICIT TEST FIXTURE ONLY"}},
            "metric_policy_sha256": "a" * 64}


@pytest.fixture(scope="module")
def tk_root():
    # Keep one Tcl interpreter for this module. Repeated full interpreter
    # teardown amid ImageTk finalizers can destabilize the Windows Tk runtime.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.update_idletasks()
    root.destroy()


def spin(root, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Tk review did not reach the expected state")


@pytest.fixture
def native_panel(tk_root, tmp_path, report):
    root = tk_root
    window = tk.Toplevel(root)
    window.withdraw()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(str(args[1]))
    panel = ui.GeometryReviewPanel(window, tmp_path / "selection")
    panel.pack(fill="both", expand=True)
    panel.report = deepcopy(report)
    panel.messages.put(("image", (panel.epoch, panel.report["scenes"][0], Image.new("RGB", (220, 90), "#304040"))))
    # Let the scheduled production poll consume the message. Calling poll()
    # manually would start a second timer chain in this persistent interpreter.
    spin(root, lambda: panel.scene is not None)
    try:
        yield root, panel
    finally:
        window.destroy()
        root.update()
        assert not errors, errors


def test_authorized_pose_exception_is_reviewable_without_auto_approval(native_panel):
    _root, panel = native_panel
    assert panel.gate_status().startswith("GOTOWE DO ODBIORU")
    assert not panel.confirm_button.instate(["disabled"])
    assert not panel.confirmed.get() and not panel.reviewed and panel.decision is None
    assert panel.approve_button.instate(["disabled"])
    assert panel.measure_button.instate(["disabled"])
    assert "bbox i zakres wyłączenia Pose" in panel.confirm_button.cget("text")
    assert "pozostałe narożniki" in panel.confirm_button.cget("text")
    assert "bez oceny kolejności P1–P4" in panel.confirm_button.cget("text")
    assert "479 bbox" in panel.image_info.get()
    assert "478 Pose" in panel.image_info.get()
    assert "449 OCR" in panel.image_info.get()
    assert "MT-n / MT-DAY / MT-NIGHT" in panel.policy_info.get()
    assert "EVAL396_POSE478_BBOX479_v1" in panel.policy_info.get()
    assert "2026-10-09" in panel.policy_info.get()


def test_excluded_plate_shows_raw_p1_p4_orange_bbox_stays_visible(native_panel):
    _root, panel = native_panel
    before = deepcopy(panel.report)
    panel.detail.delete("preview_overlay")
    panel.draw_gt(panel.detail)
    texts = [panel.detail.itemcget(item, "text") for item in panel.detail.find_withtag("preview_overlay")
             if panel.detail.type(item) == "text"]
    assert texts == ["TL", "TR", "BR", "BL", "P1", "P2", "P3", "P4"]
    lines = [panel.detail.itemcget(item, "fill") for item in panel.detail.find_withtag("preview_overlay")
             if panel.detail.type(item) == "line"]
    assert lines == ["#52ff87", "#ffb347"]
    boxes = [item for item in panel.detail.find_withtag("preview_overlay")
             if panel.detail.type(item) == "rectangle" and panel.detail.itemcget(item, "outline") == "#39d9ff"]
    assert len(boxes) == 2
    assert "bbox: w ocenie" in panel.plate_info.get()
    assert "Pose: wyłączone" in panel.plate_info.get()
    assert panel.report == before


def test_explicit_visual_confirmation_and_signature_remain_required(native_panel, monkeypatch):
    root, panel = native_panel
    saved = []

    def save(report, destination, **kwargs):
        saved.append((report, destination, kwargs))
        return Path(destination) / "TEST_ONLY", {"decision": "APPROVED", "reason": ""}

    monkeypatch.setattr(ui, "save_operator_decision", save)
    panel.operator.set("TEST FIXTURE ONLY")
    assert panel.approve_button.instate(["disabled"])
    panel.confirm_button.invoke()
    assert panel.reviewed == set(panel.report["review_scene_hashes"])
    assert not panel.approve_button.instate(["disabled"])
    assert not saved
    panel.approve_button.invoke()
    spin(root, lambda: not panel.busy)
    assert saved and saved[0][2]["accepted"] is True
    assert saved[0][2]["operator"] == "TEST FIXTURE ONLY"
    assert panel.measure_button.instate(["disabled"])
    assert not panel.measure_button.cget("command")


def test_failed_refresh_clears_previous_policy_display_and_review(native_panel):
    root, panel = native_panel
    panel.confirm_button.invoke()
    panel.operator.set("TEST FIXTURE ONLY")
    panel.messages.put(("error", "TEST_SOURCE_CHANGED"))
    spin(root, lambda: panel.report is None)
    assert panel.report is None and panel.scene is None and not panel.reviewed
    assert "EVAL396_POSE478_BBOX479_v1" not in panel.policy_info.get()
    assert panel.status.get() == "FAIL / STOP — TEST_SOURCE_CHANGED"
    assert panel.approve_button.instate(["disabled"])
    assert panel.measure_button.instate(["disabled"])


@pytest.mark.parametrize("variant", ["new_blocker", "bbox_excluded", "missing_exception", "wrong_plate", "legacy"])
def test_unrelated_or_unbound_problem_remains_blocking(native_panel, variant):
    _root, panel = native_panel
    plate = panel.scene["plates"][1]
    if variant == "new_blocker":
        panel.report["blocking_geometry_issues"] = [{"plate_id": "plate_other", "sha256": panel.scene["sha256"],
                                                      "reason": "GT_OUTSIDE_ORIENTED_SCENE"}]
        panel.report["status"] = "FAIL"
    elif variant == "bbox_excluded":
        plate["bbox_eligible"] = False
    elif variant == "missing_exception":
        plate["pose_exclusion"] = None
    elif variant == "wrong_plate":
        plate["pose_exclusion"]["plate_id"] = "different_plate"
    else:
        panel.report.pop("blocking_geometry_issues")
        for key in ("pose_exclusion", "pose_eligible", "bbox_eligible"):
            plate.pop(key)
    panel.update_controls()
    assert panel.confirm_button.instate(["disabled"])
    panel.confirmed.set(True)
    panel.confirm_scene()
    assert not panel.reviewed
    assert panel.approve_button.instate(["disabled"])
    assert panel.measure_button.instate(["disabled"])
    if variant in {"new_blocker", "legacy"}:
        assert panel.gate_status().startswith("FAIL / STOP")
