"""EXIF/GT gate regression; predictors are stubs, no checkpoint is loaded."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageOps
import pytest
import tkinter as tk
from tkinter import ttk

from auto_annotation_tool import image_orientation as orientation
from auto_annotation_tool.ranking import mt_geometry_review as gate


def asymmetric_image(tmp_path, tag=6, extension="jpg"):
    array = np.zeros((96, 160, 3), dtype=np.uint8)
    array[:48, :80] = (230, 20, 30)
    array[:48, 80:] = (30, 190, 50)
    array[48:, :80] = (10, 30, 240)
    array[48:, 80:] = (230, 180, 20)
    array[8:25, 10:22] = (255, 255, 255)
    path = tmp_path / f"zażółć_{tag}.{extension}"
    exif = Image.Exif()
    if tag is not None:
        exif[274] = tag
    Image.fromarray(array).save(path, exif=exif)
    return path


@pytest.mark.parametrize("tag,extension", [(1, "jpg"), (3, "jpg"), (6, "jpg"), (8, "jpg"), (None, "jpg"), (None, "png")])
def test_actual_pixels_and_dimensions_follow_exif_once(tmp_path, tag, extension):
    path = asymmetric_image(tmp_path, tag, extension)
    before = path.read_bytes()
    image, info = orientation.inspect_oriented_image(path)
    with Image.open(path) as source:
        expected = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    np.testing.assert_allclose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), expected, atol=3)
    assert info["raw_size"] == [160, 96]
    assert info["oriented_size"] == ([96, 160] if tag in (6, 8) else [160, 96])
    assert orientation.model_input_size(True, path) == tuple(info["oriented_size"])
    assert path.read_bytes() == before


@pytest.mark.parametrize("tag", [0, 9])
def test_invalid_exif_fails_closed(tmp_path, tag):
    with pytest.raises(ValueError, match="INVALID_EXIF_ORIENTATION"):
        orientation.inspect_oriented_image(asymmetric_image(tmp_path, tag))


def test_corrupt_image_never_becomes_valid_input(tmp_path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises((ValueError, OSError)):
        orientation.inspect_oriented_image(path)


@pytest.mark.parametrize("tag,reason", [(3, "PIXEL_MISMATCH"), (6, "DIMENSION_MISMATCH")])
def test_ignored_orientation_is_detected_even_when_size_matches(tmp_path, tag, reason):
    path = asymmetric_image(tmp_path, tag)
    def wrong_decoder(path):
        return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    with pytest.raises(ValueError, match=reason):
        orientation.inspect_oriented_image(path, decoder=wrong_decoder)


@pytest.mark.parametrize("kind", ["plate", "vehicle", "combined"])
def test_annotation_and_cvat_dimensions_match_predictor_input(tmp_path, kind):
    from auto_annotation_tool.annotators.plate_annotator import PlateAnnotator
    from auto_annotation_tool.annotators.vehicle_annotator import VehicleAnnotator
    from auto_annotation_tool.annotators.combined_annotator import CombinedAnnotator
    from auto_annotation_tool.exporters.cvat_exporter import CVATExporter
    from auto_annotation_tool.utils import get_image_size
    path = asymmetric_image(tmp_path)
    model = Mock(return_value=[])
    if kind == "combined":
        annotator = CombinedAnnotator(Path("unused.pt"), Path("unused.pt"), device="cpu")
        annotator._detect_vehicles = model
    else:
        annotator = (PlateAnnotator if kind == "plate" else VehicleAnnotator)(Path("unused.pt"), device="cpu")
        annotator.model = model
    annotation = annotator.process_image(path)
    model.assert_called_once()
    actual = model.call_args.args[0]
    assert actual.shape == (160, 96, 3)
    assert (annotation.width, annotation.height) == (96, 160)
    assert get_image_size(path) == (160, 96)  # Existing utility semantics intentionally preserved.
    root = ET.Element("annotations")
    CVATExporter()._add_image(root, 0, annotation, False)
    assert root[0].get("width") == "96" and root[0].get("height") == "160"


def test_keypoints_below_raw_height_are_not_replaced_by_bbox(tmp_path):
    from auto_annotation_tool.annotators.plate_annotator import PlateAnnotator
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
        def cpu(self):
            return self
        def numpy(self):
            return self.value
    quad = [[20, 110], [70, 108], [74, 145], [18, 148]]
    result = SimpleNamespace(names={0: "plate"}, boxes=SimpleNamespace(
        xyxy=Tensor([[15, 100, 80, 150]]), conf=Tensor([.9]), cls=Tensor([0])),
        keypoints=SimpleNamespace(data=Tensor([quad])))
    annotator = PlateAnnotator(Path("unused.pt"), device="cpu")
    annotator.model = Mock(return_value=[result])
    annotator.model.names = result.names
    annotator.is_pose_model = True
    annotation = annotator.process_image(asymmetric_image(tmp_path))
    assert len(annotation.detections) == 1
    assert annotation.detections[0].polygon == [tuple(p) for p in quad]


BOX = [10, 20, 80, 60]
QUAD = [[10, 20], [80, 20], [80, 60], [10, 60]]


@pytest.mark.parametrize("box,quad,reason", [
    (BOX, None, "GT_QUAD_MISSING"),
    (BOX, [[10, 20], [180, 20], [80, 60], [10, 60]], "GT_OUTSIDE"),
    (BOX, [QUAD[i] for i in (0, 3, 2, 1)], "GT_CORNER_ORDER_REVERSED"),
    (BOX, [QUAD[i] for i in (0, 2, 1, 3)], "GT_QUAD_INVALID"),
    (BOX, [[float("nan"), 20], *QUAD[1:]], "GT_NONFINITE"),
    ([10, 20, 10, 60], QUAD, "GT_BBOX_DEGENERATE"),
])
def test_bad_gt_is_blocked_without_mutation(box, quad, reason):
    before = repr((box, quad))
    with pytest.raises(ValueError, match=reason):
        gate.validate_gt_geometry(box, quad, 100, 100)
    assert repr((box, quad)) == before


@pytest.fixture
def review(tmp_path, monkeypatch):
    from pose478_fixture import cohort, bind_source_fixture
    source = asymmetric_image(tmp_path)
    image, info = orientation.inspect_oriented_image(source)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    scene = {"file": source.name, "sha256": sha, "domain": "DAY", "image": info,
             "plates": [{"plate_id": "fixture", "bbox": BOX, "quad": QUAD, "geometry_issue": ""}]}
    scenes = cohort(scene)
    metric_policy = gate.load_metric_policy()
    scope = gate.assess_metric_scope(scenes, metric_policy)
    report = {"schema": gate.REVIEW_SCHEMA, "status": "READY_FOR_REVIEW", "selection_sha": gate.SELECTION_SHA,
              "gt_freeze_sha": gate.FREEZE_SHA, "policy": deepcopy(gate.POLICY),
              "models": [{"label": name, "role": "plate"} for name in metric_policy["model_labels"]],
              "scenes": scenes, "counts": gate.cohort_counts(scenes), **scope,
              "review_scene_hashes": gate.required_review_scenes(scenes, scope["geometry_issues"]),
              "metric_policy": metric_policy, "metric_policy_sha256": gate.METRIC_POLICY_SHA,
              "runtime": {}, "code_sha256": {}, "source_files": {str(source.resolve()): sha},
              "repo_root": str(tmp_path), "source_scene_root": str(tmp_path), "selection_dir": str(tmp_path / "selection"),
              "freeze_dir": str(tmp_path / "freeze")}
    bind_source_fixture(report, tmp_path, monkeypatch)
    report["review_sha256"] = gate.review_fingerprint(report)
    return report


def test_gate_requires_explicit_complete_decision_and_detects_changed_source(review, tmp_path):
    output = tmp_path.parent / (tmp_path.name + "_contracts")
    with pytest.raises(ValueError, match="OPERATOR_SIGNATURE_REQUIRED"):
        gate.save_operator_decision(review, output, operator="", reviewed_scenes=[], accepted=True)
    with pytest.raises(ValueError, match="ALL_REVIEW_SCENES"):
        gate.save_operator_decision(review, output, operator="fixture", reviewed_scenes=[], accepted=True)
    assert not output.exists()
    destination, decision = gate.save_operator_decision(review, output, operator="TEST FIXTURE ONLY",
        reviewed_scenes=review["review_scene_hashes"], accepted=True)
    assert gate.require_current_approval(review, decision)
    assert gate.load_operator_decision(review, output) == (destination, decision)
    assert json.loads((destination / "status.json").read_text())["measurement_status"] == "NOT_RUN"
    changed = deepcopy(decision)
    changed["operator_signature"] = "tampered"
    with pytest.raises(ValueError, match="OPERATOR_CONTRACT_CHANGED"):
        gate.require_current_approval(review, changed)
    source = Path(next(iter(review["source_files"])))
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA_MISMATCH"):
        gate.require_current_approval(review, decision)


def test_rejection_and_failed_geometry_cannot_authorize_measurement(review, tmp_path, monkeypatch):
    output = tmp_path.parent / (tmp_path.name + "_contracts")
    review["status"] = "FAIL"
    review["scenes"][0]["plates"][0]["quad"] = list(reversed(QUAD))
    review.update(gate.assess_metric_scope(review["scenes"], review["metric_policy"]))
    review["review_scene_hashes"] = gate.required_review_scenes(review["scenes"], review["geometry_issues"])
    from pose478_fixture import bind_source_fixture
    bind_source_fixture(review, tmp_path, monkeypatch)
    review["review_sha256"] = gate.review_fingerprint(review)
    with pytest.raises(ValueError, match="PREFLIGHT_NOT_READY"):
        gate.save_operator_decision(review, output, operator="fixture", reviewed_scenes=review["review_scene_hashes"], accepted=True)
    destination, decision = gate.save_operator_decision(review, output, operator="TEST FIXTURE ONLY",
        reviewed_scenes=[], accepted=False, reason="GT reversed")
    assert gate.load_operator_decision(review, output) == (destination, decision)
    with pytest.raises(ValueError, match="PREFLIGHT_NOT_READY"):
        gate.require_current_approval(review, decision)


def test_changed_policy_and_protected_output_are_rejected(review, tmp_path):
    with pytest.raises(ValueError, match="PROTECTED_OUTPUT_DIRECTORY"):
        gate.save_operator_decision(review, tmp_path / "freeze", operator="fixture",
            reviewed_scenes=review["review_scene_hashes"], accepted=True)
    changed = deepcopy(review)
    changed["policy"]["gt_transform"] = "rotate"
    with pytest.raises(ValueError, match="PREFLIGHT_CHANGED"):
        gate.recheck_review(changed)
    changed["review_sha256"] = gate.review_fingerprint(changed)
    with pytest.raises(ValueError, match="GEOMETRY_POLICY_CHANGED"):
        gate.recheck_review(changed)


def spin(root, predicate):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        root.update()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Tk did not reach the expected state")


def test_native_mt_tab_no_auto_mz_or_approval_and_failed_gt_blocks_gate(review, tmp_path, monkeypatch):
    from auto_annotation_tool.gui import z4_eval396 as view, z4_mt_geometry as panel_module
    root = tk.Tk()
    mz_loader = Mock(side_effect=RuntimeError("MZ lazy-loading fixture"))
    monkeypatch.setattr(view, "load_eval396", mz_loader)
    monkeypatch.setattr(panel_module, "preflight_mt_geometry", lambda *a, **kw: deepcopy(review))
    monkeypatch.setattr(panel_module, "load_operator_decision", lambda *a: None)
    tab = SimpleNamespace(frame=ttk.Frame(root), rank_data_dir=tk.StringVar(value=str(tmp_path)),
                          _get_ranking_task_target=lambda: "plate")
    try:
        dialog = view.open_results(tab)
        panel = dialog.mt_geometry_panel
        root.update()
        assert dialog.evaluation_notebook.select() == str(panel)
        mz_loader.assert_not_called()
        assert panel.measure_button.instate(["disabled"])
        panel.check_button.invoke()
        spin(root, lambda: panel.scene is not None and not panel.busy)
        assert panel.status.get().startswith("GOTOWE DO ODBIORU")
        assert not panel.reviewed and not panel.confirmed.get() and not panel.decision
        assert panel.approve_button.instate(["disabled"])
        assert len(panel.detail.find_withtag("preview_overlay")) > 4
        panel.confirm_button.invoke()
        # Simulated visual confirmations for the synthetic cohort controls.
        panel.reviewed.update(panel.report["review_scene_hashes"])
        panel.operator.set("TEST FIXTURE ONLY")
        assert not panel.approve_button.instate(["disabled"])
        assert panel.measure_button.instate(["disabled"])
        panel.report["status"] = "FAIL"
        panel.report["geometry_issues"] = [{"plate_id": "fixture", "reason": "GT_CORNER_ORDER_REVERSED"}]
        panel.report["blocking_geometry_issues"] = panel.report["geometry_issues"]
        panel.scene["plates"][0]["geometry_issue"] = "GT_CORNER_ORDER_REVERSED"
        panel.update_controls()
        assert panel.approve_button.instate(["disabled"])
        assert panel.confirm_button.instate(["disabled"])
        assert panel.gate_status().startswith("FAIL / STOP")
        # Existing Z4 export entry must switch to MZ and start its reader exactly once.
        assert view.open_results(tab, export_after_load=True) is dialog
        spin(root, lambda: "FAIL" in dialog.status_variable.get())
        mz_loader.assert_called_once()
    finally:
        root.destroy()
