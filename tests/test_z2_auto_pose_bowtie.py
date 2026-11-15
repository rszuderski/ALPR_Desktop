"""Stubbed Auto/Combined outputs: raw model evidence is never the display hull."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from auto_annotation_tool.annotators.plate_annotator import PlateAnnotator
from auto_annotation_tool.annotators.combined_annotator import CombinedAnnotator
from auto_annotation_tool.data_models import Detection
from auto_annotation_tool.quality_metrics import compute_plate_polygon_fit_metrics

RAW = [[50., 60.], [90., 60.], [100., 80.], [110., 190.]]
BOX = [40., 50., 120., 200.]


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
    def cpu(self):
        return self
    def numpy(self):
        return self.values


def pose_model(quads):
    names = {0: "plate"}
    boxes = [[BOX[0] + 220*i, BOX[1], BOX[2] + 220*i, BOX[3]] for i in range(len(quads))]
    result = SimpleNamespace(names=names, boxes=SimpleNamespace(
        xyxy=Tensor(boxes), conf=Tensor([.91] * len(quads)), cls=Tensor([0] * len(quads))),
        keypoints=SimpleNamespace(data=Tensor(quads)))
    model = Mock(return_value=[result])
    model.names = names
    return model


def run(kind, quads, offset=(0, 0)):
    model = pose_model(quads)
    image = np.zeros((700, 1100, 3), dtype=np.uint8)
    if kind == "plate":
        annotator = PlateAnnotator(Path("unused.pt"), device="cpu")
        annotator.model = model
        annotator.is_pose_model = True
        annotator._read_image_for_yolo = lambda _: image
        detections = annotator.process_image(Path("fixture.jpg")).detections
    else:
        annotator = CombinedAnnotator(Path("unused.pt"), Path("unused.pt"), device="cpu")
        annotator.plate_model = model
        annotator.is_plate_pose_model = True
        detections = annotator._detect_plates(image, 1100, 700, offset_x=offset[0], offset_y=offset[1])
    return detections, model


def is_positive_convex(points):
    crosses = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        crosses.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
    return all(x > 0 for x in crosses)


@pytest.mark.parametrize("kind", ["plate", "combined"])
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_auto_does_not_create_bowtie_or_reindex_raw(kind, dims, count):
    quads = [[[x + 220*j, y] + ([.91-i*.1] if dims == 3 else []) for i, (x, y) in enumerate(RAW)]
             for j in range(count)]
    before = deepcopy(quads)
    detections, model = run(kind, quads)
    model.assert_called_once()
    assert len(detections) == count
    for index, detection in enumerate(detections):
        assert is_positive_convex(detection.polygon)
        assert detection.polygon == [tuple(k[:2]) for k in quads[index]]
        assert detection.raw_pose_keypoints == quads[index]
        assert all(len(k) == dims for k in detection.raw_pose_keypoints)
        assert detection.bbox == tuple([BOX[0]+220*index, BOX[1], BOX[2]+220*index, BOX[3]])
        assert not detection.geometry["bbox_fallback"]
        assert detection.attributes.get("manually_edited") != "true"
    assert quads == before


@pytest.mark.parametrize("kind", ["plate", "combined"])
def test_predicted_bowtie_preserves_evidence_and_marks_display_reorder(kind):
    bowtie = [RAW[i] + [.8 - .1*i] for i in (0, 1, 3, 2)]
    detections, _ = run(kind, [bowtie])
    det = detections[0]
    assert det.raw_pose_keypoints == bowtie
    assert [tuple(p[:2]) for p in det.keypoints] == [tuple(p[:2]) for p in bowtie]
    assert is_positive_convex(det.polygon)
    assert det.geometry["reorder_permutation"] == [0, 1, 3, 2]
    assert not det.geometry["bbox_fallback"]
    assert det.attributes["fit_input_geometry_status"] == "SELF_INTERSECTING"
    assert float(det.attributes["fit_score"]) == 0.


@pytest.mark.parametrize("kind", ["plate", "combined"])
@pytest.mark.parametrize("quad", [
    [RAW[0], RAW[0], RAW[2], RAW[3]],
    [[float("nan"), 60.], *RAW[1:]],
    [[-50., 60.], *RAW[1:]],
    RAW[:3],
    [*RAW, [120., 200.]],
    [[50., 60.], [90., 60.], [65., 70.], [50., 100.]],
])
def test_invalid_pose_is_explicit_bbox_fallback(kind, quad):
    detections, _ = run(kind, [quad])
    assert len(detections) == 1
    det = detections[0]
    assert det.geometry["bbox_fallback"]
    assert is_positive_convex(det.polygon)
    assert len(det.raw_pose_keypoints) == len(quad)
    assert float(det.attributes["fit_score"]) == 0.


def test_combined_roi_translates_raw_and_display_once():
    raw = [[x, y, .92-i*.1] for i, (x, y) in enumerate(RAW)]
    annotator = CombinedAnnotator(Path("unused.pt"), Path("unused.pt"), device="cpu")
    annotator.plate_model = pose_model([raw])
    annotator.is_plate_pose_model = True
    vehicle = Detection("vehicle", .9, (200., 200., 650., 700.))
    image = np.zeros((900, 1000, 3), dtype=np.uint8)
    x1, y1, x2, y2 = annotator._get_vehicle_crop_bounds(vehicle.bbox, 1000, 900)
    detections = annotator._detect_plates_for_vehicle(image, vehicle, 1000, 900)
    assert len(detections) == 1
    det = detections[0]
    assert det.raw_pose_keypoints == [[x+x1, y+y1, conf] for x, y, conf in raw]
    assert det.polygon == [(x+x1, y+y1) for x, y in RAW]
    assert annotator.plate_model.call_args.args[0].shape == (y2-y1, x2-x1, 3)
    assert raw == [[x, y, .92-i*.1] for i, (x, y) in enumerate(RAW)]


def test_fit_reports_original_topology_before_repair():
    valid = compute_plate_polygon_fit_metrics(.9, RAW, tuple(BOX))
    invalid = compute_plate_polygon_fit_metrics(.9, [RAW[i] for i in (0, 1, 3, 2)], tuple(BOX))
    assert valid["input_geometry_status"] == "VALID"
    assert invalid["input_geometry_status"] == "SELF_INTERSECTING"
    assert not invalid["input_geometry_valid"] and invalid["display_geometry_valid"]
    assert invalid["fit_score"] == 0. < valid["fit_score"]
