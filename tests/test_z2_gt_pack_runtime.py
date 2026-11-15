from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from auto_annotation_tool.data_models import Detection
from auto_annotation_tool.gt_pack import ALPRGTPack
from auto_annotation_tool.gui import z2_gt_pack_runtime as runtime
from auto_annotation_tool.plate_ground_truth import (
    GROUND_TRUTH_TEXT_ATTR,
    PLATE_ANNOTATION_ID_ATTR,
    get_plate_ground_truth,
    set_plate_ground_truth,
)


def make_image(path):
    Image.new("RGB", (80, 40), (30, 60, 90)).save(path)
    return path


def make_plate(plate_id="plate-ann-fixed", gt="", polygon=None):
    attrs = {}
    if plate_id:
        attrs[PLATE_ANNOTATION_ID_ATTR] = plate_id
    if gt:
        set_plate_ground_truth(attrs, gt)
    poly = polygon or [
        (10.0, 10.0),
        (60.0, 10.0),
        (60.0, 26.0),
        (10.0, 26.0),
    ]
    return Detection(
        label="plate",
        confidence=1.0,
        bbox=(10.0, 10.0, 60.0, 26.0),
        polygon=poly,
        attributes=attrs,
    )


class Host(SimpleNamespace):
    def _resolve_preview_image_path(self, ann):
        return Path(self.image_path)

    def _detection_polygon(self, det):
        return list(det.polygon or [])

    def _get_plate_detections(self, ann):
        return list(ann.detections or [])

    def _get_current_annotation_xml_path(self):
        return Path(self.run_dir) / "annotations.xml"


def host_for(tmp_path, image_path, pack_path):
    return Host(
        image_path=str(image_path),
        run_dir=str(tmp_path / "run"),
        _z2_gt_working_pack_path=str(pack_path),
        _z2_gt_pack_source_paths=[],
    )


def annotation(det):
    return SimpleNamespace(filename="source.png", detections=[det])


def test_sync_set_then_restore_same_plate(tmp_path):
    image = make_image(tmp_path / "source.png")
    host = host_for(tmp_path, image, tmp_path / "work.alprgt")
    det = make_plate(gt="ABC123")
    assert runtime.sync_plate_gt_after_xml_save(
        host, annotation(det), det, "ABC123"
    )["ok"]

    fresh = make_plate(gt="")
    report = runtime.restore_gt_for_annotation(host, annotation(fresh))
    assert report["restored"] == 1
    assert get_plate_ground_truth(fresh.attributes) == "ABC123"


def test_sync_clear_is_not_restored_as_old_gt(tmp_path):
    image = make_image(tmp_path / "source.png")
    host = host_for(tmp_path, image, tmp_path / "work.alprgt")
    det = make_plate(gt="ABC123")
    assert runtime.sync_plate_gt_after_xml_save(
        host, annotation(det), det, "ABC123"
    )["ok"]

    det.attributes.pop(GROUND_TRUTH_TEXT_ATTR, None)
    det.attributes.pop("ground_truth_source", None)
    assert runtime.sync_plate_gt_after_xml_save(
        host, annotation(det), det, ""
    )["ok"]

    fresh = make_plate(gt="")
    report = runtime.restore_gt_for_annotation(host, annotation(fresh))
    assert report["conflicts"] == []
    assert get_plate_ground_truth(fresh.attributes) == ""


def test_restore_never_overwrites_different_local_gt(tmp_path):
    image = make_image(tmp_path / "source.png")
    host = host_for(tmp_path, image, tmp_path / "work.alprgt")
    saved = make_plate(gt="ABC123")
    assert runtime.sync_plate_gt_after_xml_save(
        host, annotation(saved), saved, "ABC123"
    )["ok"]

    local = make_plate(gt="XYZ999")
    report = runtime.restore_gt_for_annotation(host, annotation(local))
    assert get_plate_ground_truth(local.attributes) == "XYZ999"
    assert report["restored"] == 0
    assert report["conflicts"][0]["reason"] == "local_gt_differs_from_pack"


def test_exact_geometry_can_rebind_new_local_plate_id(tmp_path):
    image = make_image(tmp_path / "source.png")
    host = host_for(tmp_path, image, tmp_path / "work.alprgt")
    saved = make_plate(plate_id="plate-ann-canonical", gt="ABC123")
    assert runtime.sync_plate_gt_after_xml_save(
        host, annotation(saved), saved, "ABC123"
    )["ok"]

    fresh = make_plate(plate_id="plate-ann-new-run", gt="")
    report = runtime.restore_gt_for_annotation(host, annotation(fresh))
    assert report["rebound_plate_ids"] == 1
    assert fresh.attributes[PLATE_ANNOTATION_ID_ATTR] == "plate-ann-canonical"
    assert get_plate_ground_truth(fresh.attributes) == "ABC123"


def test_multiple_mounted_pack_values_are_conflict(tmp_path):
    image = make_image(tmp_path / "source.png")
    source_a = tmp_path / "a.alprgt"
    source_b = tmp_path / "b.alprgt"

    for path, gt in ((source_a, "ABC123"), (source_b, "ABC124")):
        pack = ALPRGTPack.create(path)
        item = pack.add_image(image)
        pack.ensure_plate(
            image_id=item["image_id"],
            polygon=[
                (10.0, 10.0),
                (60.0, 10.0),
                (60.0, 26.0),
                (10.0, 26.0),
            ],
            plate_id="plate-ann-fixed",
        )
        pack.set_ground_truth("plate-ann-fixed", gt)

    host = Host(
        image_path=str(image),
        run_dir=str(tmp_path / "run"),
        _z2_gt_working_pack_path="",
        _z2_gt_pack_source_paths=[str(source_a), str(source_b)],
    )
    det = make_plate(gt="")
    report = runtime.restore_gt_for_annotation(host, annotation(det))
    assert report["restored"] == 0
    assert report["conflicts"]
    assert get_plate_ground_truth(det.attributes) == ""


def test_failed_pack_sync_is_queued_not_lost(tmp_path, monkeypatch):
    image = make_image(tmp_path / "source.png")
    host = host_for(tmp_path, image, tmp_path / "work.alprgt")
    det = make_plate(gt="ABC123")

    monkeypatch.setattr(
        runtime,
        "_sync_item_to_pack",
        Mock(return_value={"ok": False, "reason": "simulated_failure"}),
    )
    result = runtime.sync_plate_gt_after_xml_save(
        host, annotation(det), det, "ABC123"
    )
    assert result["ok"] is False
    assert result["queued"] is True
    assert (Path(host.run_dir) / "gt_sync_pending.json").exists()
