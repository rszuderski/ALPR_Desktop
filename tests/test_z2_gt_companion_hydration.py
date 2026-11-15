from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from auto_annotation_tool.gt_pack import ALPRGTPack
from auto_annotation_tool.gui import z2_gt_pack_runtime as runtime
from auto_annotation_tool.plate_ground_truth import (
    PLATE_ANNOTATION_ID_ATTR,
    get_plate_ground_truth,
    get_plate_layout_gt,
)


def _image(path):
    Image.new("RGB", (100, 50), (40, 70, 100)).save(path)
    return path


class Host(SimpleNamespace):
    def _resolve_preview_image_path(self, ann):
        return Path(self.image_path)

    def _detection_polygon(self, det):
        return list(det.polygon or [])

    def _get_plate_detections(self, ann):
        return [det for det in list(ann.detections or []) if det.label == "plate"]


def test_empty_z2_is_hydrated_from_pack_geometry_with_text_and_layout(tmp_path):
    image = _image(tmp_path / "source.png")
    pack_path = tmp_path / "truth.alprgt"
    pack = ALPRGTPack.create(pack_path)
    image_record = pack.add_image(image)
    pack.ensure_plate(
        image_id=image_record["image_id"],
        polygon=[(10, 8), (80, 8), (80, 35), (10, 35)],
        plate_id="plate-ann-hydrated",
    )
    pack.set_ground_truth("plate-ann-hydrated", "ABC123")
    pack.set_plate_layout_gt("plate-ann-hydrated", "two_row")

    host = Host(
        image_path=str(image),
        _z2_gt_mount_config_loaded=True,
        _z2_gt_mount_context_key=runtime._mount_context_key(),
        _z2_gt_pack_source_paths=[],
        _z2_gt_working_pack_path="",
    )
    runtime.set_gt_resource_binding(
        host,
        source_paths=[pack_path],
        resource_key="free:test",
    )
    ann = SimpleNamespace(filename="source.png", detections=[])

    first = runtime.hydrate_annotations_from_gt_pack(host, ann)
    assert first["created"] == 1
    assert first["changed"] is True
    assert len(ann.detections) == 1
    det = ann.detections[0]
    assert det.attributes[PLATE_ANNOTATION_ID_ATTR] == "plate-ann-hydrated"
    assert get_plate_ground_truth(det.attributes) == "ABC123"
    assert get_plate_layout_gt(det.attributes) == "two_row"

    second = runtime.hydrate_annotations_from_gt_pack(host, ann)
    assert second["created"] == 0
    assert len(ann.detections) == 1
