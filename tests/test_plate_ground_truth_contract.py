import xml.etree.ElementTree as ET

from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.exporters.cvat_exporter import CVATExporter
from auto_annotation_tool.plate_ground_truth import (
    GROUND_TRUTH_SOURCE_ATTR,
    GROUND_TRUTH_SOURCE_MANUAL_Z2,
    GROUND_TRUTH_TEXT_ATTR,
    PLATE_ANNOTATION_ID_ATTR,
    ensure_plate_annotation_id,
    get_plate_ground_truth,
    set_plate_ground_truth,
)


def plate(attributes=None):
    return Detection(
        label="plate",
        confidence=0.91,
        bbox=(10.0, 20.0, 110.0, 60.0),
        polygon=[
            (10.0, 20.0),
            (110.0, 20.0),
            (110.0, 60.0),
            (10.0, 60.0),
        ],
        attributes=dict(attributes or {}),
    )


def xml_attributes(path):
    polygon = ET.parse(path).getroot().find(".//polygon[@label='plate']")
    assert polygon is not None
    return {
        str(item.get("name") or ""): str(item.text or "")
        for item in polygon.findall("attribute")
    }


def test_ground_truth_is_attached_to_concrete_plate_annotation():
    det = plate()
    annotation_id = ensure_plate_annotation_id(
        det.attributes,
        id_factory=lambda: "fixed-id",
    )
    assert annotation_id == "plate-ann-fixed-id"

    assert set_plate_ground_truth(det.attributes, " wi 905-pw ") == "WI905PW"
    assert get_plate_ground_truth(det.attributes) == "WI905PW"
    assert det.attributes[GROUND_TRUTH_SOURCE_ATTR] == GROUND_TRUTH_SOURCE_MANUAL_Z2

    assert set_plate_ground_truth(det.attributes, "") == ""
    assert GROUND_TRUTH_TEXT_ATTR not in det.attributes
    assert GROUND_TRUTH_SOURCE_ATTR not in det.attributes
    assert det.attributes[PLATE_ANNOTATION_ID_ATTR] == "plate-ann-fixed-id"


def test_cvat_export_persists_plate_gt_and_stable_annotation_id(tmp_path):
    det = plate()
    set_plate_ground_truth(det.attributes, "wi905pw")
    ann = ImageAnnotation(
        filename="dowolna_nazwa_obrazu.jpg",
        width=1280,
        height=720,
        detections=[det],
    )

    first_path = tmp_path / "first.xml"
    exporter = CVATExporter(task_name="Plate GT test")
    assert exporter.export([ann], first_path)

    first = xml_attributes(first_path)
    annotation_id = first[PLATE_ANNOTATION_ID_ATTR]
    assert annotation_id.startswith("plate-ann-")
    assert first[GROUND_TRUTH_TEXT_ATTR] == "WI905PW"
    assert first[GROUND_TRUTH_SOURCE_ATTR] == GROUND_TRUTH_SOURCE_MANUAL_Z2

    second_path = tmp_path / "second.xml"
    assert exporter.export([ann], second_path)
    second = xml_attributes(second_path)
    assert second[PLATE_ANNOTATION_ID_ATTR] == annotation_id
    assert det.attributes[PLATE_ANNOTATION_ID_ATTR] == annotation_id


def test_existing_plate_annotation_id_is_never_replaced(tmp_path):
    det = plate(
        {
            PLATE_ANNOTATION_ID_ATTR: "plate-ann-existing",
            GROUND_TRUTH_TEXT_ATTR: "ab 123",
            GROUND_TRUTH_SOURCE_ATTR: "manual_z2",
        }
    )
    ann = ImageAnnotation(
        filename="image.jpg",
        width=640,
        height=480,
        detections=[det],
    )
    path = tmp_path / "annotations.xml"
    assert CVATExporter().export([ann], path)

    attrs = xml_attributes(path)
    assert attrs[PLATE_ANNOTATION_ID_ATTR] == "plate-ann-existing"
    assert attrs[GROUND_TRUTH_TEXT_ATTR] == "AB123"
