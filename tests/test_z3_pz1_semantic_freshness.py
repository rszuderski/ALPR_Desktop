import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from auto_annotation_tool.character_recognition.plate_generator import PlateGenerator
from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.gui.z3_extraction_sources import (
    count_xml_plate_cut_targets,
    current_extract_source_payload_with_signature,
    extract_manifest_freshness,
    extract_manifest_matches_current_source,
    prepare_plate_cut_detections_for_source,
    refresh_preview_gt_metadata_from_current_xml,
)


def write_xml(path: Path, gt: str) -> Path:
    path.write_text(
        (
            "<annotations>"
            "<image id='0' name='source.png' width='80' height='40'>"
            "<polygon label='plate' points='10,10;60,10;60,26;10,26'>"
            "<attribute name='plate_annotation_id'>plate-ann-fixed</attribute>"
            f"<attribute name='ground_truth_text'>{gt}</attribute>"
            "<attribute name='ground_truth_source'>manual_z2</attribute>"
            "</polygon>"
            "</image>"
            "</annotations>"
        ),
        encoding="utf-8",
    )
    return path


class Var:
    def __init__(self, value):
        self.value = str(value)

    def get(self):
        return self.value


class Host(SimpleNamespace):
    def _paths_equivalent(self, left, right):
        try:
            return Path(left).resolve() == Path(right).resolve()
        except Exception:
            return str(left) == str(right)


def make_host(xml_path: Path, images_dir: Path) -> Host:
    return Host(
        annotation_run_dir_var=Var(xml_path.parent),
        xml_path_var=Var(xml_path),
        images_dir_var=Var(images_dir),
    )


def plate(attributes=None):
    return Detection(
        label="plate",
        confidence=1.0,
        bbox=(10.0, 10.0, 60.0, 26.0),
        polygon=[
            (10.0, 10.0),
            (60.0, 10.0),
            (60.0, 26.0),
            (10.0, 26.0),
        ],
        attributes=dict(attributes or {}),
    )


def test_gt_only_xml_change_changes_gt_hash_not_geometry_hash(tmp_path):
    xml_path = write_xml(tmp_path / "annotations.xml", "ABC123")
    before = count_xml_plate_cut_targets(xml_path)

    write_xml(xml_path, "ABC124")
    after = count_xml_plate_cut_targets(xml_path)

    assert before["xml_plate_geometry_hash"] == after["xml_plate_geometry_hash"]
    assert before["xml_plate_gt_hash"] != after["xml_plate_gt_hash"]


def test_per_plate_semantic_hashes_are_added_before_pz1():
    det = plate(
        {
            "plate_annotation_id": "plate-ann-fixed",
            "ground_truth_text": "ABC123",
            "ground_truth_source": "manual_z2",
        }
    )
    host = SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
    )

    prepare_plate_cut_detections_for_source(
        host,
        "source.png",
        [det],
    )

    assert len(det.attributes["source_geometry_hash"]) == 64
    assert len(det.attributes["source_gt_hash"]) == 64

    geometry_hash = det.attributes["source_geometry_hash"]
    gt_hash = det.attributes["source_gt_hash"]

    det.attributes["ground_truth_text"] = "ABC124"
    prepare_plate_cut_detections_for_source(
        host,
        "source.png",
        [det],
    )

    assert det.attributes["source_geometry_hash"] == geometry_hash
    assert det.attributes["source_gt_hash"] != gt_hash


def test_plate_generator_carries_source_identity_and_semantic_hashes(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (80, 40), (20, 40, 60)).save(image_path)

    det = plate(
        {
            "plate_annotation_id": "plate-ann-fixed",
            "ground_truth_text": "ABC123",
            "ground_truth_source": "manual_z2",
            "source_geometry_hash": "g" * 64,
            "source_gt_hash": "t" * 64,
        }
    )
    ann = ImageAnnotation(
        image_path.name,
        80,
        40,
        [det],
    )

    generator = PlateGenerator(tmp_path / "pz1")
    results = generator.generate_from_annotations(
        image_path,
        ann,
        rectify=False,
    )

    assert len(results) == 1
    record = generator.metadata["plate_000000"]

    assert str(record["source_image_id"]).startswith("img-sha256-")
    assert len(record["source_file_sha256"]) == 64
    assert record["source_annotation_id"] == "plate-ann-fixed"
    assert record["source_geometry_hash"] == "g" * 64
    assert record["source_gt_hash"] == "t" * 64
    assert record["ground_truth_text"] == "ABC123"

def test_gt_only_xml_change_reuses_crop_but_marks_gt_metadata_stale(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    xml_path = write_xml(tmp_path / "annotations.xml", "ABC123")

    before_host = make_host(xml_path, images)
    manifest = {
        "kind": "pz1_plate_extraction",
        "status": "completed",
        "plate_count": 1,
        "source": current_extract_source_payload_with_signature(before_host),
    }
    assert extract_manifest_matches_current_source(before_host, manifest)

    write_xml(xml_path, "ABC124")
    after_host = make_host(xml_path, images)

    freshness = extract_manifest_freshness(after_host, manifest)

    assert freshness["geometry_current"] is True
    assert freshness["recrop_required"] is False
    assert freshness["gt_current"] is False
    assert freshness["gt_metadata_stale"] is True
    assert extract_manifest_matches_current_source(after_host, manifest)


def test_gt_only_refresh_updates_metadata_without_touching_raw_detection(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    xml_path = write_xml(tmp_path / "annotations.xml", "ABC123")
    host = make_host(xml_path, images)

    preview = tmp_path / "pz1"
    preview.mkdir()
    crop_dir = preview / "images"
    crop_dir.mkdir()
    crop_path = crop_dir / "plate_000000.jpg"
    crop_path.write_bytes(b"crop-bytes-stay-identical")

    manifest = {
        "kind": "pz1_plate_extraction",
        "status": "completed",
        "plate_count": 1,
        "source": current_extract_source_payload_with_signature(host),
    }
    (preview / "extract_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    det = plate(
        {
            "plate_annotation_id": "plate-ann-fixed",
            "ground_truth_text": "ABC123",
            "ground_truth_source": "manual_z2",
        }
    )
    prep_host = SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
    )
    prepare_plate_cut_detections_for_source(
        prep_host,
        "source.png",
        [det],
    )
    old_gt_hash = det.attributes["source_gt_hash"]

    raw_detection = {
        "schema": "alpr.pz2.raw_detection.v1",
        "contract": "gt_blind.v1",
        "prediction_text": "ABC123",
        "characters": [
            {"character": "A", "bbox": [0, 0, 5, 10]},
        ],
    }
    metadata = {
        "plate_000000": {
            "source_annotation_id": "plate-ann-fixed",
            "source_image_id": "img-sha256-placeholder",
            "source_polygon": [
                [10.0, 10.0],
                [60.0, 10.0],
                [60.0, 26.0],
                [10.0, 26.0],
            ],
            "ground_truth_text": "ABC123",
            "ground_truth_source": "manual_z2",
            "source_gt_hash": old_gt_hash,
            "plate_attributes": dict(det.attributes),
            "characters": [],
            "raw_detection": raw_detection,
            "raw_validation": {"status": "perfect", "old": True},
            "gt_assist": {
                "source_gt_hash": old_gt_hash,
                "ground_truth_text": "ABC123",
            },
        }
    }
    (preview / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def atomic_write(path, payload):
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def recalculate(payload):
        for record in payload.values():
            if isinstance(record, dict) and isinstance(
                record.get("raw_detection"),
                dict,
            ):
                record["raw_validation"] = {
                    "status": "needs_fix",
                    "ground_truth_text": record.get(
                        "ground_truth_text"
                    ),
                }
        return payload

    host._atomic_write_json = atomic_write
    host._recalculate_preview_statuses_in_metadata = recalculate

    crop_before = crop_path.read_bytes()
    raw_before = json.dumps(raw_detection, sort_keys=True)

    write_xml(xml_path, "ABC124")
    report = refresh_preview_gt_metadata_from_current_xml(
        host,
        preview,
    )

    assert report["changed"] is True
    assert report["complete"] is True
    assert report["updated_records"] == 1
    assert crop_path.read_bytes() == crop_before

    refreshed = json.loads(
        (preview / "metadata.json").read_text(encoding="utf-8")
    )
    row = refreshed["plate_000000"]
    assert row["ground_truth_text"] == "ABC124"
    assert row["source_gt_hash"] != old_gt_hash
    assert json.dumps(row["raw_detection"], sort_keys=True) == raw_before
    assert row["raw_validation"]["ground_truth_text"] == "ABC124"
    assert row["gt_assist"]["source_gt_hash"] == old_gt_hash

    refreshed_manifest = json.loads(
        (preview / "extract_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    freshness = extract_manifest_freshness(
        host,
        refreshed_manifest,
    )
    assert freshness["gt_current"] is True
    assert freshness["gt_metadata_stale"] is False
def test_gt_refresh_advances_manifest_when_metadata_is_already_current(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    xml_path = write_xml(tmp_path / "annotations.xml", "ABC123")
    host = make_host(xml_path, images)

    preview = tmp_path / "pz1"
    preview.mkdir()
    (preview / "images").mkdir()

    old_manifest = {
        "kind": "pz1_plate_extraction",
        "status": "completed",
        "plate_count": 1,
        "source": current_extract_source_payload_with_signature(host),
    }

    # Simulate a manifest left behind after GT changed while metadata was
    # already refreshed by another path.
    write_xml(xml_path, "ABC124")

    det = plate(
        {
            "plate_annotation_id": "plate-ann-fixed",
            "ground_truth_text": "ABC124",
            "ground_truth_source": "manual_z2",
        }
    )
    prep_host = SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
    )
    prepare_plate_cut_detections_for_source(
        prep_host,
        "source.png",
        [det],
    )

    metadata = {
        "plate_000000": {
            "source_annotation_id": "plate-ann-fixed",
            "source_image_id": "img-sha256-placeholder",
            "source_polygon": [
                [10.0, 10.0],
                [60.0, 10.0],
                [60.0, 26.0],
                [10.0, 26.0],
            ],
            "ground_truth_text": "ABC124",
            "ground_truth_source": "manual_z2",
            "source_gt_hash": det.attributes["source_gt_hash"],
            "plate_attributes": dict(det.attributes),
            "characters": [],
            "raw_detection": {
                "schema": "alpr.pz2.raw_detection.v1",
                "contract": "gt_blind.v1",
                "prediction_text": "ABC123",
                "characters": [],
            },
        }
    }

    (preview / "extract_manifest.json").write_text(
        json.dumps(old_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (preview / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    host._atomic_write_json = lambda path, payload: Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    host._recalculate_preview_statuses_in_metadata = lambda payload: payload

    report = refresh_preview_gt_metadata_from_current_xml(host, preview)

    assert report["complete"] is True
    assert report["matched_records"] == 1
    refreshed_manifest = json.loads(
        (preview / "extract_manifest.json").read_text(encoding="utf-8")
    )
    assert extract_manifest_freshness(
        host,
        refreshed_manifest,
    )["gt_current"] is True
