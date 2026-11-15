from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from auto_annotation_tool.character_recognition.plate_generator import PlateGenerator
from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.gt_pack import ALPRGTPack
from auto_annotation_tool.gui.z3_extraction_sources import (
    prepare_plate_cut_detections_for_source,
)


POLYGON = [
    (10.0, 10.0),
    (70.0, 10.0),
    (70.0, 30.0),
    (10.0, 30.0),
]


def detection(polygon=None):
    pts = list(polygon or POLYGON)
    return Detection(
        label="plate",
        confidence=1.0,
        bbox=(
            min(p[0] for p in pts),
            min(p[1] for p in pts),
            max(p[0] for p in pts),
            max(p[1] for p in pts),
        ),
        polygon=pts,
        attributes={
            "plate_annotation_id": "plate-ann-provenance",
            "ground_truth_text": "ABC123",
            "ground_truth_source": "manual_z2",
        },
    )


def prepare(det):
    host = SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
    )
    prepare_plate_cut_detections_for_source(
        host,
        "source.png",
        [det],
    )


def test_pz1_resolves_revision_ids_only_for_matching_pack_state(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (100, 50), (40, 60, 80)).save(image_path)

    pack_path = tmp_path / "source_gt.alprgt"
    pack = ALPRGTPack.create(pack_path)
    image = pack.add_image(image_path)
    pack.ensure_plate(
        image_id=image["image_id"],
        polygon=POLYGON,
        plate_id="plate-ann-provenance",
    )
    gt_revision = pack.set_ground_truth(
        "plate-ann-provenance",
        "ABC123",
        source="manual_z2",
    )
    geometry = pack.resolve_plate_geometry(
        "plate-ann-provenance"
    )

    det = detection()
    prepare(det)
    ann = ImageAnnotation(
        image_path.name,
        100,
        50,
        [det],
    )

    generator = PlateGenerator(
        tmp_path / "pz1",
        gt_pack_paths=[pack_path],
    )
    results = generator.generate_from_annotations(
        image_path,
        ann,
        rectify=False,
    )
    assert len(results) == 1

    row = generator.metadata["plate_000000"]
    assert row["source_geometry_revision_id"] == (
        geometry["geometry_ids"][0]
    )
    assert row["source_geometry_revision_ids"] == (
        geometry["geometry_ids"]
    )
    assert row["source_gt_revision_id"] == (
        gt_revision["revision_id"]
    )
    assert row["source_gt_revision_ids"] == [
        gt_revision["revision_id"]
    ]


def test_geometry_revision_is_not_claimed_for_mismatched_polygon(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (100, 50), (40, 60, 80)).save(image_path)

    pack_path = tmp_path / "source_gt.alprgt"
    pack = ALPRGTPack.create(pack_path)
    image = pack.add_image(image_path)
    pack.ensure_plate(
        image_id=image["image_id"],
        polygon=POLYGON,
        plate_id="plate-ann-provenance",
    )
    gt_revision = pack.set_ground_truth(
        "plate-ann-provenance",
        "ABC123",
        source="manual_z2",
    )

    moved = [
        (12.0, 10.0),
        (72.0, 10.0),
        (72.0, 30.0),
        (12.0, 30.0),
    ]
    det = detection(moved)
    prepare(det)
    ann = ImageAnnotation(
        image_path.name,
        100,
        50,
        [det],
    )

    generator = PlateGenerator(
        tmp_path / "pz1",
        gt_pack_paths=[pack_path],
    )
    generator.generate_from_annotations(
        image_path,
        ann,
        rectify=False,
    )

    row = generator.metadata["plate_000000"]
    assert row["source_geometry_revision_id"] is None
    assert row["source_geometry_revision_ids"] == []
    assert row["source_gt_revision_id"] == (
        gt_revision["revision_id"]
    )
