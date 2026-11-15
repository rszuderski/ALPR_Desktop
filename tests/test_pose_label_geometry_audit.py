"""Audit math is independent of the repaired application geometry implementation."""
import json
from pathlib import Path

import pytest
from PIL import Image

from audit_mt_pose_geometry import Receipt, audit_dataset, audit_xml, classify_quad, inspect_label_row, legacy_dataset_order


@pytest.mark.parametrize("points, expected", [
    ([(0, 0), (40, 0), (50, 20), (60, 130)], "VALID_POSITIVE"),
    ([(0, 0), (40, 0), (60, 130), (50, 20)], "SELF_INTERSECTING"),
    ([(0, 0), (0, 1), (1, 1), (1, 0)], "REVERSED_WINDING"),
    ([(0, 0), (1, 0), (.2, .2), (0, 1)], "CONCAVE"),
    ([(0, 0), (1, 0), (2, 0), (3, 0)], "DEGENERATE"),
    ([(0, 0), (0, 0), (1, 1), (1, 0)], "DUPLICATE"),
    ([(0, 0), (float("nan"), 0), (1, 1), (0, 1)], "NONFINITE"),
    ([(0, 0), (1, 0), (1, 1)], "KEYPOINTS_MISSING"),
])
def test_independent_statuses(points, expected):
    assert classify_quad(points) == expected


def test_bounds_bbox_and_semantics_are_distinct():
    row = inspect_label_row("0 .5 .5 .4 .4 .1 .1 .9 .1 .9 .9 .1 .9")
    assert row["geometry"] == "VALID_POSITIVE"
    assert row["bbox_mismatch"] is True
    assert row["out_of_bounds"] is False
    assert row["corner_identity"] == "UNKNOWN"
    assert inspect_label_row("0 .5 .5 1 1 -1 0 1 0 1 1 0 1")["out_of_bounds"] is True
    assert inspect_label_row("0 .5 .5 1 1 0 0 1 0 1 1 0 1", (4, 3))["geometry"] == "INVALID_FORMAT"
    visible = inspect_label_row("0 .5 .5 1 1 0 0 .2 1 0 .3 1 1 .4 0 1 .5", (4, 3))
    assert visible["visibility"] == [.2, .3, .4, .5]


def test_receipt_detects_mutation_and_blocks_final_test(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("source")
    receipt = Receipt()
    receipt.record(path)
    assert receipt.verify()["changed"] == []
    path.write_text("changed")
    assert receipt.verify()["changed"] == [str(path)]
    with pytest.raises(ValueError, match="outside"):
        receipt.record(tmp_path / "final_test" / "secret.txt")


def test_dataset_and_xml_expose_export_order_damage_without_modification(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    (root / "data.yaml").write_text("kpt_shape: [4, 2]\nflip_idx: [1, 0, 3, 2]\n")
    image = root / "images/train/example.png"
    Image.new("RGB", (200, 200)).save(image)
    raw = [(0., 0.), (40., 0.), (50., 20.), (60., 130.)]
    bad = legacy_dataset_order(raw)
    label = root / "labels/train/example.txt"
    label.write_text("0 .15 .325 .3 .65 " + " ".join(str(value / 200) for point in bad for value in point))
    xml = tmp_path / "annotations.xml"
    xml.write_text('<annotations><image name="example.png" width="200" height="200"><polygon label="plate" points="0,0;40,0;50,20;60,130"/></image></annotations>')
    receipt = Receipt()
    result, rows = audit_dataset(root, "test", receipt)
    assert result["splits"]["train"]["counts"]["SELF_INTERSECTING"] == 1
    assert result["splits"]["train"]["counts"]["corner_identity_UNKNOWN"] == 1
    assert result["augmentation_records"] == "UNKNOWN"
    comparison = audit_xml(xml, rows, receipt)
    assert comparison["label_matching"]["VALID_POSITIVE__LABEL_SELF_INTERSECTING"] == 1
    assert comparison["label_matching"]["legacy_exporter_exact_reproduction"] == 1
    assert receipt.verify()["changed"] == []
