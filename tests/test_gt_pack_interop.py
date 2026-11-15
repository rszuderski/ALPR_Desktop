import json
import shutil
from pathlib import Path

from auto_annotation_tool.gt_pack import (
    ALPRGTPack,
    PACK_SCHEMA,
    fingerprint_image,
)


FIXTURES = (
    Path(__file__).parent
    / "fixtures"
    / "alpr_gt_pack_v1"
)
SOURCES = (
    Path(__file__).parent
    / "fixtures"
    / "alpr_gt_pack_v1_sources"
)


def copied_fixture(tmp_path, name):
    source = FIXTURES / f"{name}.alprgt"
    target = tmp_path / f"{name}.alprgt"
    shutil.copytree(source, target)
    return ALPRGTPack.open(target)


def expectation(name):
    path = FIXTURES / f"{name}.alprgt" / "fixture_expectation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_interop_fixtures_validate(tmp_path):
    for path in sorted(FIXTURES.glob("*.alprgt")):
        target = tmp_path / path.name
        shutil.copytree(path, target)
        pack = ALPRGTPack.open(target)
        result = pack.validate(deep=True)
        assert result["ok"] is True, (
            path.name,
            result["issues"],
        )


def test_fixture_manifest_schema():
    for path in sorted(FIXTURES.glob("*.alprgt")):
        manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema"] == PACK_SCHEMA


def test_exact_source_identity_fixture():
    expected = expectation("single_plate")
    source_path = SOURCES / "source_red.ppm"
    fingerprint = fingerprint_image(source_path)
    assert fingerprint["image_id"] == expected["image_id"]
    assert (
        fingerprint["source_file_sha256"]
        == expected["source_file_sha256"]
    )


def test_single_plate_fixture(tmp_path):
    expected = expectation("single_plate")
    pack = copied_fixture(tmp_path, "single_plate")

    assert len(pack.list_images()) == 1
    assert len(pack.list_plates()) == 1

    state = pack.resolve_ground_truth(expected["plate_id"])
    assert state["resolved"] is True
    assert state["conflict"] is False
    assert state["text"] == "ABC123"


def test_multi_plate_fixture_keeps_gt_per_plate(tmp_path):
    expected = expectation("multi_plate")
    pack = copied_fixture(tmp_path, "multi_plate")

    assert len(pack.list_plates()) == 2
    for plate_id, gt in expected["ground_truth"].items():
        state = pack.resolve_ground_truth(plate_id)
        assert state["resolved"] is True
        assert state["text"] == gt


def test_history_fixture_resolves_latest_descendant(tmp_path):
    expected = expectation("gt_history")
    pack = copied_fixture(tmp_path, "gt_history")

    state = pack.resolve_ground_truth(expected["plate_id"])
    assert state["resolved"] is True
    assert state["text"] == "ABC124"
    assert len(state["revision_ids"]) == 1
    assert state["revision_ids"] == expected["revision_heads"]


def test_clear_fixture_is_resolved_without_active_gt(tmp_path):
    expected = expectation("clear_gt")
    pack = copied_fixture(tmp_path, "clear_gt")

    state = pack.resolve_ground_truth(expected["plate_id"])
    assert state["resolved"] is True
    assert state["conflict"] is False
    assert state["operation"] == "clear"
    assert state["has_ground_truth"] is False
    assert state["text"] == ""


def test_gt_conflict_fixture_preserves_both_heads(tmp_path):
    expected = expectation("gt_conflict")
    pack = copied_fixture(tmp_path, "gt_conflict")

    state = pack.resolve_ground_truth(expected["plate_id"])
    assert state["resolved"] is False
    assert state["conflict"] is True
    assert state["values"] == ["ABC124", "ABC125"]
    assert sorted(state["revision_ids"]) == sorted(
        expected["revision_heads"]
    )


def test_geometry_conflict_fixture_preserves_both_heads(tmp_path):
    expected = expectation("geometry_conflict")
    pack = copied_fixture(tmp_path, "geometry_conflict")

    state = pack.resolve_plate_geometry(expected["plate_id"])
    assert state["resolved"] is False
    assert state["conflict"] is True
    assert sorted(state["geometry_ids"]) == sorted(
        expected["geometry_heads"]
    )
