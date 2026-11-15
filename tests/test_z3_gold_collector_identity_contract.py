from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from auto_annotation_tool.gui import z3_goldpack_ui as gold
from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab


def _approved_crop(crop_id, **extra):
    data = {
        "crop_id": crop_id,
        "crop_identity_sha256": hashlib.sha256(crop_id.encode()).hexdigest(),
        "ground_truth_text": "ABC123",
        "status": "perfect",
        "fusion_strategy": "yolo_exact",
        "source_info": {"bucket": "auto_preview"},
        "characters": [
            {"character": symbol, "bbox": [i * 12, 0, i * 12 + 10, 20],
             "box_source": "yolo_box", "sign_source": "yolo_symbol"}
            for i, symbol in enumerate("ABC123")
        ],
        "gold_state": {"approved": True, "candidate": True},
    }
    data.update(extra)
    data["review_state"] = {
        "schema": review.REVIEW_SCHEMA,
        "status": review.REVIEW_APPROVED,
        "approved_reference": review.build_review_reference_snapshot(data),
    }
    assert gold.is_gold_export_eligible_data(data)
    return data


def _run(root, records, *, missing_images=()):
    root.mkdir()
    (root / "images").mkdir()
    path = root / "metadata.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    for pid in records:
        if pid not in missing_images:
            (root / "images" / (pid + ".jpg")).write_bytes(b"image")
    return path


def _host(paths, metadata):
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.preview_metadata = metadata
    host._loaded_meta_path = paths[0]
    host.preview_dir_var = SimpleNamespace(get=lambda: str(paths[0].parent))
    host._get_gold_export_meta_candidates = lambda: paths
    return host


@pytest.mark.parametrize("prepare_records", [False, True])
def test_35_distinct_imported_crops_with_missing_legacy_source_are_all_collected(tmp_path, prepare_records):
    records = {f"plate_{i:06d}": _approved_crop(f"CROP-{i:032X}") for i in range(35)}
    # All 35 used to share the legacy key u_0 despite different registry identities.
    assert all(not row.get("source_image") and not row.get("source_bbox") for row in records.values())
    before = copy.deepcopy(records)
    path = _run(tmp_path / "run", records)
    host = _host([path], records)

    entries, *_ = gold.collect_gold_export_plate_candidates(host, set(), set(), prepare_records=prepare_records)

    assert {entry["pid"] for entry in entries} == set(records)
    assert len(entries) == 35
    assert len({entry["data"]["crop_id"] for entry in entries}) == 35
    assert sum(host._count_exportable_characters_in_data(entry["data"]) for entry in entries) == 210
    assert records == before
    counts = gold.build_merged_gold_export_counts(host)
    assert counts["selected_plate_count"] == 35
    assert counts["selected_char_count"] == 210


@pytest.mark.parametrize("prepare_records", [False, True])
def test_copies_of_one_logical_crop_in_two_runs_are_deduplicated(tmp_path, prepare_records):
    first = {"p1": _approved_crop("CROP-SAME", source_image="old.jpg", artifact_id="ART-ONE")}
    second = {"renamed": _approved_crop("CROP-SAME", source_image="new.jpg", artifact_id="ART-TWO")}
    paths = [_run(tmp_path / "first", first), _run(tmp_path / "second", second)]

    entries, *_ = gold.collect_gold_export_plate_candidates(_host(paths, first), set(), set(), prepare_records=prepare_records)

    assert len(entries) == 1
    assert entries[0]["data"]["crop_id"] == "CROP-SAME"


def test_missing_image_does_not_consume_the_identity_of_a_later_valid_copy(tmp_path):
    first = {"p1": _approved_crop("CROP-SAME")}
    second = {"p2": _approved_crop("CROP-SAME")}
    paths = [_run(tmp_path / "first", first, missing_images={"p1"}), _run(tmp_path / "second", second)]

    entries, *_ = gold.collect_gold_export_plate_candidates(_host(paths, first), set(), set(), prepare_records=False)

    assert [entry["pid"] for entry in entries] == ["p2"]


@pytest.mark.parametrize("extra", [
    {"source_image": "same.jpg", "source_bbox": [11, 0, 30, 20]},
    {"mobile_acquisition": {"archive_sha256": "archive", "group_key": "group"}},
    {"artifact_id": "same-artifact", "crop_identity_sha256": "a" * 64},
])
def test_distinct_crop_ids_win_over_auxiliary_source_and_artifact_keys(tmp_path, extra):
    records = {"p1": _approved_crop("CROP-ONE", **extra), "p2": _approved_crop("CROP-TWO", **extra)}
    path = _run(tmp_path / "run", records)

    entries, *_ = gold.collect_gold_export_plate_candidates(_host([path], records), set(), set(), prepare_records=False)

    assert {entry["data"]["crop_id"] for entry in entries} == {"CROP-ONE", "CROP-TWO"}


def test_legacy_mobile_copies_keep_archive_and_group_deduplication(tmp_path):
    records = {pid: _approved_crop("unused", source_image=pid + ".jpg",
        mobile_acquisition={"archive_sha256": "archive", "group_key": "group"}) for pid in ["p1", "p2"]}
    for row in records.values():
        row.pop("crop_id")
        row.pop("crop_identity_sha256")
    path = _run(tmp_path / "run", records)

    entries, *_ = gold.collect_gold_export_plate_candidates(_host([path], records), set(), set(), prepare_records=False)

    assert len(entries) == 1


def test_legacy_source_copies_keep_source_and_bbox_bucket_deduplication(tmp_path):
    records = {pid: _approved_crop("unused", source_image="legacy.jpg", source_bbox=[x, 0, 30, 20])
               for pid, x in [("p1", 11), ("p2", 19)]}
    for row in records.values():
        row.pop("crop_id")
        row.pop("crop_identity_sha256")
    path = _run(tmp_path / "run", records)

    entries, *_ = gold.collect_gold_export_plate_candidates(_host([path], records), set(), set(), prepare_records=False)

    assert len(entries) == 1


def test_identity_hash_fallback_deduplicates_copies_without_crop_ids(tmp_path):
    records = {pid: _approved_crop("unused", crop_identity_sha256="a" * 64, source_image=pid + ".jpg")
               for pid in ["p1", "p2"]}
    for row in records.values():
        row.pop("crop_id")
    path = _run(tmp_path / "run", records)

    entries, *_ = gold.collect_gold_export_plate_candidates(_host([path], records), set(), set(), prepare_records=False)

    assert len(entries) == 1


def test_distinct_identity_hashes_keep_crops_without_crop_ids(tmp_path):
    records = {pid: _approved_crop("unused", crop_identity_sha256=sha) for pid, sha in [("p1", "a" * 64), ("p2", "b" * 64)]}
    for row in records.values():
        row.pop("crop_id")
    path = _run(tmp_path / "run", records)

    entries, *_ = gold.collect_gold_export_plate_candidates(_host([path], records), set(), set(), prepare_records=False)

    assert len(entries) == 2


@pytest.mark.parametrize("invalid_hash", ["", "not-a-hash", "x" * 64])
def test_invalid_identity_hash_does_not_replace_the_legacy_key(invalid_hash):
    assert gold.build_gold_export_plate_unique_key({"crop_identity_sha256": invalid_hash,
        "source_image": "legacy.jpg", "source_bbox": [19, 0, 30, 20]}) == "legacy.jpg_1"
