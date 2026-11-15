from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest
from PIL import Image

from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui import z3_plate_gt_runtime as gt
from auto_annotation_tool.gui import z3_list_review as bulk
from auto_annotation_tool.gt_pack import ALPRGTPack
from test_z3_edit_gt_runtime import plate, quality_host, records


def make_host(tmp_path, text="AX37", number="AT37"):
    data = plate(text, number, editing=False)
    source = tmp_path / "not_a_registration.jpg"
    Image.new("RGB", (240, 64), "white").save(source)
    data.update(source_image=str(source), source_image_name=source.name,
                source_polygon=[[0, 0], [239, 0], [239, 63], [0, 63]])
    host = quality_host(data)
    host.preview_dir_var = SimpleNamespace(get=lambda: str(tmp_path))
    return host, data


@pytest.mark.parametrize("text,number,expected", [("XATZ37Y", "AT37", "AT37"),
                                                    ("AX37", "AT37", "AT37"),
                                                    ("AT3", "AT37", "AT3"),
                                                    ("AX37", "", "AX37")])
def test_detection_materializes_working_once_without_changing_raw(tmp_path, text, number, expected):
    host, data = make_host(tmp_path, text, number)
    raw = deepcopy(data["raw_detection"])
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="plate")
    assert host._characters_to_text(data["characters"], data=data) == expected
    assert data["raw_detection"] == raw
    assert data["review_state"]["status"] == "in_progress"
    assert not data["gold_state"]["approved"]
    frozen = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data)
    assert frozen == data


@pytest.mark.parametrize("approved", [False, True])
def test_rerun_preserves_manual_characters_and_approval(tmp_path, approved):
    host, data = make_host(tmp_path)
    review.prepare_working_annotation_from_raw(host, data)
    data["characters"][0]["bbox"][0] += 1
    if approved:
        assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    protected = deepcopy({k: v for k, v in data.items() if k != "raw_detection"})
    data["raw_detection"] = {"contract": "gt_blind.v1", "characters": records("NEW")}
    assert not review.prepare_working_annotation_from_raw(host, data)
    assert {k: v for k, v in data.items() if k != "raw_detection"} == protected
    assert host._review_approval_is_current(data) is approved


def test_approval_never_trims_or_relabels_existing_characters(tmp_path):
    host, data = make_host(tmp_path)
    data["ground_truth_source"] = "manual_z2"
    review.prepare_working_annotation_from_raw(host, data)
    data["characters"].append(records("X")[-1])
    before = deepcopy(data["characters"])
    assert not review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert data["characters"] == before
    assert not bulk.change_plate_approval(host, "plate", True)["ok"]
    assert data["characters"] == before


def test_bulk_approval_does_not_create_an_annotation(tmp_path):
    host, data = make_host(tmp_path)
    before = deepcopy(data)
    assert bulk.change_plate_approval(host, "plate", True)["reason"] == "working_annotation_missing"
    assert data == before


def test_approval_without_gt_creates_canonical_revision_from_corrected_symbols(tmp_path):
    host, data = make_host(tmp_path, number="")
    review.prepare_working_annotation_from_raw(host, data)
    data["characters"][1]["character"] = "T"
    raw = deepcopy(data["raw_detection"])
    chars = deepcopy(data["characters"])
    assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert data["ground_truth_text"] == "AT37"
    assert data["ground_truth_source"] == "manual_z3"
    assert data["source_gt_revision_id"]
    assert data["characters"] == chars and data["raw_detection"] == raw
    pack = ALPRGTPack.open(tmp_path / "current_work.alprgt")
    assert pack.resolve_ground_truth("annotation-1")["text"] == "AT37"
    restored = json.loads(json.dumps(data))
    assert not gt.refresh_working_ground_truth(host, {"plate": restored})
    assert host._review_approval_is_current(restored)


def test_gt_edit_revises_canonical_pack_and_invalidates_old_approval(tmp_path):
    host, data = make_host(tmp_path, number="")
    review.prepare_working_annotation_from_raw(host, data)
    assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    old = deepcopy(data)
    raw = deepcopy(data["raw_detection"])
    gt.save_plate_ground_truth(host, data, "AT37")
    assert data["source_gt_revision_id"] != old["source_gt_revision_id"]
    assert data["ground_truth_text"] == "AT37"
    assert host._characters_to_text(data["characters"]) == "AT37"
    assert data["raw_detection"] == raw
    assert not host._review_approval_is_current(data)
    assert gt.refresh_working_ground_truth(host, {"old": old})
    assert not host._review_approval_is_current(old)
    assert old["ground_truth_text"] == "AT37"


def test_failed_gt_write_leaves_review_unapproved(tmp_path, monkeypatch):
    host, data = make_host(tmp_path, number="")
    review.prepare_working_annotation_from_raw(host, data)
    before = deepcopy(data)
    monkeypatch.setattr(gt, "save_plate_ground_truth", Mock(side_effect=OSError("read-only")))
    assert review.confirm_review_gold(host, persist=False, quiet=True)["reason"] == "gt_write_failed"
    assert data == before


def test_changed_gt_does_not_override_labels_on_approval(tmp_path):
    host, data = make_host(tmp_path)
    data["ground_truth_source"] = "manual_z2"
    review.prepare_working_annotation_from_raw(host, data)
    data["characters"][0]["character"] = "Z"
    before = deepcopy(data["characters"])
    assert not review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert data["characters"] == before


@pytest.mark.parametrize("box", [[0, 0, float("inf"), 40], [10, 0, 5, 40], [0, 0, float("nan"), 40]])
def test_invalid_geometry_cannot_author_gt_or_be_approved(tmp_path, box):
    host, data = make_host(tmp_path, number="")
    review.prepare_working_annotation_from_raw(host, data)
    data["characters"][0]["bbox"] = box
    assert not review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert not (tmp_path / "current_work.alprgt").exists()


def test_pz3_checks_current_gt_before_exporting_an_older_approved_record(tmp_path):
    from auto_annotation_tool.gui.z3_goldpack_ui import collect_gold_export_plate_candidates
    host, data = make_host(tmp_path, number="")
    review.prepare_working_annotation_from_raw(host, data)
    assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    old = deepcopy(data)
    gt.save_plate_ground_truth(host, data, "AT37")
    host.preview_metadata = {"plate": old}
    (tmp_path / "images").mkdir()
    Image.new("RGB", (240, 64)).save(tmp_path / "images/plate.jpg")
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps(host.preview_metadata), encoding="utf-8")
    host._loaded_meta_path = meta
    host._get_gold_export_meta_candidates = lambda: [meta]
    assert not collect_gold_export_plate_candidates(host, set())[0]
    assert old["ground_truth_text"] == "AT37"
    assert not host._review_approval_is_current(old)


def test_gt_edit_inherits_imported_revision_without_modifying_source_pack(tmp_path):
    import hashlib
    host, data = make_host(tmp_path)
    original = ALPRGTPack.create(tmp_path / "imported.alprgt")
    existing = original.upsert_z2_plate(image_path=data["source_image"], polygon=data["source_polygon"],
                                        plate_annotation_id=data["source_annotation_id"], ground_truth_text="AT37")
    old_id = existing["revision"]["revision_id"]
    data["source_gt_revision_ids"] = [old_id]
    before = {str(path.relative_to(original.root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in original.root.rglob("*") if path.is_file()}
    revision = gt.save_plate_ground_truth(host, data, "AT38")
    working = ALPRGTPack.open(tmp_path / "current_work.alprgt")
    assert working.get_revision(old_id)
    assert old_id in revision["parents"]
    assert working.resolve_ground_truth(data["source_annotation_id"])["text"] == "AT38"
    after = {str(path.relative_to(original.root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in original.root.rglob("*") if path.is_file()}
    assert before == after


def test_rerun_preserves_legacy_empty_manual_annotation(tmp_path):
    host, data = make_host(tmp_path)
    data["fusion_strategy"] = "manual_correction"
    before = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data)
    assert data == before
