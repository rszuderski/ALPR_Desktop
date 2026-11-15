from __future__ import annotations

import copy
from types import SimpleNamespace

from auto_annotation_tool.gui import z3_goldpack_ui as gold
from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab


def _record(character="A", index=0):
    return {
        "character": character,
        "bbox": [index * 12.0, 0.0, index * 12.0 + 10.0, 20.0],
        "confidence": 1.0,
    }


def _data():
    return {
        "status": "needs_fix",
        "characters": [_record(symbol, index) for index, symbol in enumerate("ABC123")],
        "source_gt_hash": "gt-hash-1",
        "source_gt_revision_ids": ["gt-r1"],
        "source_geometry_revision_ids": ["geom-r1"],
        "ground_truth_text": "ABC123",
        "source_annotation_id": "ann-1",
        "plate_annotation_id": "plate-1",
        "ground_truth_source": "alprgt",
        "raw_detection": {
            "result_hash": "raw-1",
            "characters": [_record()],
        },
        "review_state": {
            "schema": review.REVIEW_SCHEMA,
            "status": review.REVIEW_IN_PROGRESS,
            "source": "raw_detection",
            "raw_result_hash": "raw-1",
            "started_at": "2026-09-22T00:00:00+02:00",
            "approved_at": None,
        },
        "gold_state": {
            "candidate": False,
            "approved": False,
        },
    }


class Host(SimpleNamespace):
    def _characters_to_text(self, chars, data=None):
        # Use the production reading-order helper without constructing Tk UI.
        tab = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
        return tab._characters_to_text(chars, data=data)

    def _derive_preview_status_from_data(self, probe, chars):
        state = probe.get("review_state") or {}
        return "perfect" if state.get("status") == review.REVIEW_APPROVED else "needs_fix"

    def _ensure_plate_source_metadata(self, data, **kwargs):
        data.setdefault("source_info", {}).setdefault("bucket", "auto_preview")
        data.setdefault("gold_state", {})
        return True

    def _get_plate_source_bucket(self, data):
        return "auto_preview"

    def _persist_preview_metadata(self, **kwargs):
        return None

    def _refresh_preview_listbox_row(self, pid):
        return None

    def _refresh_detection_review_controls(self):
        return None

    def _on_preview_select(self, event=None):
        return None

    def _update_preview_edit_status(self, *args, **kwargs):
        return None


def _host(data):
    return Host(preview_metadata={"p1": data}, _preview_active_pid="p1")


def _approve(data):
    result = review.confirm_review_gold(
        _host(data),
        persist=False,
        quiet=True,
    )
    assert result["ok"] is True
    return data


def test_gold_approval_captures_reference_snapshot():
    data = _approve(_data())

    snapshot = data["review_state"]["approved_reference"]

    assert snapshot["gt_hash"] == "gt-hash-1"
    assert snapshot["gt_revision_ids"] == ["gt-r1"]
    assert snapshot["geometry_revision_ids"] == ["geom-r1"]
    assert snapshot["ground_truth_text"] == "ABC123"
    assert snapshot["source_annotation_id"] == "ann-1"
    assert snapshot["plate_annotation_id"] == "plate-1"
    assert review.review_approval_is_current(data) is True


def test_gt_revision_change_invalidates_gold():
    data = _approve(_data())

    data["source_gt_revision_ids"] = ["gt-r2"]

    assert review.review_approval_is_current(data) is False
    assert review.reconcile_review_approval(data) is True
    assert data["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert data["review_state"]["approval_invalidated_reason"] == "reference_changed"
    assert data["status"] == "needs_fix"
    assert data["gold_state"]["candidate"] is False
    assert data["gold_state"]["approved"] is False


def test_geometry_revision_change_invalidates_gold():
    data = _approve(_data())

    data["source_geometry_revision_ids"] = ["geom-r2"]

    assert review.reconcile_review_approval(data) is True
    assert data["review_state"]["status"] == review.REVIEW_IN_PROGRESS


def test_gt_text_or_hash_change_invalidates_gold_even_without_revision_change():
    for key, value in (
        ("ground_truth_text", "XYZ999"),
        ("source_gt_hash", "gt-hash-2"),
    ):
        data = _approve(_data())
        data[key] = value
        assert review.review_approval_is_current(data) is False


def test_new_raw_result_does_not_invalidate_human_gold():
    data = _approve(_data())
    before = copy.deepcopy(data["review_state"]["approved_reference"])

    data["raw_detection"]["result_hash"] = "raw-2"
    data["raw_detection"]["characters"][0]["bbox"][0] = 999.0

    assert review.review_approval_is_current(data) is True
    assert review.reconcile_review_approval(data) is False
    assert data["review_state"]["status"] == review.REVIEW_APPROVED
    assert data["review_state"]["approved_reference"] == before
    assert data["gold_state"]["approved"] is True


def test_gold_export_rejects_stale_approved_reference():
    data = _approve(_data())
    assert gold.is_gold_export_eligible_data(data) is True

    data["source_gt_revision_ids"] = ["gt-r2"]

    assert gold.is_gold_export_eligible_data(data) is False


def test_old_z3gold001_approval_without_snapshot_requires_reapproval():
    data = _data()
    data["status"] = "perfect"
    data["review_state"]["status"] = review.REVIEW_APPROVED
    data["review_state"]["approved_at"] = "2026-09-22T00:01:00+02:00"
    data["gold_state"]["candidate"] = True
    data["gold_state"]["approved"] = True

    assert "approved_reference" not in data["review_state"]
    assert review.review_approval_is_current(data) is False
    assert review.reconcile_review_approval(data) is True
    assert data["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert data["gold_state"]["approved"] is False
