from __future__ import annotations

import copy
from types import SimpleNamespace

from auto_annotation_tool.gui import z3_preview_metadata_runtime as metadata
from auto_annotation_tool.gui import z3_review_runtime as review


class Host(SimpleNamespace):
    def _sort_character_records_by_x(self, chars, data=None):
        return list(chars or [])

    def _annotate_preview_character_reading_positions(self, chars, data=None):
        return list(chars or [])

    def _update_preview_plate_layout_metadata(self, data, chars):
        return None

    def _ensure_plate_source_metadata(self, data, **kwargs):
        data.setdefault("source_info", {}).setdefault("bucket", "auto_preview")
        data.setdefault("gold_state", {})
        return True

    def _get_plate_source_bucket(self, data):
        return str((data.get("source_info") or {}).get("bucket", "auto_preview"))

    def _characters_to_text(self, chars, data=None):
        return "".join(
            str(rec.get("character") or "")
            for rec in list(chars or [])
            if isinstance(rec, dict)
        )

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


def _base_data():
    return {
        "ground_truth_text": "A",
        "characters": [],
        "raw_detection": {
            "schema": "alpr.pz2.raw_detection.v1",
            "contract": "gt_blind.v1",
            "result_hash": "raw-hash-1",
            "characters": [
                {
                    "character": "A",
                    "bbox": [1.0, 2.0, 10.0, 20.0],
                    "confidence": 0.9,
                    "method": "ocr",
                }
            ],
        },
        "raw_validation": {
            "schema": "alpr.pz2.raw_validation.v1",
            "status": "perfect",
        },
        "gt_assist": {
            "schema": "alpr.pz2.gt_assist.v1",
            "status": "ready",
            "operations": [{"op": "example"}],
        },
    }


def test_start_review_is_deep_copy_and_preserves_frozen_layers():
    data = _base_data()
    host = Host(preview_metadata={"p1": data}, _preview_active_pid="p1")
    raw_before = copy.deepcopy(data["raw_detection"])
    validation_before = copy.deepcopy(data["raw_validation"])
    assist_before = copy.deepcopy(data["gt_assist"])

    result = review.start_review_from_raw(host, persist=False, quiet=True)

    assert result["ok"] is True
    assert data["review_state"]["status"] == "in_progress"
    assert data["status"] == "needs_fix"
    assert data["characters"] == data["raw_detection"]["characters"]
    assert data["characters"] is not data["raw_detection"]["characters"]
    assert data["characters"][0] is not data["raw_detection"]["characters"][0]

    data["characters"][0]["bbox"][0] = 999.0
    assert data["raw_detection"] == raw_before
    assert data["raw_validation"] == validation_before
    assert data["gt_assist"] == assist_before


def test_start_review_does_not_overwrite_existing_review():
    data = _base_data()
    data["characters"] = [{"character": "X", "bbox": [0, 0, 4, 4]}]
    before = copy.deepcopy(data["characters"])
    host = Host(preview_metadata={"p1": data}, _preview_active_pid="p1")

    result = review.start_review_from_raw(host, persist=False, quiet=True)

    assert result["ok"] is False
    assert result["reason"] == "review_exists"
    assert data["characters"] == before


def test_review_in_progress_blocks_automatic_perfect():
    data = {
        "review_state": {
            "schema": review.REVIEW_SCHEMA,
            "status": "in_progress",
        }
    }
    status = metadata._derive_preview_status_from_data(
        SimpleNamespace(),
        data,
        [{"character": "A", "bbox": [0, 0, 4, 4]}],
    )
    assert status == "needs_fix"


def test_explicit_confirm_promotes_review_to_gold_without_touching_raw():
    data = _base_data()
    host = Host(preview_metadata={"p1": data}, _preview_active_pid="p1")
    assert review.start_review_from_raw(host, persist=False, quiet=True)["ok"]

    raw_before = copy.deepcopy(data["raw_detection"])
    validation_before = copy.deepcopy(data["raw_validation"])
    assist_before = copy.deepcopy(data["gt_assist"])

    def derive(probe, chars):
        state = probe.get("review_state") or {}
        return "perfect" if state.get("status") == "approved" else "needs_fix"

    host._derive_preview_status_from_data = derive
    result = review.confirm_review_gold(host, persist=False, quiet=True)

    assert result["ok"] is True
    assert data["review_state"]["status"] == "approved"
    assert data["review_state"]["approved_by"] == "human"
    assert data["status"] == "perfect"
    assert data["gold_state"]["candidate"] is True
    assert data["gold_state"]["approved"] is True
    assert data["raw_detection"] == raw_before
    assert data["raw_validation"] == validation_before
    assert data["gt_assist"] == assist_before


def test_manual_edit_reopens_approved_gold():
    data = _base_data()
    data["review_state"] = {
        "schema": review.REVIEW_SCHEMA,
        "status": "approved",
        "source": "raw_detection",
        "raw_result_hash": "raw-hash-1",
        "started_at": "2026-09-22T00:00:00+02:00",
        "approved_at": "2026-09-22T00:01:00+02:00",
    }
    data["gold_state"] = {"candidate": True, "approved": True}

    review.mark_review_edit_started(SimpleNamespace(), data)

    assert data["review_state"]["status"] == "in_progress"
    assert data["review_state"]["approved_at"] is None
    assert data["gold_state"]["candidate"] is False
    assert data["gold_state"]["approved"] is False
