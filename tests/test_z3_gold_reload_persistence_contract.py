from __future__ import annotations

import json
from pathlib import Path

from auto_annotation_tool.gui import z3_review_runtime as review


ROOT = Path(__file__).resolve().parents[1]


def _approved_data():
    data = {
        "status": "perfect",
        "characters": [
            {
                "character": "A",
                "bbox": [0.0, 0.0, 10.0, 20.0],
            }
        ],
        "source_gt_hash": "gt-hash-1",
        "source_gt_revision_ids": ["gt-r1"],
        "source_geometry_revision_ids": ["geom-r1"],
        "ground_truth_text": "ABC123",
        "review_state": {
            "schema": review.REVIEW_SCHEMA,
            "status": review.REVIEW_APPROVED,
            "approved_at": "2026-09-22T09:00:00+02:00",
            "approved_by": "human",
        },
        "gold_state": {
            "candidate": True,
            "approved": True,
        },
    }
    data["review_state"]["approved_reference"] = (
        review.build_review_reference_snapshot(data)
    )
    return data


def test_valid_gold_survives_json_round_trip():
    data = _approved_data()
    loaded = json.loads(json.dumps(data))

    assert review.review_approval_is_current(loaded) is True
    assert review.reconcile_review_gold_integrity(loaded) is False
    assert loaded["review_state"]["status"] == review.REVIEW_APPROVED
    assert loaded["gold_state"]["approved"] is True
    assert loaded["status"] == "perfect"


def test_old_approved_record_without_reference_is_reopened():
    data = _approved_data()
    data["review_state"].pop("approved_reference")

    assert review.reconcile_review_gold_integrity(data) is True
    assert data["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert data["review_state"]["approval_invalidated_reason"] == "reference_changed"
    assert data["gold_state"]["candidate"] is False
    assert data["gold_state"]["approved"] is False
    assert data["status"] == "needs_fix"


def test_in_progress_record_cannot_keep_persisted_gold_flags():
    data = _approved_data()
    data["review_state"]["status"] = review.REVIEW_IN_PROGRESS

    assert review.reconcile_review_gold_integrity(data) is True
    assert data["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert "approved_reference" not in data["review_state"]
    assert data["review_state"]["approved_at"] is None
    assert data["gold_state"]["candidate"] is False
    assert data["gold_state"]["approved"] is False
    assert data["status"] == "needs_fix"


def test_approved_review_without_gold_state_approval_fails_closed():
    data = _approved_data()
    data["gold_state"]["approved"] = False

    assert review.reconcile_review_gold_integrity(data) is True
    assert data["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert (
        data["review_state"]["approval_invalidated_reason"]
        == "approval_state_incomplete"
    )
    assert data["gold_state"]["approved"] is False
    assert data["status"] == "needs_fix"


def test_malformed_review_state_fails_closed():
    data = _approved_data()
    data["review_state"]["status"] = "mystery"

    assert review.reconcile_review_gold_integrity(data) is True
    assert data["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert (
        data["review_state"]["approval_invalidated_reason"]
        == "invalid_review_state"
    )
    assert data["gold_state"]["approved"] is False
    assert data["status"] == "needs_fix"


def test_legacy_record_without_review_state_is_untouched():
    data = {
        "status": "perfect",
        "characters": [{"character": "A", "bbox": [0, 0, 10, 20]}],
        "gold_state": {"candidate": True, "approved": True},
    }
    before = json.loads(json.dumps(data))

    assert review.reconcile_review_gold_integrity(data) is False
    assert data == before


def test_quiet_load_wires_lightweight_integrity_repair_and_persists_it():
    source = (
        ROOT / "auto_annotation_tool/gui/z3_preview_ui.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def load_preview_data(")
    end = source.index("\ndef on_preview_select(", start)
    body = source[start:end]

    assert "review_contract_changed = False" in body
    assert "self._reconcile_review_gold_integrity(d)" in body
    assert "review_contract_changed = True" in body
    assert "if changed and (not quiet or review_contract_changed):" in body

    ensure_pos = body.index("self._ensure_plate_source_metadata(")
    reconcile_pos = body.index("self._reconcile_review_gold_integrity(d)")
    assert ensure_pos < reconcile_pos
